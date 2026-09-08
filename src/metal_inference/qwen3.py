"""Qwen3 embedding forward assembled entirely from this project's Metal kernels."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ClosedError, InferenceError, ManifestError, UnsupportedProfileError
from .metal import Buffer, MetalRuntime
from .weights import SafeTensors, read_artifact, read_json


class Qwen3Backend:
    hidden = 1024
    intermediate = 3072
    layers = 28
    heads = 16
    kv_heads = 8
    head_dim = 128
    vocab_size = 151669
    max_padded_tokens = 4096

    def __init__(self, model_dir: str) -> None:
        config = read_json(model_dir, "config.json")
        expected: dict[str, Any] = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": self.hidden,
            "intermediate_size": self.intermediate,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_scaling": None,
            "use_sliding_window": False,
            "quantization": {"bits": 4, "group_size": 64},
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise UnsupportedProfileError()
        self.eps = float(config["rms_norm_eps"])
        self.theta = float(config["rope_theta"])
        snapshots = SafeTensors(read_artifact(model_dir, "model.safetensors"))
        specs: dict[str, tuple[tuple[int, ...], str]] = {}

        def norm(prefix: str, dims: int) -> None:
            specs[prefix + ".weight"] = ((dims,), "BF16")

        def quant(prefix: str, outputs: int, inputs: int) -> None:
            specs[prefix + ".weight"] = ((outputs, inputs // 8), "U32")
            for kind in ("scales", "biases"):
                specs[prefix + "." + kind] = ((outputs, inputs // 64), "BF16")

        quant("model.embed_tokens", self.vocab_size, self.hidden)
        norm("model.norm", self.hidden)
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}"
            norm(prefix + ".input_layernorm", self.hidden)
            norm(prefix + ".post_attention_layernorm", self.hidden)
            for name, outputs in (
                ("q", self.heads * self.head_dim),
                ("k", self.kv_heads * self.head_dim),
                ("v", self.kv_heads * self.head_dim),
            ):
                quant(prefix + f".self_attn.{name}_proj", outputs, self.hidden)
            quant(prefix + ".self_attn.o_proj", self.hidden, self.heads * self.head_dim)
            norm(prefix + ".self_attn.q_norm", self.head_dim)
            norm(prefix + ".self_attn.k_norm", self.head_dim)
            quant(prefix + ".mlp.gate_proj", self.intermediate, self.hidden)
            quant(prefix + ".mlp.up_proj", self.intermediate, self.hidden)
            quant(prefix + ".mlp.down_proj", self.hidden, self.intermediate)
        if set(specs) != set(snapshots.tensors):
            raise ManifestError()
        # Validate every tensor before creating any GPU allocation.
        for name, (shape, dtype) in specs.items():
            snapshots.view(name, shape=shape, dtype=dtype)
        self.runtime = MetalRuntime()
        self.weights: dict[str, Buffer] = {}
        self._closed = False
        try:
            for name, (shape, dtype) in specs.items():
                data = snapshots.view(name, shape=shape, dtype=dtype)
                self.weights[name] = self.runtime.buffer(data.nbytes, data)
        except BaseException:
            self.close()
            raise

    def _quant(self, prefix: str) -> list[Buffer]:
        return [self.weights[prefix + "." + suffix] for suffix in ("weight", "scales", "biases")]

    def forward(
        self, ids: NDArray[np.uint32], lengths: NDArray[np.uint32], *, dimensions: int
    ) -> NDArray[np.float32]:
        if self._closed:
            raise ClosedError()
        if (
            ids.dtype != np.uint32
            or lengths.dtype != np.uint32
            or ids.ndim != 2
            or not 1 <= ids.shape[0] <= 32
            or not 1 <= ids.shape[1] <= 512
            or ids.size > self.max_padded_tokens
            or lengths.shape != (ids.shape[0],)
            or not np.all((lengths > 0) & (lengths <= ids.shape[1]))
            or np.any(ids >= self.vocab_size)
            or not 32 <= dimensions <= self.hidden
        ):
            raise InferenceError()
        rt = self.runtime
        batch, seq = ids.shape
        tokens = batch * seq
        allocated: list[Buffer] = []

        def new(size: int, data: NDArray[Any] | None = None) -> Buffer:
            buffer = rt.buffer(size, data)
            allocated.append(buffer)
            return buffer

        def norm(x: Buffer, y: Buffer, name: str, rows: int, cols: int) -> None:
            rt._dispatch(
                "rms_norm",
                [x, self.weights[name + ".weight"], y],
                threads=rows * 32,
                group_size=32,
                cols=cols,
                eps=self.eps,
            )

        def linear(x: Buffer, y: Buffer, name: str, outputs: int, inputs: int) -> None:
            rt._dispatch(
                "linear4",
                [x, *self._quant(name), y],
                threads=((tokens + 3) // 4) * outputs * 32,
                group_size=32,
                rows=tokens,
                cols=outputs,
                k=inputs,
            )

        def residual(x: Buffer, y: Buffer) -> None:
            rt._dispatch("add", [x, y, x], threads=tokens * self.hidden, n=tokens * self.hidden)

        try:
            with rt.command():
                token_buffer = new(ids.nbytes, np.ascontiguousarray(ids))
                length_buffer = new(lengths.nbytes, np.ascontiguousarray(lengths))
                x, normalized, projected = (new(tokens * self.hidden * 4) for _ in range(3))
                q, attended = (new(tokens * self.heads * self.head_dim * 4) for _ in range(2))
                k, v = (new(tokens * self.kv_heads * self.head_dim * 4) for _ in range(2))
                gate, up = (new(tokens * self.intermediate * 4) for _ in range(2))
                output = new(batch * dimensions * 4)
                rt._dispatch(
                    "embedding4",
                    [token_buffer, *self._quant("model.embed_tokens"), x],
                    threads=tokens * self.hidden,
                    n=tokens * self.hidden,
                    cols=self.hidden,
                )
                for layer in range(self.layers):
                    prefix = f"model.layers.{layer}"
                    attn = prefix + ".self_attn"
                    norm(x, normalized, prefix + ".input_layernorm", tokens, self.hidden)
                    linear(normalized, q, attn + ".q_proj", self.heads * self.head_dim, self.hidden)
                    linear(
                        normalized, k, attn + ".k_proj", self.kv_heads * self.head_dim, self.hidden
                    )
                    linear(
                        normalized, v, attn + ".v_proj", self.kv_heads * self.head_dim, self.hidden
                    )
                    for value, heads, name in ((q, self.heads, "q"), (k, self.kv_heads, "k")):
                        norm(value, value, attn + f".{name}_norm", tokens * heads, self.head_dim)
                        count = tokens * heads * self.head_dim // 2
                        rt._dispatch(
                            "rope",
                            [value],
                            threads=count,
                            n=count,
                            heads=heads,
                            seq=seq,
                            dim=self.head_dim,
                            theta=self.theta,
                        )
                    rt._dispatch(
                        "attention",
                        [q, k, v, length_buffer, attended],
                        threads=tokens * self.heads * 32,
                        group_size=32,
                        seq=seq,
                        heads=self.heads,
                        kv_heads=self.kv_heads,
                        dim=self.head_dim,
                        scale=self.head_dim**-0.5,
                    )
                    linear(
                        attended,
                        projected,
                        attn + ".o_proj",
                        self.hidden,
                        self.heads * self.head_dim,
                    )
                    residual(x, projected)
                    norm(x, normalized, prefix + ".post_attention_layernorm", tokens, self.hidden)
                    linear(
                        normalized, gate, prefix + ".mlp.gate_proj", self.intermediate, self.hidden
                    )
                    linear(normalized, up, prefix + ".mlp.up_proj", self.intermediate, self.hidden)
                    rt._dispatch(
                        "silu_gate",
                        [gate, up],
                        threads=tokens * self.intermediate,
                        n=tokens * self.intermediate,
                    )
                    linear(
                        gate, projected, prefix + ".mlp.down_proj", self.hidden, self.intermediate
                    )
                    residual(x, projected)
                norm(x, normalized, "model.norm", tokens, self.hidden)
                rt._dispatch(
                    "pool_project",
                    [normalized, length_buffer, output],
                    threads=batch * 32,
                    group_size=32,
                    seq=seq,
                    cols=self.hidden,
                    dim=dimensions,
                )
            result = rt.read(output, (batch, dimensions))
            if not np.isfinite(result).all() or not np.allclose(
                np.linalg.norm(result, axis=1), 1.0, atol=1e-4, rtol=0.0
            ):
                raise InferenceError()
            return result
        finally:
            for buffer in allocated:
                buffer.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            for buffer in self.weights.values():
                buffer.close()
            self.weights.clear()
            self.runtime.close()
