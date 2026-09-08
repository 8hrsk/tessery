"""Float32 BERT encoder adapter using the shared native Metal runtime."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import config_float, config_int
from .errors import ClosedError, InferenceError, ManifestError, UnsupportedProfileError
from .metal import Buffer, MetalRuntime
from .profiles import ModelProfile
from .weights import SafeTensors, read_artifact, read_json


class BertBackend:
    max_padded_tokens = 4096

    def __init__(self, model_dir: str, profile: ModelProfile) -> None:
        self.profile = profile
        config = read_json(model_dir, "config.json", profile=profile)
        self.hidden = config_int(config, "hidden_size", 32, 4096)
        self.intermediate = config_int(config, "intermediate_size", 32, 16384)
        self.layers = config_int(config, "num_hidden_layers", 1, 64)
        self.heads = config_int(config, "num_attention_heads", 1, 64)
        self.vocab_size = config_int(config, "vocab_size", 5, 200000)
        positions = config_int(config, "max_position_embeddings", profile.max_length, 8192)
        types = config_int(config, "type_vocab_size", 1, 16)
        if (
            profile.architecture != "bert_f32"
            or self.hidden != profile.native_dimensions
            or self.hidden % self.heads
            or not 1 <= self.hidden // self.heads <= 256
            or config.get("model_type") != "bert"
            or config.get("architectures") != ["BertModel"]
            or config.get("hidden_act") != "gelu"
            or config.get("position_embedding_type", "absolute") != "absolute"
            or config.get("is_decoder", False)
            or config.get("add_cross_attention", False)
            or config.get("pruned_heads", {})
        ):
            raise UnsupportedProfileError()
        self.head_dim = self.hidden // self.heads
        self.eps = config_float(config, "layer_norm_eps", 1e-12, 1)
        snapshots = SafeTensors(read_artifact(model_dir, "model.safetensors", profile=profile))
        specs: dict[str, tuple[tuple[int, ...], str]] = {}

        def norm(prefix: str) -> None:
            for suffix in ("weight", "bias"):
                specs[prefix + "." + suffix] = ((self.hidden,), "F32")

        def linear(prefix: str, outputs: int, inputs: int) -> None:
            specs[prefix + ".weight"] = ((outputs, inputs), "F32")
            specs[prefix + ".bias"] = ((outputs,), "F32")

        for name, rows in (
            ("word", self.vocab_size),
            ("position", positions),
            ("token_type", types),
        ):
            specs[f"embeddings.{name}_embeddings.weight"] = ((rows, self.hidden), "F32")
        norm("embeddings.LayerNorm")
        for layer in range(self.layers):
            prefix = f"encoder.layer.{layer}"
            for name in ("query", "key", "value"):
                linear(prefix + ".attention.self." + name, self.hidden, self.hidden)
            linear(prefix + ".attention.output.dense", self.hidden, self.hidden)
            norm(prefix + ".attention.output.LayerNorm")
            linear(prefix + ".intermediate.dense", self.intermediate, self.hidden)
            linear(prefix + ".output.dense", self.hidden, self.intermediate)
            norm(prefix + ".output.LayerNorm")
        # CLS pooling uses the encoder state, not the optional tanh pooler.
        unused: set[str] = set()
        if "pooler.dense.weight" in snapshots.tensors:
            linear("pooler.dense", self.hidden, self.hidden)
            unused.update(("pooler.dense.weight", "pooler.dense.bias"))
        if "embeddings.position_ids" in snapshots.tensors:
            specs["embeddings.position_ids"] = ((1, positions), "I64")
            unused.add("embeddings.position_ids")
        if set(specs) != set(snapshots.tensors):
            raise ManifestError()
        for name, (shape, dtype) in specs.items():
            data = snapshots.view(name, shape=shape, dtype=dtype)
            if dtype == "F32" and not np.isfinite(data.view(np.float32)).all():
                raise ManifestError()
            if dtype == "I64" and not np.array_equal(data.view(np.int64), np.arange(positions)):
                raise ManifestError()
        self.runtime = MetalRuntime()
        self.weights: dict[str, Buffer] = {}
        self._closed = False
        try:
            for name, (shape, dtype) in specs.items():
                if name not in unused:
                    data = snapshots.view(name, shape=shape, dtype=dtype)
                    self.weights[name] = self.runtime.buffer(data.nbytes, data)
        except BaseException:
            self.close()
            raise

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
            or not 2 <= ids.shape[1] <= self.profile.max_length
            or ids.size > self.max_padded_tokens
            or lengths.shape != (ids.shape[0],)
            or not np.all((lengths >= 2) & (lengths <= ids.shape[1]))
            or np.any(ids >= self.vocab_size)
            or type(dimensions) is not int
            or dimensions != self.hidden
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

        def norm(x: Buffer, y: Buffer, name: str) -> None:
            rt._dispatch(
                "layer_norm",
                [x, self.weights[name + ".weight"], self.weights[name + ".bias"], y],
                threads=tokens * 32,
                group_size=32,
                cols=self.hidden,
                eps=self.eps,
            )

        def linear(x: Buffer, y: Buffer, name: str, outputs: int, inputs: int) -> None:
            rt._dispatch(
                "matmul_f32",
                [x, self.weights[name + ".weight"], y],
                threads=((tokens + 3) // 4) * outputs * 32,
                group_size=32,
                rows=tokens,
                cols=outputs,
                k=inputs,
            )
            rt._dispatch(
                "add_bias",
                [y, self.weights[name + ".bias"]],
                threads=tokens * outputs,
                n=tokens * outputs,
                cols=outputs,
            )

        def add(x: Buffer, y: Buffer) -> None:
            rt._dispatch("add", [x, y, x], threads=tokens * self.hidden, n=tokens * self.hidden)

        try:
            with rt.command():
                token_buffer = new(ids.nbytes, np.ascontiguousarray(ids))
                length_buffer = new(lengths.nbytes, np.ascontiguousarray(lengths))
                x, projected, q, k, v, attended = (new(tokens * self.hidden * 4) for _ in range(6))
                activated = new(tokens * self.intermediate * 4)
                output = new(batch * dimensions * 4)
                rt._dispatch(
                    "embedding_position",
                    [
                        token_buffer,
                        self.weights["embeddings.word_embeddings.weight"],
                        self.weights["embeddings.position_embeddings.weight"],
                        self.weights["embeddings.token_type_embeddings.weight"],
                        x,
                    ],
                    threads=tokens * self.hidden,
                    n=tokens * self.hidden,
                    cols=self.hidden,
                    seq=seq,
                )
                norm(x, x, "embeddings.LayerNorm")
                for layer in range(self.layers):
                    prefix = f"encoder.layer.{layer}"
                    for target, name in ((q, "query"), (k, "key"), (v, "value")):
                        linear(
                            x, target, prefix + ".attention.self." + name, self.hidden, self.hidden
                        )
                    rt._dispatch(
                        "attention",
                        [q, k, v, length_buffer, attended],
                        threads=tokens * self.heads * 32,
                        group_size=32,
                        seq=seq,
                        heads=self.heads,
                        kv_heads=self.heads,
                        dim=self.head_dim,
                        scale=self.head_dim**-0.5,
                        bidirectional=True,
                    )
                    linear(
                        attended,
                        projected,
                        prefix + ".attention.output.dense",
                        self.hidden,
                        self.hidden,
                    )
                    add(x, projected)
                    norm(x, x, prefix + ".attention.output.LayerNorm")
                    linear(
                        x, activated, prefix + ".intermediate.dense", self.intermediate, self.hidden
                    )
                    rt._dispatch(
                        "gelu_f32",
                        [activated],
                        threads=tokens * self.intermediate,
                        n=tokens * self.intermediate,
                    )
                    linear(
                        activated,
                        projected,
                        prefix + ".output.dense",
                        self.hidden,
                        self.intermediate,
                    )
                    add(x, projected)
                    norm(x, x, prefix + ".output.LayerNorm")
                rt._dispatch(
                    "pool_project",
                    [x, length_buffer, output],
                    threads=batch * 32,
                    group_size=32,
                    seq=seq,
                    cols=self.hidden,
                    dim=dimensions,
                    first_token=True,
                )
            result = rt.read(output, (batch, dimensions))
            if not np.isfinite(result).all() or not np.allclose(
                np.linalg.norm(result, axis=1), 1, atol=1e-4, rtol=0
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
