"""Paired full-Qwen comparison with an independent graph built from public MLX ops.

Development-only harness; MLX is never imported by Tessery's runtime. Both paths
use the same verified local weights, tokenizer, API admission and batch splitting.
No mlx-embeddings code is loaded. No models or packages are downloaded.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from metal_inference.profiles import QWEN3_PROFILE
from metal_inference.weights import SafeTensors, read_artifact, read_json
from tessery import EmbeddingModel


class MLXReference:
    max_padded_tokens = 4096

    def __init__(self, model_dir, *, causal_fast_path=False):
        self.causal_fast_path = causal_fast_path
        config = read_json(model_dir, "config.json")
        self.layers = config["num_hidden_layers"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        self.theta = config["rope_theta"]
        snapshots = SafeTensors(read_artifact(model_dir, "model.safetensors"))
        self.weights = {}
        for name, info in snapshots.tensors.items():
            raw = snapshots.view(name, shape=info.shape, dtype=info.dtype)
            if info.dtype == "U32":
                array = mx.array(raw.view(np.uint32).reshape(info.shape))
            elif info.dtype == "BF16":
                # Exact BF16 values promoted to F32 before arithmetic, as in Tessery.
                values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
                array = mx.array(values.reshape(info.shape))
            else:
                raise ValueError("Unexpected tensor dtype")
            self.weights[name] = array
        mx.eval(list(self.weights.values()))

    def quant(self, prefix):
        return [self.weights[prefix + "." + key] for key in ("weight", "scales", "biases")]

    def forward(self, ids, lengths, *, dimensions):
        batch, seq = ids.shape
        weight, scales, biases = self.quant("model.embed_tokens")
        tokens = mx.array(ids)
        x = mx.dequantize(weight[tokens], scales[tokens], biases[tokens], group_size=64, bits=4)

        def norm(value, prefix):
            return mx.fast.rms_norm(value, self.weights[prefix + ".weight"], self.eps)

        def linear(value, prefix):
            return mx.quantized_matmul(value, *self.quant(prefix), group_size=64, bits=4)

        if self.causal_fast_path and np.all(lengths == seq):
            mask = "causal"
        else:
            positions = mx.arange(seq)
            allowed = (positions[None, :] <= positions[:, None])[None, None, :, :]
            allowed = allowed & (
                positions[None, None, None, :] < mx.array(lengths)[:, None, None, None]
            )
            mask = mx.where(allowed, mx.array(0, mx.float32), mx.array(-float("inf"), mx.float32))
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}"
            attention = prefix + ".self_attn"
            normalized = norm(x, prefix + ".input_layernorm")
            q = norm(
                linear(normalized, attention + ".q_proj").reshape(batch, seq, self.heads, self.dim),
                attention + ".q_norm",
            )
            k = norm(
                linear(normalized, attention + ".k_proj").reshape(
                    batch, seq, self.kv_heads, self.dim
                ),
                attention + ".k_norm",
            )
            v = linear(normalized, attention + ".v_proj").reshape(
                batch, seq, self.kv_heads, self.dim
            )
            q = mx.fast.rope(
                q.transpose(0, 2, 1, 3),
                self.dim,
                traditional=False,
                base=self.theta,
                scale=1.0,
                offset=0,
            )
            k = mx.fast.rope(
                k.transpose(0, 2, 1, 3),
                self.dim,
                traditional=False,
                base=self.theta,
                scale=1.0,
                offset=0,
            )
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v.transpose(0, 2, 1, 3), scale=self.dim**-0.5, mask=mask
            )
            x = x + linear(
                attended.transpose(0, 2, 1, 3).reshape(batch, seq, -1), attention + ".o_proj"
            )
            normalized = norm(x, prefix + ".post_attention_layernorm")
            gate = linear(normalized, prefix + ".mlp.gate_proj")
            up = linear(normalized, prefix + ".mlp.up_proj")
            x = x + linear(gate * mx.sigmoid(gate) * up, prefix + ".mlp.down_proj")
        pooled = norm(x, "model.norm")[
            mx.arange(batch), mx.array(lengths.astype(np.int32) - 1), :dimensions
        ]
        result = pooled * mx.rsqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
        mx.eval(result)
        return np.array(result)

    def close(self):
        self.weights.clear()
        mx.clear_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100:
        parser.error("samples must be 3..100")
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "scope": "same_host_full_qwen_public_api",
        "baseline": "independent F32 Qwen graph using public MLX ops; not mlx-embeddings",
        "platform": platform.platform(),
        "mlx_version": importlib.metadata.version("mlx"),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_sha256": QWEN3_PROFILE.identity_sha256,
        "conditions": "random paired order; 2 warmups; both models resident; uncontrolled thermals",
        "memory_note": (
            "allocator-owned bytes only; MLX BF16 values expanded to F32; excludes process RSS"
        ),
        "results": [],
    }
    try:
        rng = np.random.default_rng(94)
        with (
            EmbeddingModel.load(args.model_dir) as tessery,
            EmbeddingModel(
                MLXReference(args.model_dir),
                tessery._tokenizer,
                tessery.dimensions,
                tessery.max_length,
                8,
                tessery.descriptor,
            ) as reference,
        ):
            for texts in [
                [" token" * (n - 1)] * batch for batch, n in [(1, 7), (1, 8), (1, 31), (4, 33)]
            ] + [["Hello world", "Кошки и собаки", "A much longer passage about Paris in France."]]:
                samples = {"tessery": [], "mlx": []}
                outputs = {}
                difference = 0.0
                for iteration in range(args.samples + 2):
                    for name in rng.permutation(list(samples)):
                        started = time.perf_counter()
                        outputs[name] = (tessery if name == "tessery" else reference).encode(texts)
                        if iteration >= 2:
                            samples[name].append(time.perf_counter() - started)
                    np.testing.assert_allclose(
                        outputs["tessery"], outputs["mlx"], atol=5e-6, rtol=1e-4
                    )
                    difference = max(
                        difference, float(np.max(np.abs(outputs["tessery"] - outputs["mlx"])))
                    )
                _, lengths = tessery._tokenizer.batch(texts, max_length=tessery.max_length)
                row = {
                    "texts": texts,
                    "lengths": lengths.tolist(),
                    "seconds": samples,
                    "max_abs_vector_difference": difference,
                    "tessery_over_mlx_latency": float(
                        np.median(samples["tessery"]) / np.median(samples["mlx"])
                    ),
                }
                payload["results"].append(row)
                print({k: v for k, v in row.items() if k not in ("texts", "seconds")}, flush=True)
                args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            payload["tessery_runtime"] = tessery._backend.runtime.diagnostics()
            payload["mlx_allocator"] = {
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
