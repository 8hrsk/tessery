"""Independent development-only BGE F32 graph using public MLX operations.

No upstream model implementation is imported. Weights are the verified local
BGE pack. This module is never imported by Tessery's runtime.
"""

import mlx.core as mx
import numpy as np
from mlx_reference_runner import ReferenceRunner

from metal_inference.weights import SafeTensors, read_artifact, read_json


class MLXBertReference:
    max_padded_tokens = 4096

    def __init__(self, model_dir, profile, *, compiled=False):
        self.runner = ReferenceRunner(mx, compiled=compiled)
        assert profile.architecture == "bert_f32" and profile.pooling == "cls"
        config = read_json(model_dir, "config.json", profile=profile)
        assert config["model_type"] == "bert" and config["hidden_act"] == "gelu"
        assert config["position_embedding_type"] == "absolute"
        assert not config.get("is_decoder", False) and not config.get("add_cross_attention", False)
        self.layers = config["num_hidden_layers"]
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.dim = self.hidden // self.heads
        self.eps = config["layer_norm_eps"]
        snapshot = SafeTensors(read_artifact(model_dir, "model.safetensors", profile=profile))
        self.weights = {}
        for name, info in snapshot.tensors.items():
            if name.startswith("pooler.") or name == "embeddings.position_ids":
                continue  # CLS encoder state; unused tanh pooler is excluded on both sides.
            assert info.dtype == "F32"
            raw = snapshot.view(name, shape=info.shape, dtype=info.dtype)
            values = raw.view(np.float32).reshape(info.shape)
            assert np.isfinite(values).all()
            self.weights[name] = mx.array(values)
        mx.eval(list(self.weights.values()))

    def forward(self, ids, lengths, *, dimensions):
        assert dimensions == self.hidden
        seq = ids.shape[1]
        dynamic_ids = mx.array(ids)
        dynamic_lengths = mx.array(lengths.astype(np.int32))
        mask = None
        if not np.all(lengths == seq):
            valid = mx.arange(seq)[None, None, None, :] < dynamic_lengths[:, None, None, None]
            mask = mx.where(valid, mx.array(0, mx.float32), mx.array(-float("inf"), mx.float32))
        return self.runner.run(self._graph, dynamic_ids, dynamic_lengths, mask, dimensions)

    def _graph(self, ids, lengths, mask, dimensions):
        batch, seq = ids.shape
        w = self.weights

        def norm(x, prefix):
            return mx.fast.layer_norm(x, w[prefix + ".weight"], w[prefix + ".bias"], self.eps)

        def linear(x, prefix):
            return mx.addmm(w[prefix + ".bias"], x, w[prefix + ".weight"].T)

        x = w["embeddings.word_embeddings.weight"][ids]
        x = x + w["embeddings.position_embeddings.weight"][mx.arange(seq)][None, :, :]
        x = x + w["embeddings.token_type_embeddings.weight"][0]
        x = norm(x, "embeddings.LayerNorm")
        for layer in range(self.layers):
            prefix = f"encoder.layer.{layer}"
            q, k, v = [
                linear(x, prefix + ".attention.self." + name)
                .reshape(batch, seq, self.heads, self.dim)
                .transpose(0, 2, 1, 3)
                for name in ("query", "key", "value")
            ]
            attended = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.dim**-0.5, mask=mask
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq, self.hidden)
            x = norm(
                x + linear(attended, prefix + ".attention.output.dense"),
                prefix + ".attention.output.LayerNorm",
            )
            expanded = linear(x, prefix + ".intermediate.dense")
            activated = 0.5 * expanded * (1 + mx.erf(expanded * (2**-0.5)))
            x = norm(x + linear(activated, prefix + ".output.dense"), prefix + ".output.LayerNorm")
        pooled = x[:, 0, :]
        result = pooled * mx.rsqrt(mx.sum(pooled * pooled, axis=-1, keepdims=True))
        return result

    def close(self):
        self.runner.clear()
        self.weights.clear()
        mx.clear_cache()
