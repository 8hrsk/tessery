"""Small synthetic packs prove the adapters are not restricted to registered weight hashes."""

import hashlib
import json
import os
import struct
from dataclasses import replace

import numpy as np
import pytest

from metal_inference.errors import ManifestError, UnsupportedProfileError
from metal_inference.tokenizer import QWEN_SPLIT, byte_alphabet
from tessery import Artifact, EmbeddingModel, ModelProfile, get_profile

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in native GPU tests"),
]


def save_pack(root, architecture, config, tokenizer, tensors):
    payload = bytearray()
    header = {}
    for name, (array, dtype) in tensors.items():
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    files = {
        "config.json": json.dumps(config).encode(),
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "model.safetensors": struct.pack("<Q", len(encoded)) + encoded + payload,
    }
    artifacts = []
    for name, data in files.items():
        (root / name).write_bytes(data)
        artifacts.append(Artifact(name, name, len(data), hashlib.sha256(data).hexdigest()))
    base = get_profile(
        "bge-small-en-v1.5" if architecture == "bert_f32" else "qwen3-embedding-0.6b-dwq"
    )
    profile = replace(
        base,
        model_id="synthetic-tests/" + architecture,
        revision="seed17",
        native_dimensions=config["hidden_size"],
        min_dimensions=config["hidden_size"] if architecture == "bert_f32" else 32,
        default_dimensions=config["hidden_size"] if architecture == "bert_f32" else 32,
        max_length=16,
        artifacts=tuple(artifacts),
    )
    (root / "profile.json").write_text(json.dumps(profile.to_dict()))
    return ModelProfile.from_file(root / "profile.json")


@pytest.fixture(params=["bert_f32", "qwen3_uint4"])
def pack(tmp_path, request):
    root = tmp_path.resolve()
    rng = np.random.default_rng(17)
    tensors = {}
    if request.param == "bert_f32":
        hidden, intermediate, vocab = (
            32,
            64,
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"],
        )
        config = {
            "model_type": "bert",
            "architectures": ["BertModel"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "vocab_size": len(vocab),
            "max_position_embeddings": 16,
            "type_vocab_size": 2,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
        }
        tokenizer = {
            "model": {
                "type": "WordPiece",
                "vocab": {v: i for i, v in enumerate(vocab)},
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100,
            },
            "normalizer": {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": True,
                "strip_accents": None,
                "lowercase": True,
            },
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "added_tokens": [
                {
                    "id": i,
                    "content": v,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                for i, v in enumerate(vocab[:5])
            ],
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                ],
                "special_tokens": {"[CLS]": {"ids": [2]}, "[SEP]": {"ids": [3]}},
            },
        }

        def put(name, shape, norm=False, bias=False):
            array = (
                np.ones(shape, np.float32)
                if norm
                else (
                    np.zeros(shape, np.float32)
                    if bias
                    else rng.normal(0, 0.02, shape).astype(np.float32)
                )
            )
            tensors[name] = array, "F32"

        for name, rows in (("word", len(vocab)), ("position", 16), ("token_type", 2)):
            put(f"embeddings.{name}_embeddings.weight", (rows, hidden))
        prefix = "encoder.layer.0"
        for name in (
            "embeddings.LayerNorm",
            prefix + ".attention.output.LayerNorm",
            prefix + ".output.LayerNorm",
        ):
            put(name + ".weight", (hidden,), norm=True)
            put(name + ".bias", (hidden,), bias=True)
        for name, rows, cols in [
            *((prefix + ".attention.self." + v, hidden, hidden) for v in ("query", "key", "value")),
            (prefix + ".attention.output.dense", hidden, hidden),
            (prefix + ".intermediate.dense", intermediate, hidden),
            (prefix + ".output.dense", hidden, intermediate),
        ]:
            put(name + ".weight", (rows, cols))
            put(name + ".bias", (rows,), bias=True)
    else:
        hidden, intermediate = 64, 128
        config = {
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "vocab_size": 257,
            "hidden_act": "silu",
            "attention_bias": False,
            "rope_scaling": None,
            "use_sliding_window": False,
            "quantization": {"bits": 4, "group_size": 64},
            "rms_norm_eps": 1e-6,
            "rope_theta": 1e6,
        }
        tokenizer = {
            "normalizer": {"type": "NFC"},
            "model": {
                "type": "BPE",
                "vocab": {v: i for i, v in enumerate(byte_alphabet())},
                "merges": [],
            },
            "pre_tokenizer": {"pretokenizers": [{"pattern": {"Regex": r".+"}}]},
            "added_tokens": [
                {
                    "content": "<eot>",
                    "id": 256,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                }
            ],
            "post_processor": {
                "processors": [{}, {"special_tokens": {"<|endoftext|>": {"ids": [256]}}}]
            },
        }

        def bf16(value, shape):
            return (np.full(shape, value, np.float32).view(np.uint32) >> 16).astype(np.uint16)

        def norm(name, dims):
            tensors[name + ".weight"] = bf16(1, (dims,)), "BF16"

        def quant(name, outputs, inputs):
            tensors[name + ".weight"] = (
                rng.integers(0, 2**32, size=(outputs, inputs // 8), dtype=np.uint32),
                "U32",
            )
            tensors[name + ".scales"] = bf16(0.02, (outputs, inputs // 64)), "BF16"
            tensors[name + ".biases"] = bf16(-0.16, (outputs, inputs // 64)), "BF16"

        quant("model.embed_tokens", 257, hidden)
        norm("model.norm", hidden)
        prefix = "model.layers.0"
        norm(prefix + ".input_layernorm", hidden)
        norm(prefix + ".post_attention_layernorm", hidden)
        for name, width in (("q", 64), ("k", 32), ("v", 32)):
            quant(prefix + f".self_attn.{name}_proj", width, hidden)
        quant(prefix + ".self_attn.o_proj", hidden, 64)
        norm(prefix + ".self_attn.q_norm", 32)
        norm(prefix + ".self_attn.k_norm", 32)
        quant(prefix + ".mlp.gate_proj", intermediate, hidden)
        quant(prefix + ".mlp.up_proj", intermediate, hidden)
        quant(prefix + ".mlp.down_proj", hidden, intermediate)
    if request.param == "qwen3_uint4":
        byte_level = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
        tokenizer["pre_tokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN_SPLIT},
                    "behavior": "Isolated",
                    "invert": False,
                },
                byte_level,
            ],
        }
        tokenizer["added_tokens"][0]["content"] = "<|endoftext|>"
        template = tokenizer["post_processor"]["processors"][1]
        template.update(
            type="TemplateProcessing",
            single=[
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}},
            ],
        )
        tokenizer["post_processor"].update(type="Sequence", processors=[byte_level, template])
    return (
        root,
        save_pack(root, request.param, config, tokenizer, tensors),
        config,
        tokenizer,
        tensors,
    )


def test_custom_shapes_and_weights_run_without_core_changes(pack):
    root, profile, _, _, _ = pack
    with EmbeddingModel.load(root, profile=profile) as model:
        output = model.encode(["hello", "another input"])
        assert output.shape == (2, 32)
        np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-5)
        assert model.descriptor.model_id == profile.model_id
        assert model.descriptor.compatibility_id == profile.compatibility_id
        runtime = model._backend.runtime
    assert runtime.active_bytes == 0


def test_custom_pack_integrity_and_config_rejection(pack):
    root, profile, config, tokenizer, tensors = pack
    original = (root / "config.json").read_bytes()
    (root / "config.json").write_bytes(b"x" * len(original))
    with pytest.raises(ManifestError):
        EmbeddingModel.load(root, profile=profile)
    config["hidden_size"] = 33
    bad = save_pack(root, profile.architecture, config, tokenizer, tensors)
    with pytest.raises(UnsupportedProfileError):
        EmbeddingModel.load(root, profile=bad)


def test_custom_pack_rejects_tensor_layout_before_allocating(pack):
    root, profile, config, tokenizer, tensors = pack
    tensors.pop(next(iter(tensors)))
    bad = save_pack(root, profile.architecture, config, tokenizer, tensors)
    with pytest.raises(ManifestError):
        EmbeddingModel.load(root, profile=bad)
