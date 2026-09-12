"""Full-model replay regressions; run explicitly with the existing local model packs."""

import os
from pathlib import Path

import numpy as np
import pytest

from tessery import EmbeddingModel, ModelProfile

pytestmark = [
    pytest.mark.metal,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in local models"),
]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", params=["qwen", "bert"])
def loaded_model(request):
    if request.param == "qwen":
        directory = os.getenv(
            "METAL_INFERENCE_MODEL_DIR",
            str(Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ"),
        )
        profile = "qwen3-embedding-0.6b-dwq"
    else:
        directory = os.getenv(
            "METAL_INFERENCE_BGE_MODEL_DIR",
            str(Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs"),
        )
        profile = ModelProfile.from_file(
            os.getenv(
                "METAL_INFERENCE_BGE_PROFILE_FILE",
                str(ROOT / "model-manifests/bge-small-en-v1.5-hf-cache.json"),
            )
        )
    with EmbeddingModel.load(directory, profile=profile) as model:
        yield model
    assert model.memory_stats().active_bytes == model.memory_stats().plan_cache_bytes == 0


@pytest.fixture
def model(loaded_model):
    rt = loaded_model._backend.runtime
    previous_enabled = rt._plans_enabled
    previous_limit = rt.workspace_limit_bytes
    loaded_model.trim_memory()
    try:
        yield loaded_model
    finally:
        rt._plans_enabled = previous_enabled
        rt.workspace_limit_bytes = previous_limit
        loaded_model.trim_memory()


def inputs(model, lengths, word="token"):
    special = 2 if model.descriptor.architecture == "bert_f32" else 1
    texts = [f" {word}" * (length - special) for length in lengths]
    ids, actual = model._tokenizer.batch(texts, max_length=model.max_length)
    assert actual.tolist() == lengths
    return ids, actual


def forward(model, ids, lengths, *, enabled, dimensions=384):
    model._backend.runtime._plans_enabled = enabled
    return model._backend.forward(ids, lengths, dimensions=dimensions)


def assert_budget(model):
    rt = model._backend.runtime
    stats = model.memory_stats()
    diag = rt.diagnostics()
    assert stats.plan_cache_bytes == diag["plan_cache_bytes"]
    assert 0 <= stats.cache_bytes + stats.plan_cache_bytes <= rt.workspace_limit_bytes
    assert diag["plan_cache_entries"] <= 4


def test_full_model_replay_uses_changed_lengths_masks_and_tokens(model):
    rt = model._backend.runtime
    original_ids, original_lengths = inputs(model, [6, 12])
    expected = forward(model, original_ids, original_lengths, enabled=False)
    builds = rt.diagnostics()["plan_builds"]
    np.testing.assert_array_equal(
        forward(model, original_ids, original_lengths, enabled=True), expected
    )
    assert rt.diagnostics()["plan_builds"] == builds + 1
    for lengths, word in [([9, 12], "token"), ([9, 12], "world"), ([6, 12], "token")]:
        ids, actual = inputs(model, lengths, word)
        assert ids.shape == original_ids.shape == (2, 12)
        reference = forward(model, ids, actual, enabled=False)
        before = rt.diagnostics()
        for _ in range(2):
            np.testing.assert_array_equal(forward(model, ids, actual, enabled=True), reference)
            assert_budget(model)
        after = rt.diagnostics()
        assert after["plan_hits"] == before["plan_hits"] + 2
        assert after["plan_builds"] == before["plan_builds"]
    model.trim_memory()
    assert model.memory_stats().cache_bytes == model.memory_stats().plan_cache_bytes == 0


def test_qwen_full_model_dimension_changes_preserve_replay(model):
    if model.descriptor.architecture != "qwen3_uint4":
        pytest.skip("BGE exposes only its native dimensions")
    rt = model._backend.runtime
    ids, lengths = inputs(model, [7])
    for dimensions in [384, 64, 1024, 384]:
        reference = forward(model, ids, lengths, enabled=False, dimensions=dimensions)
        before = rt.diagnostics()
        np.testing.assert_array_equal(
            forward(model, ids, lengths, enabled=True, dimensions=dimensions), reference
        )
        hits = rt.diagnostics()["plan_hits"]
        np.testing.assert_array_equal(
            forward(model, ids, lengths, enabled=True, dimensions=dimensions), reference
        )
        assert rt.diagnostics()["plan_hits"] == hits + 1
        assert_budget(model)
    # The final 384-dimensional call reuses the first shape/dimension plan.
    assert rt.diagnostics()["plan_builds"] == before["plan_builds"]
    assert rt.diagnostics()["plan_cache_entries"] == 3


def test_full_model_plan_eviction_rebuild_and_combined_budget(model):
    rt = model._backend.runtime
    rt.workspace_limit_bytes = 1024 * 1024
    builds = rt.diagnostics()["plan_builds"]
    saved = {}
    for width in [3, 7, 12, 17, 24, 33]:
        ids, lengths = inputs(model, [width])
        reference = forward(model, ids, lengths, enabled=False)
        np.testing.assert_array_equal(forward(model, ids, lengths, enabled=True), reference)
        np.testing.assert_array_equal(forward(model, ids, lengths, enabled=True), reference)
        assert_budget(model)
        saved[width] = (ids, lengths, reference)
    assert rt.diagnostics()["plan_builds"] == builds + 6
    assert rt.diagnostics()["plan_cache_entries"] == 4
    ids, lengths, reference = saved[3]
    before = rt.diagnostics()
    np.testing.assert_array_equal(forward(model, ids, lengths, enabled=True), reference)
    assert rt.diagnostics()["plan_builds"] == before["plan_builds"] + 1
    assert rt.diagnostics()["plan_hits"] == before["plan_hits"]
    assert_budget(model)
    model.trim_memory()
    assert model.memory_stats().cache_bytes == model.memory_stats().plan_cache_bytes == 0
    assert rt.diagnostics()["plan_cache_entries"] == 0
    np.testing.assert_array_equal(forward(model, ids, lengths, enabled=True), reference)
    assert rt.diagnostics()["plan_builds"] == before["plan_builds"] + 2
    assert_budget(model)


def test_zero_cache_full_model_uses_exact_uncached_path(model):
    rt = model._backend.runtime
    rt.workspace_limit_bytes = 0
    ids, lengths = inputs(model, [6, 12])
    reference = forward(model, ids, lengths, enabled=False)
    before = rt.diagnostics()
    resident = model.memory_stats().active_bytes
    for _ in range(2):
        np.testing.assert_array_equal(forward(model, ids, lengths, enabled=True), reference)
        assert_budget(model)
        assert model.memory_stats().active_bytes == resident
        assert rt.diagnostics()["plan_builds"] == before["plan_builds"]
        assert rt.diagnostics()["plan_hits"] == before["plan_hits"]
    assert model.memory_stats().cache_bytes == model.memory_stats().plan_cache_bytes == 0
