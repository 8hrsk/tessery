"""Compare final host routing with the prior tile on ragged and neighboring inputs."""

import os
from pathlib import Path

import numpy as np
import pytest

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel

pytestmark = [
    pytest.mark.metal,
    pytest.mark.real_model,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in pinned model"),
]


@pytest.fixture(scope="module")
def models():
    directory = os.getenv(
        "METAL_INFERENCE_MODEL_DIR",
        str(Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ"),
    )
    with EmbeddingModel.load(directory) as old, EmbeddingModel.load(directory) as new:
        rt = old._backend.runtime
        dispatch = rt._dispatch

        def old_dispatch(kernel, buffers, **params):
            if kernel == "linear4_32x32_k64":
                kernel = "linear4_16x32_k64"
                params["threads"] *= 2
            return dispatch(kernel, buffers, **params)

        # Installed before the first capture; the two models keep independent plans.
        rt._dispatch = old_dispatch
        yield old, new
    for model in (old, new):
        stats = model.memory_stats()
        assert stats.active_bytes == stats.cache_bytes == stats.plan_cache_bytes == 0


@pytest.mark.parametrize(
    "lengths",
    [
        [3],
        [7],
        [17],
        [24],
        [31],
        [32],
        [33],
        [33] * 4,
        [128],
        [127, 128],
        [129],
        [160],
        [159, 160],
        [161],
        [256],
        [255, 256],
        [257],
        [511],
        [512],
    ],
)
def test_large_m_ragged_and_neighbors(models, lengths):
    old, new = models
    texts = [" token" * (n - 1) for n in lengths]
    _, actual = new._tokenizer.batch(texts, max_length=512)
    assert actual.tolist() == lengths
    plans = list(execution_batches(actual, 4096, 512, new.descriptor.architecture))
    target = sum(len(rows) * width in (128, 160, 256, 512) for rows, width in plans)
    rt = new._backend.runtime
    for _ in range(2):
        before = rt.diagnostics()
        reference = old.encode(texts)
        result = new.encode(texts)
        np.testing.assert_array_equal(result, reference)
        after = rt.diagnostics()
        kernel = "linear4_32x32_k64"
        assert (
            after["dispatches"].get(kernel, 0) - before["dispatches"].get(kernel, 0)
            == target * 28 * 5
        )
