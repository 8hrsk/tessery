"""Bounded exact/F64 and sentinel validation of down-projection experiments."""

import os
from pathlib import Path

import numpy as np
import pytest

from tessery import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in GPU"),
]


@pytest.mark.parametrize(
    "shape", [(64, 96, 256), (128, 1024, 3072), (256, 1024, 3072), (512, 1024, 3072)]
)
@pytest.mark.parametrize("pattern", ["integer", "cancellation", "large", "zero"])
def test_down_projection(monkeypatch, shape, pattern):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from benchmark_down_projection import KERNELS, measure

    m, n, k = shape
    rng = np.random.default_rng(91913)
    x = (rng.integers(-4, 5, size=(m, k)) / 8).astype(np.float32)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
    if pattern == "cancellation":
        x[:, ::2], x[:, 1::2] = 256.0, -256.0
        codes[:, 1::2] = codes[:, ::2]
    elif pattern == "large":
        x.fill(0)
        x[:, 0] = np.arange(m) * 32 - m * 16
    elif pattern == "zero":
        x.fill(0)
    packed = np.bitwise_or.reduce(
        codes.reshape(n, k // 8, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
    )
    arrays = [packed]
    for value in (0.125, -1.0):
        values = np.full((n, k // 64), value, np.float32)
        arrays.append((values.view(np.uint32) >> 16).astype(np.uint16))
    with MetalRuntime() as rt:
        result = measure(rt, arrays, x, list(KERNELS)[1:], validate_only=True)
        assert result["exact_baseline_equality"] and result["inputs_unchanged"]
    assert rt.active_bytes == rt.cache_bytes == 0
