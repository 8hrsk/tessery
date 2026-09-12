"""Independent F64 references, cancellation/saturation and guarded writes."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tessery import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in GPU"),
]


@pytest.mark.parametrize("rows", [32, 64, 128, 160, 256, 512])
@pytest.mark.parametrize("pattern", ["integer", "cancellation", "saturation", "zero"])
def test_fused32_independent_reference(monkeypatch, rows, pattern):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    from benchmark_fused32 import measure

    rng = np.random.default_rng(1313)
    x = (rng.integers(-4, 5, size=(rows, 1024)) / 8).astype(np.float32)
    arrays = []
    for _ in range(2):
        codes = rng.integers(0, 16, size=(3072, 1024), dtype=np.uint32)
        if pattern == "cancellation":
            codes[:, 1::2] = codes[:, ::2]
            x[:, ::2], x[:, 1::2] = 1.0, -1.0
        elif pattern == "saturation":
            codes.fill(8)
            codes[:, 0] = 9
            x.fill(0)
            x[::2, 0], x[1::2, 0] = 768.0, -768.0
        elif pattern == "zero":
            x.fill(0)
        packed = np.bitwise_or.reduce(
            codes.reshape(3072, 128, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
        )
        scales = np.full((3072, 16), 0.125, dtype=np.float32)
        biases = np.full_like(scales, -1.0)
        arrays.extend(
            [
                packed,
                (scales.view(np.uint32) >> 16).astype(np.uint16),
                (biases.view(np.uint32) >> 16).astype(np.uint16),
            ]
        )
    with MetalRuntime() as rt:
        row = measure(rt, arrays, x, SimpleNamespace(validate_only=True), 1314)
        assert row["exact_baseline_equality"] and row["output_guard_untouched"]
        assert rt.active_bytes == 0
