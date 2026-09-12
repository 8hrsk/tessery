"""Exact affine epilogue equivalence, independent reference, and output bounds."""

import os

import numpy as np
import pytest

from metal_inference.metal import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in native GPU tests"),
]


@pytest.mark.parametrize("rows", [1, 3, 7, 8, 9, 15, 16, 159, 160, 161, 512])
@pytest.mark.parametrize("cols,k", [(384, 384), (1536, 384), (384, 1536), (65, 37)])
def test_affine_exact_baseline_reference_and_output_guard(rows, cols, k):
    rng = np.random.default_rng(rows + k + cols)
    x = (rng.standard_normal((rows, k)) / 16).astype(np.float32)
    weight = (rng.standard_normal((cols, k)) / 16).astype(np.float32)
    bias = (rng.standard_normal(cols) / 16).astype(np.float32)
    # Deliberately include repeated cancellation and signed/bare zero cases.
    weight[0, 1::2] = -weight[0, : k // 2 * 2 : 2]
    x[0, 1::2] = x[0, : k // 2 * 2 : 2]
    bias[:3] = [-0.0, 0.0, -0.125]
    initial = np.full(rows * cols + 128, 123456.0, dtype=np.float32)
    with MetalRuntime() as rt:
        buffers = [rt.buffer(a.nbytes, a) for a in (x, weight, initial, bias)]
        selected = rt.buffer(initial.nbytes, initial)
        try:
            with rt.command():
                rt._matmul_f32(buffers[:3], rows=rows, cols=cols, k=k)
                rt._dispatch("add_bias", buffers[2:], threads=rows * cols, n=rows * cols, cols=cols)
            expected = rt.read(buffers[2], initial.shape)
            with rt.command():
                rt._matmul_bias_f32(
                    [buffers[0], buffers[1], selected, buffers[3]], rows=rows, cols=cols, k=k
                )
            actual = rt.read(selected, initial.shape)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[rows * cols :], initial[rows * cols :])
            reference = x.astype(np.float64) @ weight.astype(np.float64).T + bias
            np.testing.assert_allclose(
                actual[: rows * cols].reshape(rows, cols), reference, atol=5e-5, rtol=5e-5
            )
        finally:
            selected.close()
            for buffer in buffers:
                buffer.close()
        assert rt.active_bytes == 0
