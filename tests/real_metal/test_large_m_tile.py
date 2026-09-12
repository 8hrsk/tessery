"""Larger row tiles retain the original K32 reduction and bounded writes."""

import os

import numpy as np
import pytest

from tessery import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in GPU"),
]


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32, 64),
        (64, 96, 256),
        (128, 1024, 1024),
        (160, 2048, 1024),
        (256, 3072, 1024),
        (512, 2048, 1024),
        (512, 1024, 3072),
    ],
)
@pytest.mark.parametrize("pattern", ["random", "cancellation", "zero"])
def test_large_row_tile_exact(shape, pattern):
    m, n, k = shape
    rng = np.random.default_rng(9104)
    x = rng.normal(size=(m, k)).astype(np.float32)
    if pattern == "cancellation":
        x[:, ::2] = 256
        x[:, 1::2] = -256
    elif pattern == "zero":
        x.fill(0)
    w = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32)
    s = (rng.uniform(0.01, 0.2, size=(n, k // 64)).astype(np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    b = (rng.uniform(-1, 0.1, size=s.shape).astype(np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )
    with MetalRuntime() as rt:
        buffers = [rt.buffer(a.nbytes, a) for a in (x, w, s, b)]
        # A guard row catches accidental stores beyond the last complete tile.
        sentinel = np.full((m + 1, n), -123.25, np.float32)
        out = rt.buffer(sentinel.nbytes, sentinel)
        try:
            values = []
            for kernel, tile in [("linear4_16x32_k64", 16), ("linear4_32x32_k64", 32)]:
                out.close()
                out = rt.buffer(sentinel.nbytes, sentinel)
                with rt.command():
                    rt._dispatch(
                        kernel,
                        [*buffers, out],
                        threads=m // tile * (n // 32) * 256,
                        group_size=256,
                        rows=m,
                        cols=n,
                        k=k,
                    )
                value = rt.read(out, sentinel.shape)
                np.testing.assert_array_equal(value[m], sentinel[m])
                values.append(value[:m])
            np.testing.assert_array_equal(*values)
        finally:
            out.close()
            for buffer in buffers:
                buffer.close()
    assert rt.active_bytes == 0
