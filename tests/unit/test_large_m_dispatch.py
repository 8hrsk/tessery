"""The experimental full-row tile must never consume a partial tile."""

from types import SimpleNamespace

import pytest

from metal_inference.metal import MetalRuntime


@pytest.mark.parametrize(
    "rows",
    [3, 16, 24, 31, 32, 33, 127, 128, 129, 159, 160, 161, 255, 256, 257, 511, 512, 513, 4096],
)
@pytest.mark.parametrize(
    "cols,k",
    [(1024, 1024), (2048, 1024), (3072, 1024), (1024, 2048), (1024, 3072), (96, 64), (1025, 1024)],
)
def test_full_large_m_guard(rows, cols, k):
    calls = []
    rt = SimpleNamespace(_dispatch=lambda *a, **kw: calls.append((a, kw)))
    MetalRuntime._linear4(rt, [], rows=rows, cols=cols, k=k)
    large = [kw for a, kw in calls if a[0] == "linear4_32x32_k64"]
    expected = rows in (128, 160, 256, 512) and (cols, k) in (
        (1024, 1024),
        (2048, 1024),
        (3072, 1024),
        (1024, 2048),
        (1024, 3072),
    )
    assert bool(large) == expected
    if expected:
        assert len(calls) == 1
        assert large[0]["threads"] // 256 * 32 * 32 == rows * cols
        assert large[0]["group_size"] == 256
        assert large[0].get("n", 0) == 0
