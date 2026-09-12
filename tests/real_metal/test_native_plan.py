"""Prepared graphs must bind current buffers and obey command/workspace lifetime."""

import os

import numpy as np
import pytest

from metal_inference.errors import InferenceError
from metal_inference.metal import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in GPU"),
]


def forward(rt, a, key=(16,), fail=False):
    with rt._workspace() as new:
        with rt.command():
            x = new(a.nbytes, a)
            y = new(a.nbytes)
            z = new(a.nbytes)
            with rt._execution_plan(key, [x, y, z]) as replayed:
                if not replayed:
                    rt._dispatch("add", [x, x, y], threads=a.size, n=a.size)
                    rt._dispatch("add", [x, y, z], threads=a.size, n=a.size)
                if fail:
                    raise ValueError("abort")
        return rt.read(z, a.shape)


def test_replay_uses_current_inputs_and_scratch_and_trim():
    with MetalRuntime() as rt:
        rt._plans_enabled = True
        for i in range(5):
            a = np.arange(16, dtype=np.float32) + i
            np.testing.assert_array_equal(forward(rt, a), a * 3)
        diag = rt.diagnostics()
        assert diag["plan_hits"] == 4 and diag["plan_builds"] == 1
        assert diag["dispatches"]["add"] == 10
        assert 0 < diag["plan_cache_bytes"] + diag["cache_bytes"] <= rt.workspace_limit_bytes
        rt.trim_workspace()
        assert rt.active_bytes == rt._plan_bytes == 0
        np.testing.assert_array_equal(forward(rt, a), a * 3)
        assert rt._plan_builds == 2
    assert rt.active_bytes == rt._plan_bytes == 0


def test_abort_replay_and_capture_invalidate_plans():
    with MetalRuntime() as rt:
        rt._plans_enabled = True
        a = np.ones(16, np.float32)
        for prepared in [False, True]:
            if prepared:
                forward(rt, a)
            with pytest.raises(ValueError, match="abort"):
                forward(rt, a, fail=True)
            assert rt._plan_bytes == 0 and not rt._plans
            np.testing.assert_array_equal(forward(rt, a), a * 3)
            rt.trim_workspace()


def test_plan_count_size_guard_and_profile_bypass():
    with MetalRuntime(workspace_limit_bytes=4096) as rt:
        rt._plans_enabled = True
        for size in range(4, 12):
            a = np.ones(size, np.float32)
            np.testing.assert_array_equal(forward(rt, a, key=(size,)), a * 3)
            assert len(rt._plans) <= 4
            assert rt.cache_bytes + rt._plan_bytes <= rt.workspace_limit_bytes
        # A stale key with different-sized bindings is rejected before encoding.
        with pytest.raises(InferenceError):
            forward(rt, np.ones(12, np.float32), key=(11,))
        assert not rt._plans
        a = np.ones(16, np.float32)
        forward(rt, a)
        hits = rt._plan_hits
        with rt.profile_kernels() as records:
            np.testing.assert_array_equal(forward(rt, a), a * 3)
        assert len(records) == 2 and rt._plan_hits == hits


def test_zero_cache_and_closed_slot():
    with MetalRuntime(workspace_limit_bytes=0) as rt:
        rt._plans_enabled = True
        a = np.ones(16, np.float32)
        for _ in range(2):
            np.testing.assert_array_equal(forward(rt, a), a * 3)
            assert rt.active_bytes == rt._plan_bytes == 0
        rt.workspace_limit_bytes = 4096
        with rt._workspace() as new:
            with rt.command():
                x = new(a.nbytes, a)
                x.close()
                with pytest.raises(InferenceError), rt._execution_plan((16,), [x]):
                    pass
