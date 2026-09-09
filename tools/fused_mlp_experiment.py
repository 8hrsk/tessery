"""Private experimental runtime; never patches production files or selectors."""

import hashlib
from pathlib import Path
from unittest.mock import patch

from diagnose_metal import ROOT

from tessery import MetalRuntime

SHADER = ROOT / "src/metal_inference/native/kernels.metal"
CANDIDATE = Path(__file__).with_name("shaders") / "fused_mlp.metal"
KERNELS = ("fused_mlp_serial", "fused_mlp_parallel")


def experimental_runtime(**kwargs):
    source = SHADER.read_bytes() + b"\n" + CANDIDATE.read_bytes()
    read_bytes = Path.read_bytes

    def read(path):
        return source if path == SHADER else read_bytes(path)

    with patch.object(Path, "read_bytes", read):
        rt = MetalRuntime(**kwargs)
    assert rt.diagnostics()["shader_sha256"] == hashlib.sha256(source).hexdigest()
    return rt


def fused(rt, kernel, buffers, *, rows, cols, k):
    assert kernel in KERNELS and rows >= 16 and rows % 16 == 0
    assert (cols, k) == (3072, 1024)
    rt._dispatch(
        kernel,
        buffers,
        threads=(rows // 16) * (cols // 32) * 256,
        group_size=256,
        rows=rows,
        cols=cols,
        k=k,
    )


def hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), CANDIDATE, Path(__file__).with_name("benchmark_mlp_isolated.py"))
    }


def install_model_route(model):
    """Intercept only adjacent, verified gate/up/SiLU calls in this private model.

    Keeps the normal up allocation for a fair first integration experiment.
    All non-aligned shapes retain the original complete path.
    """
    rt = model._backend.runtime
    if hasattr(rt, "_gated4"):
        selected = rt._gated4
        state = {"kernel": None, "pending": None, "skip": None}

        def gated(buffers, **kw):
            kernel = state["kernel"]
            if kernel == "selected":
                return selected(buffers, **kw)
            if (
                kernel is not None
                and kw["rows"] >= 16
                and kw["rows"] % 16 == 0
                and (kw["cols"], kw["k"]) == (3072, 1024)
            ):
                return fused(rt, kernel, buffers[:8], **kw)
            rt._linear4([*buffers[:4], buffers[7]], **kw)
            rt._linear4([buffers[0], *buffers[4:7], buffers[8]], **kw)
            rt._dispatch(
                "silu_gate", buffers[7:], threads=kw["rows"] * kw["cols"], n=kw["rows"] * kw["cols"]
            )
            return None

        rt._gated4 = gated
        return state
    original_linear, original_dispatch = rt._linear4, rt._dispatch
    weights = model._backend.weights
    pairs = {
        weights[name].pointer: weights[name.replace("gate_proj", "up_proj")].pointer
        for name in weights
        if name.endswith(".mlp.gate_proj.weight")
    }
    assert len(pairs) == 28
    state = {"kernel": None, "pending": None, "skip": None}

    def linear(buffers, **kw):
        candidate = state["kernel"]
        if (
            candidate is None
            or kw["rows"] < 16
            or kw["rows"] % 16
            or (kw["cols"], kw["k"]) != (3072, 1024)
        ):
            assert state["pending"] is None and state["skip"] is None
            return original_linear(buffers, **kw)
        assert candidate in KERNELS and state["skip"] is None
        if buffers[1].pointer in pairs:
            assert state["pending"] is None
            state["pending"] = (buffers, kw)
            return None
        assert state["pending"] is not None
        gate, shape = state["pending"]
        assert shape == kw and gate[0] is buffers[0]
        assert pairs[gate[1].pointer] == buffers[1].pointer
        fused(rt, candidate, [*gate[:4], *buffers[1:4], gate[-1]], **kw)
        state["pending"] = None
        state["skip"] = (gate[-1], buffers[-1])
        return None

    def dispatch(name, buffers, **kw):
        if state["skip"] is not None:
            assert name == "silu_gate" and tuple(buffers) == state["skip"]
            state["skip"] = None
            return None
        return original_dispatch(name, buffers, **kw)

    rt._linear4, rt._dispatch = linear, dispatch
    return state
