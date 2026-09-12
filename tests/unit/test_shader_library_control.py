import hashlib
import importlib
from collections import Counter
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    return importlib.import_module("shader_library_control")


@pytest.mark.parametrize("raise_error", [False, True])
def test_exact_shader_substitution_restores_reads(helper, monkeypatch, tmp_path, raise_error):
    original = Path.read_bytes
    other = tmp_path / "other.metal"
    other.write_bytes(b"unchanged")

    def construct(**kwargs):
        assert helper.SHADER.read_bytes() == b"replacement"
        assert other.read_bytes() == b"unchanged"
        if raise_error:
            raise ValueError("construction failed")
        return SimpleNamespace(
            diagnostics=lambda: {"shader_sha256": hashlib.sha256(b"replacement").hexdigest()}
        )

    monkeypatch.setattr(helper, "MetalRuntime", construct)
    if raise_error:
        with pytest.raises(ValueError):
            helper.runtime_with_shader(b"replacement")
    else:
        helper.runtime_with_shader(b"replacement")
    assert Path.read_bytes is original
    assert helper.SHADER.read_bytes() != b"replacement"


def test_legacy_route_changes_only_new_kernel_before_capture(helper):
    calls = []
    runtime = SimpleNamespace(
        diagnostics=lambda: {"plan_builds": 0}, _dispatch=lambda *a, **kw: calls.append((a, kw))
    )
    helper.force_old_linear(runtime)
    params = dict(rows=128, cols=1024, k=1024, group_size=256, threads=32768)
    runtime._dispatch("linear4_32x32_k64", [], **params)
    assert calls[-1][0][0] == "linear4_16x32_k64"
    assert calls[-1][1]["threads"] == 65536
    assert params["threads"] == 32768
    runtime._dispatch("rms_norm", [], threads=32)
    assert calls[-1] == (("rms_norm", []), {"threads": 32})
    runtime.diagnostics = lambda: {"plan_builds": 1}
    with pytest.raises(AssertionError):
        helper.force_old_linear(runtime)


@pytest.mark.parametrize("samples", [24, 48])
def test_all_labels_balanced_in_every_position(helper, samples):
    bench = importlib.import_module("benchmark_shader_library_control")
    labels = ["baseline_a", "baseline_b", "legacy", "candidate"]
    orders = bench.balanced_orders(labels, samples, 1234)
    assert Counter(orders) == {order: samples // 24 for order in permutations(labels)}
    for position in range(4):
        assert Counter(order[position] for order in orders) == {
            label: samples // 4 for label in labels
        }
