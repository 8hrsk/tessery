import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIAG = runpy.run_path(str(Path(__file__).resolve().parents[2] / "tools/diagnose_metal.py"))


def test_counter_delta_does_not_subtract_memory_peaks():
    before = dict.fromkeys(
        (
            "allocations",
            "allocated_bytes",
            "completed_commands",
            "gpu_timed_commands",
            "gpu_seconds",
            "encode_seconds",
            "submit_wait_seconds",
        ),
        2,
    )
    after = {k: v + 3 for k, v in before.items()}
    before.update(dispatches={"a": 2}, peak_bytes=100)
    after.update(dispatches={"a": 4, "b": 1}, peak_bytes=100)
    result = DIAG["delta"](before, after)
    assert result["completed_commands"] == 3
    assert result["dispatches"] == {"a": 2, "b": 1}
    assert "peak_bytes" not in result


def test_latency_summary_uses_total_time_and_all_texts():
    summary = DIAG["summarize"]([1.0, 2.0, 3.0], 4)
    assert summary["p50_seconds"] == 2.0
    assert summary["p95_seconds"] == pytest.approx(2.9)
    assert summary["texts_per_second"] == 2.0


def test_diagnostic_rejects_unverified_token_count():
    model = SimpleNamespace(
        descriptor=SimpleNamespace(architecture="bert_f32"),
        max_length=512,
        _tokenizer=SimpleNamespace(batch=lambda *a, **kw: (np.zeros((1, 2)), np.array([2]))),
    )
    with pytest.raises(ValueError, match="token count"):
        DIAG["benchmark"](model, 1, 32, 2, 1)


def test_missing_or_zero_gpu_timings_are_not_a_speedup():
    benchmark = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "tools/benchmark_matmul.py")
    )
    assert benchmark["speedup"]([], []) is None
    assert benchmark["speedup"]([0.0], [0.0]) is None
    assert benchmark["speedup"]([2.0, 4.0], [1.0, 2.0]) == 2.0


def test_checkpoint_failure_keeps_previous_json(tmp_path, monkeypatch):
    import json
    import os

    path = tmp_path / "result.json"
    DIAG["save_json"](path, {"status": "running"})
    old = path.read_bytes()
    with pytest.raises(ValueError):
        DIAG["save_json"](path, {"value": float("nan")})
    assert path.read_bytes() == old

    def fail_replace(*args):
        raise OSError("injected filesystem failure")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError):
            DIAG["save_json"](path, {"status": "passed"})
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]
    DIAG["save_json"](path, {"status": "passed"})
    assert json.loads(path.read_text())["status"] == "passed"
