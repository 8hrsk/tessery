"""Reject invalid provenance/comparisons rather than reporting misleading speedups."""

import copy
import importlib.util
from pathlib import Path

import pytest


def comparer(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "benchmark_revisions", tools / "benchmark_revisions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare


def workers():
    row = {
        "lengths": [3],
        "input_ids_sha256": "tokens",
        "plans": [[[0], 3]],
        "vectors": [[0.5, 0.5]],
        "p50_seconds": 0.1,
        "p95_seconds": 0.11,
        "identical_control_a_over_b": 1.0,
        "control_within_10_percent": True,
    }
    w = {
        "source_hashes": {"kernel": "old"},
        "results": [row],
        "python": "3.12",
        "numpy": "2.5.2",
        "regex_version": "pinned",
        "compatibility_id": "same",
        "profile_sha256": "profile",
        "timing_scope": "api",
    }
    result = [copy.deepcopy(w) for _ in range(4)]
    for i in [1, 2]:
        result[i]["source_hashes"]["kernel"] = "candidate"
        result[i]["results"][0]["p50_seconds"] = 0.05
    return result


def test_revision_comparison_separates_timing_from_correctness(monkeypatch):
    compare = comparer(monkeypatch)
    data = workers()
    assert compare(data)[0]["baseline_over_candidate"] == [2.0, 2.0]
    data[1]["results"][0]["control_within_10_percent"] = False
    assert not compare(data)[0]["timing_screen_passed"]


@pytest.mark.parametrize(
    "change", ["vector", "token", "plan", "source", "environment", "duplicate"]
)
def test_revision_comparison_rejects_invalid_identity(monkeypatch, change):
    data = workers()
    if change == "vector":
        data[1]["results"][0]["vectors"][0][0] = 0.6
    elif change == "token":
        data[1]["results"][0]["input_ids_sha256"] = "other"
    elif change == "plan":
        data[1]["results"][0]["plans"] = [[[0], 8]]
    elif change == "source":
        data[2]["source_hashes"]["kernel"] = "changed"
    elif change == "environment":
        data[1]["regex_version"] = "other"
    else:
        data[1]["results"] *= 2
    with pytest.raises(AssertionError):
        comparer(monkeypatch)(data)
