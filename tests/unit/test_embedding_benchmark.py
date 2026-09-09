import copy
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def compare(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    return runpy.run_path(str(folder / "compare_embedding_runs.py"))["compare"]


def reports():
    row = {
        "case": 0,
        "lengths": [128],
        "input_ids_sha256": "ids",
        "plans": [[[0], 128]],
        "vectors": [[0.25, 0.5]],
        "p50_seconds": 1.0,
        "control_within_10_percent": True,
    }
    worker = {
        "source_hashes": {"runtime": "digest"},
        "python": "p",
        "numpy": "n",
        "regex_version": "r",
        "compatibility_id": "id",
        "mlx_version": "m",
        "results": [row],
    }
    first = {
        "status": "passed",
        "engine_order": "tessery-first",
        "paired_controls": True,
        "workers": {name: copy.deepcopy(worker) for name in ("tessery", "mlx")},
    }
    for key in (
        "harness_sha256",
        "reference_sha256",
        "profile_sha256",
        "mlx_mask",
        "requested_lengths",
        "include_batches",
        "samples_per_label",
    ):
        first[key] = "same"
    first["workers"]["tessery"]["results"][0]["p50_seconds"] = 2.0
    second = copy.deepcopy(first)
    second["engine_order"] = "mlx-first"
    return first, second


def test_full_model_repeat_preserves_real_gap(compare):
    row = compare(*reports())[0]
    assert row["tessery_over_mlx_range"] == [2.0, 2.0]
    assert row["timing_screen_passed"]


@pytest.mark.parametrize("change", ["input", "duplicate", "source", "missing"])
def test_full_model_repeat_rejects_identity_mismatch(compare, change):
    first, second = reports()
    worker = second["workers"]["mlx"]
    if change == "input":
        worker["results"][0]["input_ids_sha256"] = "different"
    elif change == "duplicate":
        worker["results"].append(copy.deepcopy(worker["results"][0]))
    elif change == "missing":
        worker["results"].clear()
    else:
        worker["source_hashes"] = {"runtime": "changed"}
    with pytest.raises(AssertionError):
        compare(first, second)


def test_full_model_repeat_retains_failed_noise_screen(compare):
    first, second = reports()
    second["workers"]["mlx"]["results"][0]["control_within_10_percent"] = False
    assert not compare(first, second)[0]["timing_screen_passed"]
