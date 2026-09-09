import runpy
from pathlib import Path

import pytest


@pytest.fixture
def summarize_stability(monkeypatch):
    tools = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    return runpy.run_path(str(tools / "benchmark_mlp_isolated.py"))["stability_summary"]


def workers(times=(2.0, 1.0, 1.0, 2.0)):
    return [
        {
            "results": {
                "projection:128": {
                    "timings": {"p50_seconds": value},
                    "control_within_10_percent": True,
                }
            }
        }
        for value in times
    ]


def test_stable_engine_gap_is_not_mistaken_for_measurement_drift(summarize_stability):
    row = summarize_stability(workers())[0]
    assert row["tessery_over_mlx_range"] == [2.0, 2.0]
    assert row["timing_screen_passed"]


def test_good_local_controls_do_not_hide_cross_process_drift(summarize_stability):
    row = summarize_stability(workers((2.0, 1.0, 1.0, 3.0)))[0]
    assert row["repeat_max_over_min"]["tessery"] == 1.5
    assert not row["timing_screen_passed"]


def test_matching_process_medians_do_not_hide_bad_local_control(summarize_stability):
    data = workers()
    data[2]["results"]["projection:128"]["control_within_10_percent"] = False
    assert not summarize_stability(data)[0]["timing_screen_passed"]


def test_missing_case_cannot_be_silently_dropped(summarize_stability):
    data = workers()
    data[-1]["results"] = {}
    with pytest.raises(AssertionError):
        summarize_stability(data)


@pytest.mark.parametrize("invalid", [0.0, float("nan")])
def test_invalid_timing_cannot_pass_screen(summarize_stability, invalid):
    with pytest.raises(AssertionError):
        summarize_stability(workers((invalid, 1.0, 1.0, 2.0)))
