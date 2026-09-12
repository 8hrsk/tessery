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


@pytest.mark.parametrize(
    "field,value",
    [("mlx_compile", True), ("timing_scope", "backend"), ("runner_sha256", "changed")],
)
def test_repeat_rejects_different_compilation_or_scope(compare, field, value):
    first, second = reports()
    second[field] = value
    with pytest.raises(AssertionError):
        compare(first, second)


def test_prepared_backend_matches_buckets_padding_and_order(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    prepare = runpy.run_path(str(folder / "benchmark_mlx_isolated.py"))["prepared_forward"]
    seen = []

    def forward(ids, lengths, *, dimensions):
        seen.append((ids.copy(), lengths.copy()))
        assert ids.flags.c_contiguous and lengths.flags.c_contiguous
        return np.column_stack((ids[:, 0], lengths)).astype(np.float32)

    model = SimpleNamespace(
        dimensions=2,
        max_length=512,
        _backend=SimpleNamespace(max_padded_tokens=4096, forward=forward),
        _tokenizer=SimpleNamespace(pad_id=99),
        descriptor=SimpleNamespace(architecture="qwen3_quantized"),
    )
    # A supported architecture is required to exercise Qwen alignment.
    from metal_inference.profiles import QWEN3_PROFILE

    model.descriptor.architecture = QWEN3_PROFILE.architecture
    ids = np.zeros((3, 10), np.uint32)
    ids[:, 0] = [42, 17, 88]
    lengths = np.array([10, 3, 7], np.uint32)
    call = prepare(model, ids, lengths)
    expected = np.array([[42, 10], [17, 3], [88, 7]], np.float32)
    np.testing.assert_array_equal(call(), expected)
    assert any(batch.shape[1] == 12 for batch, _ in seen)
    assert any((batch[:, 10:] == 99).all() for batch, _ in seen if batch.shape[1] == 12)
    np.testing.assert_array_equal(call(), expected)


def test_compile_cache_keeps_values_dynamic_and_rejects_retrace(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    runner_class = runpy.run_path(str(folder / "mlx_reference_runner.py"))["ReferenceRunner"]
    functions = []

    def graph(ids, lengths, mask, dimensions):
        return ids[:, :dimensions] + lengths[:, None] + mask[:, :dimensions]

    def compile_once(fn):
        called = False
        functions.append(fn)

        def compiled(*args):
            nonlocal called
            if not called:
                fn(*args)
                called = True
            # Simulate an already traced graph evaluating fresh array inputs.
            return graph(*args, 1)

        return compiled

    runner = runner_class(SimpleNamespace(compile=compile_once, eval=lambda x: None), compiled=True)
    ids = np.array([[1, 2]], np.uint32)
    lengths = np.array([2], np.int32)
    mask = np.array([[0, 0]], np.float32)
    np.testing.assert_array_equal(runner.run(graph, ids, lengths, mask, 1), [[3]])
    np.testing.assert_array_equal(runner.run(graph, ids + 5, lengths - 1, mask + 2, 1), [[9]])
    assert len(functions) == runner.traces == 1
    assert len(runner.diagnostics()["first_calls"]) == 1
    runner.functions[next(iter(runner.functions))] = functions[0]
    with pytest.raises(AssertionError, match="retraced"):
        runner.run(graph, ids, lengths, mask, 1)
    runner.clear()
    assert not runner.functions
