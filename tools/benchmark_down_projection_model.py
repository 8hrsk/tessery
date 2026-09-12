"""Symmetric full-API comparison of a selected down projection against the old shader."""

import argparse
import hashlib
import importlib.metadata
import math
import platform
import time
from contextlib import ExitStack
from functools import partial
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from benchmark_shader_library_control import memory_snapshot
from diagnose_metal import delta, save_json, source_hashes, summarize
from shader_library_control import revision_shader, runtime_with_shader

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel

KERNELS = ("linear4_32x32_transposed_k64", "linear4_64x32_wide_k64")


def select(runtime, kernel):
    original = runtime._linear4

    def linear(buffers, *, rows, cols, k):
        if rows == 512 and (cols, k) == (1024, 3072):
            group = 512 if kernel == KERNELS[1] else 256
            runtime._dispatch(
                kernel,
                buffers,
                threads=(rows // (64 if kernel == KERNELS[1] else 32)) * (cols // 32) * group,
                group_size=group,
                rows=rows,
                cols=cols,
                k=k,
            )
            return None
        return original(buffers, rows=rows, cols=cols, k=k)

    assert runtime.diagnostics()["plan_builds"] == 0
    runtime._linear4 = linear


def old_route(runtime):
    original = runtime._dispatch

    def dispatch(kernel, buffers, **params):
        if kernel in KERNELS:
            kernel = "linear4_32x32_k64"
            params["group_size"] = 256
            params["threads"] = (params["rows"] // 32) * (params["cols"] // 32) * 256
        return original(kernel, buffers, **params)

    assert runtime.diagnostics()["plan_builds"] == 0
    runtime._dispatch = dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--old-revision", default="f6e9169")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=[KERNELS[0]])
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [
        "baseline_a",
        "baseline_b",
        *[k + suffix for k in args.kernels for suffix in ("_a", "_b")],
    ]
    order_count = math.factorial(len(labels))
    if (
        not order_count <= args.samples <= 96
        or args.samples % order_count
        or len(args.kernels) != 1
        or not all(n in (128, 160, 256, 512) for n in args.lengths)
    ):
        parser.error("unique kernels; full permutation blocks; lengths128/160/256/512")
    revision, shader = revision_shader(args.old_revision)
    helpers = [
        Path(__file__),
        *[
            Path(__file__).with_name(n)
            for n in (
                "diagnose_metal.py",
                "shader_library_control.py",
                "benchmark_shader_library_control.py",
            )
        ],
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "helper_hashes": hashes,
        "old_revision": revision,
        "old_shader_sha256": hashlib.sha256(shader).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "regex": importlib.metadata.version("regex"),
        "samples_per_label": args.samples,
        "reverse": args.reverse,
        "kernels": args.kernels,
        "results": [],
        "memory": [],
        "conditions": (
            "Fixed independent plans; old shader baseline; two labels per resident model; "
            "all label permutations; "
            ">=1s warmup per label; sequential GPU; uncontrolled thermals"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    models = {}
    initial_swap = None

    def memory_check(phase):
        nonlocal initial_swap
        sample = memory_snapshot()
        sample["phase"] = phase
        payload["memory"].append(sample)
        if initial_swap is None:
            initial_swap = sample["system_swap_used_bytes"]
        assert sample["system_pressure_level"] == 1, "memory pressure"
        assert sample["system_swap_used_bytes"] <= initial_swap + 16 * 2**20, "swap grew"

    try:
        memory_check("before_load")
        allocation = ["baseline", *args.kernels]
        if args.reverse:
            allocation.reverse()
        payload["allocation_order"] = allocation
        with ExitStack() as stack:
            for name in allocation:
                if name == "baseline":
                    with patch(
                        "metal_inference.qwen3.MetalRuntime", partial(runtime_with_shader, shader)
                    ):
                        model = stack.enter_context(EmbeddingModel.load(args.model_dir))
                    old_route(model._backend.runtime)
                else:
                    model = stack.enter_context(EmbeddingModel.load(args.model_dir))
                    select(model._backend.runtime, name)
                models[name] = model
                assert model._backend.runtime._plans_enabled
                memory_check("loaded_" + name)
            payload["models"] = {
                name: {
                    "shader_sha256": model._backend.runtime.diagnostics()["shader_sha256"],
                    "profile_sha256": model._backend.profile.identity_sha256,
                    "model_id": model.descriptor.model_id,
                    "compatibility_id": model.descriptor.compatibility_id,
                }
                for name, model in models.items()
            }
            assert len({r["profile_sha256"] for r in payload["models"].values()}) == 1
            routes = {
                "baseline_a": models["baseline"],
                "baseline_b": models["baseline"],
                **{k + suffix: models[k] for k in args.kernels for suffix in ("_a", "_b")},
            }
            cases = [[n] for n in args.lengths] + [[16], [3, 7, 10], [159, 160]]
            if args.reverse:
                cases.reverse()
            for lengths in cases:
                texts = [" token" * (n - 1) for n in lengths]
                ids, actual = models["baseline"]._tokenizer.batch(texts, max_length=512)
                assert actual.tolist() == lengths
                plans = list(execution_batches(actual, 4096, 512, "qwen3_uint4"))
                targets = sum(len(rows) * width == 512 for rows, width in plans)
                warm = {n: 0.0 for n in labels}
                counts = {n: 0 for n in labels}
                refs = {}
                rng = np.random.default_rng(1310 + sum(lengths))
                while any(warm[n] < 1 or counts[n] < 3 for n in labels):
                    for name in rng.permutation(labels):
                        start = time.perf_counter()
                        refs[name] = routes[name].encode(texts)
                        warm[name] += time.perf_counter() - start
                        counts[name] += 1
                for value in refs.values():
                    np.testing.assert_array_equal(value, refs["baseline_a"])
                timings, counters = {n: [] for n in labels}, {n: [] for n in labels}
                all_orders = list(permutations(labels))
                orders = [
                    all_orders[i]
                    for _ in range(args.samples // order_count)
                    for i in rng.permutation(order_count)
                ]
                for order in orders:
                    for name in order:
                        runtime = routes[name]._backend.runtime
                        before = runtime.diagnostics()
                        start = time.perf_counter()
                        value = routes[name].encode(texts)
                        timings[name].append(time.perf_counter() - start)
                        diff = delta(before, runtime.diagnostics())
                        assert diff["plan_hits"] > 0 and diff["plan_builds"] == 0
                        for kernel in KERNELS:
                            assert diff["dispatches"].get(kernel, 0) == (
                                targets * 28 if name in (kernel + "_a", kernel + "_b") else 0
                            )
                        np.testing.assert_array_equal(value, refs["baseline_a"])
                        counters[name].append(diff)
                expected = counters["baseline_a"][0]["dispatches"]
                for samples in counters.values():
                    for sample in samples:
                        normalized = dict(sample["dispatches"])
                        for kernel in KERNELS:
                            count = normalized.pop(kernel, 0)
                            if count:
                                normalized["linear4_32x32_k64"] = (
                                    normalized.get("linear4_32x32_k64", 0) + count
                                )
                        assert normalized == expected
                memory_check("measured_" + str(lengths))
                baseline = float(np.median(timings["baseline_a"] + timings["baseline_b"]))
                row = {
                    "lengths": lengths,
                    "plans": [(r.tolist(), w) for r, w in plans],
                    "target_buckets": targets,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "warmups": counts,
                    "orders": orders,
                    "vectors": refs["baseline_a"].tolist(),
                    "exact_baseline_equality": True,
                    "unchanged_dispatches_verified": True,
                    "timings": {k: summarize(v, len(lengths)) for k, v in timings.items()},
                    "normal_runtime_samples": counters,
                    "candidate_control_a_over_b": {
                        k: float(np.median(timings[k + "_a"]) / np.median(timings[k + "_b"]))
                        for k in args.kernels
                    },
                    "speedup": {
                        k: baseline / float(np.median(timings[k + "_a"] + timings[k + "_b"]))
                        for k in args.kernels
                    },
                    "identical_control_a_over_b": float(
                        np.median(timings["baseline_a"]) / np.median(timings["baseline_b"])
                    ),
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {k: row[k] for k in ("lengths", "speedup", "identical_control_a_over_b")},
                    flush=True,
                )
        assert all(
            m._backend.runtime.active_bytes
            == m._backend.runtime.cache_bytes
            == m._backend.runtime._plan_bytes
            == 0
            for m in models.values()
        )
        assert source_hashes() == payload["source_hashes"]
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
        memory_check("after_close")
        payload["status"] = "passed"
    except BaseException as error:
        payload.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
