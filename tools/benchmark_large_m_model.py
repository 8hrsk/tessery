"""Balanced long-Qwen linear tile comparison with fixed native-plan caches."""

import argparse
import hashlib
import importlib.metadata
import platform
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from diagnose_metal import delta, save_json, source_hashes, summarize

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 6 or args.samples > 96 or args.samples % 6:
        parser.error("samples must be a multiple of six from 6..96")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    helper_paths = [Path(__file__), Path(__file__).with_name("diagnose_metal.py")]
    helper_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helper_paths}
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_hashes": helper_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "regex": importlib.metadata.version("regex"),
        "samples_per_label": args.samples,
        "reverse_cases": args.reverse_cases,
        "conditions": (
            "Public encode; identical A/B baseline; balanced six permutations; "
            ">=1s warmup per route"
        ),
        "results": [],
    }
    try:
        with (
            EmbeddingModel.load(args.model_dir) as baseline,
            EmbeddingModel.load(args.model_dir) as candidate,
        ):
            payload["model_id"] = baseline.descriptor.model_id
            payload["compatibility_id"] = baseline.descriptor.compatibility_id
            payload["profile_sha256"] = baseline._backend.profile.identity_sha256
            assert candidate._backend.profile.identity_sha256 == payload["profile_sha256"]
            runtimes = [model._backend.runtime for model in (baseline, candidate)]
            assert all(rt._plans_enabled for rt in runtimes)
            baseline_rt = runtimes[0]
            baseline_dispatch = baseline_rt._dispatch

            def old_dispatch(kernel, buffers, **params):
                if kernel == "linear4_32x32_k64":
                    kernel = "linear4_16x32_k64"
                    params["threads"] *= 2
                return baseline_dispatch(kernel, buffers, **params)

            # Keep the historical baseline after the guarded route is promoted.
            baseline_rt._dispatch = old_dispatch
            rt = runtimes[1]
            original = rt._linear4

            def linear(buffers, *, rows, cols, k):
                if rows in (128, 160, 256, 512) and (cols, k) in (
                    (1024, 1024),
                    (2048, 1024),
                    (3072, 1024),
                    (1024, 2048),
                    (1024, 3072),
                ):
                    rt._dispatch(
                        "linear4_32x32_k64",
                        buffers,
                        threads=(rows // 32) * (cols // 32) * 256,
                        group_size=256,
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                    return None
                return original(buffers, rows=rows, cols=cols, k=k)

            rt._linear4 = linear
            models = {"baseline_a": baseline, "baseline_b": baseline, "candidate": candidate}
            cases = [[24], [128], [160], [256], [512], [33, 33, 33, 33], [159, 160]]
            if args.reverse_cases:
                cases.reverse()
            names = ["baseline_a", "baseline_b", "candidate"]
            for case in cases:
                texts = [" token" * (n - 1) for n in case]
                ids, lengths = baseline._tokenizer.batch(texts, max_length=512)
                assert lengths.tolist() == case
                plans = list(
                    execution_batches(lengths, 4096, 512, baseline.descriptor.architecture)
                )
                targets = sum(len(rows) * width in (128, 160, 256, 512) for rows, width in plans)
                rng = np.random.default_rng(2900 + sum(case))
                elapsed = {name: 0.0 for name in names}
                warmups = {name: 0 for name in names}
                references = {}
                while any(elapsed[name] < 1.0 or warmups[name] < 3 for name in names):
                    for name in rng.permutation(names):
                        model = models[name]
                        started = time.perf_counter()
                        references[name] = model.encode(texts)
                        elapsed[name] += time.perf_counter() - started
                        warmups[name] += 1
                timings = {name: [] for name in names}
                counters = {name: [] for name in names}
                orders = list(permutations(names))
                for order in [
                    orders[i] for _ in range(args.samples // 6) for i in rng.permutation(6)
                ]:
                    for name in order:
                        model = models[name]
                        runtime = model._backend.runtime
                        before = runtime.diagnostics()
                        started = time.perf_counter()
                        result = model.encode(texts)
                        timings[name].append(time.perf_counter() - started)
                        d = delta(before, runtime.diagnostics())
                        assert runtime.diagnostics()["plan_hits"] > before["plan_hits"]
                        if name == "candidate":
                            assert d["dispatches"].get("linear4_32x32_k64", 0) == targets * 28 * 5
                        counters[name].append(d)
                        np.testing.assert_array_equal(result, references["baseline_a"])
                a, b, candidate_median = [float(np.median(timings[name])) for name in names]
                row = {
                    "lengths": case,
                    "plans": [(r.tolist(), w) for r, w in plans],
                    "target_buckets": targets,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "exact_baseline_equality": True,
                    "vectors": references["baseline_a"].tolist(),
                    "timings": {n: summarize(v, len(case)) for n, v in timings.items()},
                    "normal_runtime_samples": counters,
                    "speedup": float(np.median(timings["baseline_a"] + timings["baseline_b"]))
                    / candidate_median,
                    "identical_control_a_over_b": a / b,
                    "control_within_10_percent": 0.9 <= a / b <= 1.1,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {k: row[k] for k in ("lengths", "speedup", "control_within_10_percent")},
                    flush=True,
                )
            payload["runtime_final"] = [r.diagnostics() for r in runtimes]
        assert all(r.active_bytes == r.cache_bytes == r._plan_bytes == 0 for r in runtimes)
        assert source_hashes() == payload["source_hashes"]
        assert helper_hashes == {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helper_paths
        }
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
