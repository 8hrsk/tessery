"""Balanced Qwen M8/M16 existing fused kernels with two fixed native-plan caches."""

import argparse
import hashlib
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
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
            runtimes = [model._backend.runtime for model in (baseline, candidate)]
            assert all(rt._plans_enabled for rt in runtimes)
            baseline_runtime = runtimes[0]
            baseline_original = baseline_runtime._gated4

            def unfused(buffers, *, rows, cols, k):
                if rows in (8, 16) and (cols, k) == (3072, 1024):
                    baseline_runtime._linear4([*buffers[:4], buffers[7]], rows=rows, cols=cols, k=k)
                    baseline_runtime._linear4(
                        [buffers[0], *buffers[4:7], buffers[8]],
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                    baseline_runtime._dispatch(
                        "silu_gate",
                        buffers[7:],
                        threads=rows * cols,
                        n=rows * cols,
                    )
                    return None
                return baseline_original(buffers, rows=rows, cols=cols, k=k)

            # Preserve the historical control even after the selected guard is integrated.
            baseline_runtime._gated4 = unfused
            rt = runtimes[1]
            original = rt._gated4

            def gated(buffers, *, rows, cols, k):
                if rows in (8, 16) and (cols, k) == (3072, 1024):
                    kernel = "gated4_8x32" if rows == 8 else "gated4_16x32_k64"
                    group = 128 if rows == 8 else 256
                    rt._dispatch(
                        kernel,
                        buffers[:8],
                        threads=(cols // 32) * group,
                        group_size=group,
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                    return None
                return original(buffers, rows=rows, cols=cols, k=k)

            # Each model captures its own immutable route before its first forward.
            rt._gated4 = gated
            models = {"baseline_a": baseline, "baseline_b": baseline, "candidate": candidate}
            cases = [[7], [8], [9], [12], [16], [3], [24], [3, 7], [7, 7], [3, 7, 10]]
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
                targets = sum(len(rows) * width in (8, 16) for rows, width in plans)
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
        assert all(r.active_bytes == r.cache_bytes == 0 for r in runtimes)
        assert source_hashes() == payload["source_hashes"]
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
