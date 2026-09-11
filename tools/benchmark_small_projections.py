"""Paired full Qwen encode timings for the bounded three-row scalar specialization."""

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
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 6 <= args.samples <= 96 or args.samples % 6:
        parser.error("samples must be a multiple of six from 6..96")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_commit": "1435226",
        "samples_per_label": args.samples,
        "reverse_cases": args.reverse_cases,
        "conditions": (
            "Public encode; all six permutations of old A/B and selected routes in shuffled "
            "blocks, balancing positions; each label warmed "
            "for >=3 calls and >=1 second; normal command boundaries; uncontrolled thermals. "
            "GPU time overlaps submit/wait time and must not be added to it."
        ),
        "results": [],
    }
    try:
        with EmbeddingModel.load(args.model_dir) as model:
            rt = model._backend.runtime
            selected = rt._linear4

            def previous(buffers, *, rows, cols, k):
                if rows == 3 and (cols, k) in (
                    (1024, 1024),
                    (2048, 1024),
                    (3072, 1024),
                    (1024, 2048),
                    (1024, 3072),
                ):
                    rt._dispatch(
                        "linear4",
                        buffers,
                        threads=cols * 32,
                        group_size=32,
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                else:
                    selected(buffers, rows=rows, cols=cols, k=k)

            routes = {"previous_a": previous, "previous_b": previous, "selected": selected}
            cases = [[2], [3], [4], [3, 7, 10], [10, 3, 7], [7], [3, 3]]
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            for case in reversed(cases) if args.reverse_cases else cases:
                texts = [" token" * (n - 1) for n in case]
                ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert lengths.tolist() == case
                plan = [
                    (r.tolist(), w) for r, w in execution_batches(lengths, 4096, 512, "qwen3_uint4")
                ]
                expected_calls = 196 * sum(len(r) * w == 3 for r, w in plan)
                rng = np.random.default_rng(2000 + sum(case))
                references, warm_counts = {}, {n: 0 for n in routes}
                warm_time = {n: 0.0 for n in routes}
                while any(warm_counts[n] < 3 or warm_time[n] < 1.0 for n in routes):
                    for name in rng.permutation(list(routes)):
                        rt._linear4 = routes[name]
                        start = time.perf_counter()
                        references[name] = model.encode(texts)
                        warm_time[name] += time.perf_counter() - start
                        warm_counts[name] += 1
                for value in references.values():
                    np.testing.assert_array_equal(value, references["previous_a"])
                timings, counters = {n: [] for n in routes}, {n: [] for n in routes}
                orders = list(permutations(routes))
                balanced = [orders[i] for _ in range(args.samples // 6) for i in rng.permutation(6)]
                for order in balanced:
                    for name in order:
                        rt._linear4 = routes[name]
                        before = rt.diagnostics()
                        start = time.perf_counter()
                        output = model.encode(texts)
                        elapsed = time.perf_counter() - start
                        difference = delta(before, rt.diagnostics())
                        np.testing.assert_array_equal(output, references["previous_a"])
                        assert difference["dispatches"].get("linear4_small3", 0) == (
                            expected_calls if name == "selected" else 0
                        )
                        assert difference["completed_commands"] == len(plan)
                        assert difference["gpu_timed_commands"] == len(plan)
                        timings[name].append(elapsed)
                        counters[name].append(difference)
                a, b, new = [float(np.median(timings[n])) for n in routes]
                old = float(np.median(timings["previous_a"] + timings["previous_b"]))
                row = {
                    "lengths": case,
                    "plans": plan,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "vectors": references["previous_a"].tolist(),
                    "exact_baseline_equality": True,
                    "warmup_calls": warm_counts,
                    "timings": {n: summarize(v, len(case)) for n, v in timings.items()},
                    "normal_runtime_samples": counters,
                    "speedup": old / new,
                    "identical_control_a_over_b": a / b,
                    "control_within_10_percent": 0.9 <= a / b <= 1.1,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {k: row[k] for k in ("lengths", "speedup", "control_within_10_percent")},
                    flush=True,
                )
            rt._linear4 = selected
            model.trim_memory()
            payload["runtime_before_close"] = rt.diagnostics()
        payload["runtime_after_close"] = rt.diagnostics()
        assert rt.active_bytes == rt.cache_bytes == 0
        assert payload["source_hashes"] == source_hashes()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
