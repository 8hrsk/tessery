"""Balanced full-API old/fused BGE affine comparison; run again with --reverse-cases."""

import argparse
import hashlib
import time
from itertools import permutations
from pathlib import Path

import numpy as np
from diagnose_metal import delta, save_json, source_hashes, summarize

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel, ModelProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--cases", nargs="+", help="Comma-separated logical lengths per case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 6 <= args.samples <= 120 or args.samples % 6:
        parser.error("samples must be a multiple of six from 6..120")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(args.profile_file.read_bytes()).hexdigest(),
        "samples_per_label": args.samples,
        "reverse_cases": args.reverse_cases,
        "results": [],
        "conditions": (
            "Public encode; old matmul plus add_bias versus fused affine; exact same weights; "
            "at least 3 warmups and 1s per label; balanced six permutations in shuffled blocks; "
            "uncontrolled thermals. Confidence interval resamples permutation blocks."
        ),
    }
    try:
        profile = ModelProfile.from_file(args.profile_file)
        with EmbeddingModel.load(args.model_dir, profile=profile) as model:
            rt = model._backend.runtime
            # Routing changes within one model; a recorded plan would otherwise
            # replay the first label and bypass the affine selector altogether.
            if hasattr(rt, "_plans_enabled"):
                rt._plans_enabled = False
            payload["execution_plan_enabled"] = False
            selected = rt._matmul_bias_f32
            state = {"label": "previous_a"}

            def route(buffers, *, rows, cols, k):
                if state["label"] == "selected":
                    return selected(buffers, rows=rows, cols=cols, k=k)
                rt._matmul_f32(buffers[:3], rows=rows, cols=cols, k=k)
                rt._dispatch("add_bias", buffers[2:], threads=rows * cols, n=rows * cols, cols=cols)

            rt._matmul_bias_f32 = route
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            payload["runtime_shader_sha256"] = rt.diagnostics()["shader_sha256"]
            cases = [[3], [7], [17], [24], [159], [160], [161], [256], [512], [33] * 4, [3, 7, 10]]
            if args.cases:
                cases = [[int(n) for n in case.split(",")] for case in args.cases]
                if any(not c or len(c) > 32 or any(not 3 <= n <= 512 for n in c) for c in cases):
                    raise ValueError("cases require 1..32 lengths from 3..512")
            if args.reverse_cases:
                cases.reverse()
            for case in cases:
                texts = [" token" * (n - 2) for n in case]
                ids, lengths = model._tokenizer.batch(texts, max_length=512)
                assert lengths.tolist() == case
                plans = [
                    (r.tolist(), w) for r, w in execution_batches(lengths, 4096, 512, "bert_f32")
                ]
                names = ["previous_a", "previous_b", "selected"]
                refs, warm = {}, {n: 0.0 for n in names}
                counts = {n: 0 for n in names}
                rng = np.random.default_rng(12200 + sum(case))
                while any(warm[n] < 1.0 or counts[n] < 3 for n in names):
                    for name in rng.permutation(names):
                        state["label"] = name
                        started = time.perf_counter()
                        refs[name] = model.encode(texts)
                        warm[name] += time.perf_counter() - started
                        counts[name] += 1
                for reference in refs.values():
                    np.testing.assert_array_equal(reference, refs["previous_a"])
                timings, counters = {n: [] for n in names}, {n: [] for n in names}
                orders = list(permutations(names))
                blocks = []
                for _ in range(args.samples // 6):
                    block = {n: [] for n in names}
                    for i in rng.permutation(6):
                        for name in orders[i]:
                            state["label"] = name
                            before = rt.diagnostics()
                            started = time.perf_counter()
                            output = model.encode(texts)
                            elapsed = time.perf_counter() - started
                            d = delta(before, rt.diagnostics())
                            np.testing.assert_array_equal(output, refs["previous_a"])
                            if name == "selected":
                                fallback_buckets = sum(len(r) * w < 8 for r, w in plans)
                                assert d["dispatches"].get("add_bias", 0) == 72 * fallback_buckets
                            else:
                                assert d["dispatches"]["add_bias"] == 72 * len(plans)
                            timings[name].append(elapsed)
                            block[name].append(elapsed)
                            counters[name].append(d)
                    blocks.append(block)
                block_differences = np.asarray(
                    [
                        np.mean(b["previous_a"] + b["previous_b"]) - np.mean(b["selected"])
                        for b in blocks
                    ]
                )
                boot = rng.choice(block_differences, (10000, len(blocks)), replace=True).mean(1)
                a, b, new = [float(np.median(timings[n])) for n in names]
                old = float(np.median(timings["previous_a"] + timings["previous_b"]))
                row = {
                    "lengths": case,
                    "plans": plans,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "vectors": refs["previous_a"].tolist(),
                    "exact_baseline_equality": True,
                    "timings": {n: summarize(v, len(case)) for n, v in timings.items()},
                    "normal_runtime_samples": counters,
                    "permutation_blocks": blocks,
                    "mean_latency_saved_seconds_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
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
            rt._matmul_bias_f32 = selected
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
