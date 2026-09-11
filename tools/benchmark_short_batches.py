"""Paired public Qwen API benchmark against the pre-fallback batching policy.

Uses an existing local model. Normal command timestamps do not split encoders;
optional per-kernel profiling runs separately and perturbs execution timings.
"""

import argparse
import hashlib
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from diagnose_metal import delta, save_json, source_hashes, summarize

from metal_inference.batching import execution_batches, length_batches
from tessery import EmbeddingModel


def previous_batches(lengths, max_padded_tokens, max_length, architecture):
    """Frozen execution policy from 26dba97, for in-process paired controls."""
    for rows, width in length_batches(lengths, max_padded_tokens):
        aligned = (width + 7) // 8 * 8
        attention = aligned >= 64 and aligned % 32 == 0
        projection = width * len(rows) % 8 != 0 and (architecture == "bert_f32" or width < 128)
        if (
            architecture in {"qwen3_uint4", "bert_f32"}
            and width >= 5
            and (attention or projection)
            and aligned <= max_length
            and aligned <= 2 * int(lengths[rows].min())
            and aligned * len(rows) <= max_padded_tokens
        ):
            width = aligned
        yield rows, width


def stage_profile(model, texts, reference):
    rt = model._backend.runtime
    # profile_kernels holds the runtime lock: run directly in the caller thread,
    # not through the API worker. Normal timings below always use public encode.
    ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
    output = np.empty_like(reference)
    records = []
    for rows, width in model_plan(model, lengths):
        batch = np.full((len(rows), width), model._tokenizer.pad_id, np.uint32)
        copied = min(width, ids.shape[1])
        batch[:, :copied] = ids[rows, :copied]
        with rt.profile_kernels() as part:
            output[rows] = model._backend.forward(batch, lengths[rows], dimensions=model.dimensions)
        records.extend(part)
    np.testing.assert_allclose(output, reference, atol=5e-6, rtol=1e-4)
    groups = defaultdict(lambda: {"calls": 0, "gpu_seconds": 0.0})
    for row in records:
        assert np.isfinite(row["gpu_seconds"]) and row["gpu_seconds"] >= 0
        key = row["kernel"], row["rows"], row["cols"], row["k"]
        groups[key]["calls"] += 1
        groups[key]["gpu_seconds"] += row["gpu_seconds"]
    return sorted(
        [{"kernel": k[0], "rows": k[1], "cols": k[2], "k": k[3], **v} for k, v in groups.items()],
        key=lambda r: -r["gpu_seconds"],
    )


def model_plan(model, lengths):
    from metal_inference.api import execution_batches as planner

    return list(
        planner(
            lengths,
            model._backend.max_padded_tokens,
            model.max_length,
            model.descriptor.architecture,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--case", action="append", help="Comma-separated token lengths, repeatable")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--profile-first-case", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        cases = [list(map(int, value.split(","))) for value in (args.case or ["3,7,10", "7,10"])]
    except ValueError:
        parser.error("Each case must contain comma-separated integers")
    if not 3 <= args.samples <= 100 or any(
        not 1 <= len(case) <= 32 or any(not 3 <= n <= 512 for n in case) for case in cases
    ):
        parser.error("samples 3..100; cases of 1..32 lengths from 3..512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "samples_per_label": args.samples,
        "reverse_cases": args.reverse_cases,
        "requested_cases": cases,
        "conditions": (
            "Public encode API, randomized paired old A/B and selected policy; each label "
            "warmed for >=3 calls and >=1 second; uncontrolled thermals. GPU duration overlaps "
            "submit/wait duration; do not add them. Per-kernel profiles are intrusive, separate "
            "forward calls, and are not used for speedup claims."
        ),
        "results": [],
    }
    routes = {
        "previous_a": previous_batches,
        "previous_b": previous_batches,
        "selected": execution_batches,
    }
    try:
        with EmbeddingModel.load(args.model_dir) as model:
            rt = model._backend.runtime
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            order = list(reversed(cases)) if args.reverse_cases else cases
            for case in order:
                texts = [" token" * (n - 1) for n in case]
                ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert lengths.tolist() == case
                references, plans = {}, {}
                timings = {name: [] for name in routes}
                samples = {name: [] for name in routes}
                warm_time = {name: 0.0 for name in routes}
                warm_count = {name: 0 for name in routes}
                rng = np.random.default_rng(1900 + sum(case))
                while any(warm_count[n] < 3 or warm_time[n] < 1.0 for n in routes):
                    for name in rng.permutation(list(routes)):
                        with patch("metal_inference.api.execution_batches", routes[name]):
                            plans[name] = [(r.tolist(), w) for r, w in model_plan(model, lengths)]
                            start = time.perf_counter()
                            references[name] = model.encode(texts)
                            warm_time[name] += time.perf_counter() - start
                            warm_count[name] += 1
                difference = 0.0
                for _ in range(args.samples):
                    for name in rng.permutation(list(routes)):
                        with patch("metal_inference.api.execution_batches", routes[name]):
                            before = rt.diagnostics()
                            start = time.perf_counter()
                            output = model.encode(texts)
                            elapsed = time.perf_counter() - start
                            counters = delta(before, rt.diagnostics())
                        np.testing.assert_array_equal(output, references[name])
                        np.testing.assert_allclose(
                            output, references["previous_a"], atol=5e-6, rtol=1e-4
                        )
                        difference = max(
                            difference, float(np.max(np.abs(output - references["previous_a"])))
                        )
                        assert counters["completed_commands"] == len(plans[name])
                        assert counters["gpu_timed_commands"] == len(plans[name])
                        timings[name].append(elapsed)
                        samples[name].append(counters)
                a, b, selected = [float(np.median(timings[n])) for n in routes]
                row = {
                    "lengths": case,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "plans": plans,
                    "warmup_calls": warm_count,
                    "max_abs_difference": difference,
                    "timings": {n: summarize(t, len(case)) for n, t in timings.items()},
                    "normal_runtime_samples": samples,
                    "speedup": float(np.median(timings["previous_a"] + timings["previous_b"]))
                    / selected,
                    "identical_control_a_over_b": a / b,
                    "control_within_10_percent": 0.9 <= a / b <= 1.1,
                }
                if args.profile_first_case and case == cases[0]:
                    row["intrusive_profiles"] = {}
                    for name in ("previous_a", "selected"):
                        with patch("metal_inference.api.execution_batches", routes[name]):
                            row["intrusive_profiles"][name] = stage_profile(
                                model, texts, references[name]
                            )
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {
                        k: row[k]
                        for k in (
                            "lengths",
                            "speedup",
                            "max_abs_difference",
                            "control_within_10_percent",
                        )
                    },
                    flush=True,
                )
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
