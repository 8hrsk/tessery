"""Balanced complete-API native-plan experiment, including identical A/B controls."""

import argparse
import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from diagnose_metal import delta, save_json, source_hashes

from tessery import EmbeddingModel, ModelProfile


def interval(ratios):
    """Bootstrap independent balanced blocks, never individual correlated calls."""
    rng = np.random.default_rng(213)
    values = np.asarray(ratios)
    boots = np.median(rng.choice(values, (10000, len(values))), axis=1)
    return np.quantile(boots, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--cases", nargs="+", help="Comma-separated logical lengths per case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 6 or args.samples % 6:
        parser.error("samples must be a positive multiple of six")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write("{}")
    payload = {
        "status": "running",
        "sources": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reverse": args.reverse,
        "samples": args.samples,
        "cases": [],
    }
    profile = (
        ModelProfile.from_file(args.profile_file)
        if args.profile_file
        else "qwen3-embedding-0.6b-dwq"
    )
    with EmbeddingModel.load(args.model_dir, profile=profile) as model:
        rt = model._backend.runtime
        cases = [[3], [7], [17], [24], [3, 7, 10], [33] * 4, [160], [512]]
        if args.cases:
            cases = [[int(n) for n in value.split(",")] for value in args.cases]
        if not all(1 <= len(case) <= 32 and all(3 <= n <= 512 for n in case) for case in cases):
            parser.error("Invalid cases")
        if args.reverse:
            cases.reverse()
        special = 2 if model.descriptor.architecture == "bert_f32" else 1
        payload["model"] = model.descriptor.model_id
        payload["compatibility_id"] = model.descriptor.compatibility_id
        for lengths in cases:
            texts = [" token" * (n - special) for n in lengths]
            ids, actual = model._tokenizer.batch(texts, max_length=model.max_length)
            assert actual.tolist() == lengths
            rt._plans_enabled = False
            reference = model.encode(texts)
            cold = {}
            for enabled in [False, True]:
                rt._plans_enabled = enabled
                begin = time.perf_counter()
                output = model.encode(texts)
                cold[str(enabled)] = time.perf_counter() - begin
                np.testing.assert_array_equal(output, reference)
                start = time.perf_counter()
                count = 0
                while count < 3 or time.perf_counter() - start < 1.0:
                    np.testing.assert_array_equal(model.encode(texts), reference)
                    count += 1
            samples = {label: [] for label in ["baseline_a", "baseline_b", "plan"]}
            runtime_samples = {label: [] for label in samples}
            orders = list(itertools.permutations(samples))
            if args.reverse:
                orders.reverse()
            ratios = []
            rng = np.random.default_rng(2400 + sum(lengths))
            before = rt.diagnostics()
            for _block in range(args.samples // 6):
                block_samples = {label: [] for label in samples}
                for order in [orders[i] for i in rng.permutation(len(orders))]:
                    for label in order:
                        rt._plans_enabled = label == "plan"
                        runtime_before = rt.diagnostics()
                        start = time.perf_counter()
                        output = model.encode(texts)
                        elapsed = time.perf_counter() - start
                        runtime_samples[label].append(delta(runtime_before, rt.diagnostics()))
                        np.testing.assert_array_equal(output, reference)
                        samples[label].append(elapsed)
                        block_samples[label].append(elapsed)
                ratios.append(
                    float(
                        np.median(block_samples["baseline_a"] + block_samples["baseline_b"])
                        / np.median(block_samples["plan"])
                    )
                )
            after = rt.diagnostics()
            # A changed token with the same execution shape must never reuse input contents.
            rt._plans_enabled = False
            changed_texts = [text.replace("token", "world") for text in texts]
            changed_ref = model.encode(changed_texts)
            rt._plans_enabled = True
            np.testing.assert_array_equal(model.encode(changed_texts), changed_ref)
            row = {
                "lengths": lengths,
                "token_sha256": hashlib.sha256(ids.tobytes() + actual.tobytes()).hexdigest(),
                "samples_seconds": samples,
                "normal_runtime_samples": runtime_samples,
                "vector_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "cold_seconds": cold,
                "exact_equality": True,
                "baseline_over_plan": float(
                    np.median(samples["baseline_a"] + samples["baseline_b"])
                    / np.median(samples["plan"])
                ),
                "baseline_a_over_b": float(
                    np.median(samples["baseline_a"]) / np.median(samples["baseline_b"])
                ),
                "block_ratios": ratios,
                "median_block_ratio": float(np.median(ratios)),
                "block_bootstrap_95pct": interval(ratios),
                "before": before,
                "after": after,
                "p95_seconds": {
                    label: float(np.quantile(values, 0.95)) for label, values in samples.items()
                },
            }
            payload["cases"].append(row)
            save_json(args.output, payload)
            print(
                {
                    k: row[k]
                    for k in [
                        "lengths",
                        "baseline_over_plan",
                        "baseline_a_over_b",
                        "block_bootstrap_95pct",
                    ]
                },
                flush=True,
            )
            model.trim_memory()
            assert rt.diagnostics()["plan_cache_bytes"] == 0
        rt._plans_enabled = True
    assert rt.diagnostics()["active_bytes"] == rt.diagnostics()["plan_cache_bytes"] == 0
    payload["status"] = "passed"
    assert payload["sources"] == source_hashes()
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
