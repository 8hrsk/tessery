"""Paired public-API timings of identical texts with execution padding policies."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize

import metal_inference.api as api
from metal_inference.batching import length_batches
from tessery import EmbeddingModel, ModelProfile


def candidate_batches(lengths, budget, max_length, architecture, alignment):
    for rows, width in length_batches(lengths, budget):
        aligned = (width + alignment - 1) // alignment * alignment
        if aligned <= min(max_length, 2 * int(lengths[rows].min()), budget // len(rows)):
            width = aligned
        yield rows, width


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[7, 9, 12, 15, 31, 33, 63, 65, 127, 129]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100 or any(n < 3 or n > 512 for n in args.lengths):
        parser.error("samples must be 3..100 and lengths 3..512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": "paired randomized order; 3 warmups; full encode API; uncontrolled thermals",
        "results": [],
    }
    selected = api.execution_batches
    policies = {
        "previous": lambda lengths, budget, limit, arch: length_batches(lengths, budget),
        "selected": selected,
    }
    if args.explore:
        policies = {
            "previous": policies["previous"],
            "eight": lambda *a: candidate_batches(*a, alignment=8),
            "thirtytwo": lambda *a: candidate_batches(*a, alignment=32),
        }
    try:
        profile = (
            ModelProfile.from_file(args.profile_file)
            if args.profile_file
            else "qwen3-embedding-0.6b-dwq"
        )
        with EmbeddingModel.load(args.model_dir, profile=profile) as model:
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            rng = np.random.default_rng(428)
            special = 2 if model.descriptor.architecture == "bert_f32" else 1
            cases = [[n] for n in args.lengths] + [[7] * 3, [33] * 8, [127, 65, 33]]
            for lengths in cases:
                texts = [" token" * (n - special) for n in lengths]
                _, actual = model._tokenizer.batch(texts, max_length=model.max_length)
                assert actual.tolist() == lengths
                plans = {
                    name: [
                        (r.tolist(), w)
                        for r, w in planner(
                            actual,
                            model._backend.max_padded_tokens,
                            model.max_length,
                            model.descriptor.architecture,
                        )
                    ]
                    for name, planner in policies.items()
                }
                samples = {name: [] for name in policies}
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(policies)):
                        api.execution_batches = policies[name]
                        started = time.perf_counter()
                        outputs[name] = model.encode(texts)
                        if iteration >= 3:
                            samples[name].append(time.perf_counter() - started)
                    for name in policies:
                        np.testing.assert_allclose(
                            outputs[name], outputs["previous"], atol=5e-6, rtol=1e-4
                        )
                        difference = max(
                            difference, float(np.max(np.abs(outputs[name] - outputs["previous"])))
                        )
                row = {
                    "lengths": lengths,
                    "plans": plans,
                    "max_abs_error": difference,
                    "timings": {k: summarize(v, len(texts)) for k, v in samples.items()},
                    "speedup": {
                        k: float(np.median(samples["previous"]) / np.median(v))
                        for k, v in samples.items()
                    },
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print({k: v for k, v in row.items() if k not in {"timings", "plans"}}, flush=True)
            payload["runtime"] = model._backend.runtime.diagnostics()
        assert payload["source_hashes"] == source_hashes()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        api.execution_batches = selected
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
