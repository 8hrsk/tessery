"""Paired full Qwen API check of the experimental four-row-group traversal."""

import argparse
import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from benchmark_quantized_traversal import BASE, experimental_runtime, install_route
from diagnose_metal import save_json, source_hashes, summarize

from tessery import EmbeddingModel


def factory(**kwargs):
    assert kwargs == {"workspace_limit_bytes": 64 * 1024 * 1024}
    return experimental_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100:
        parser.error("samples must be 3..100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_quantized_traversal.py").read_bytes()
        ).hexdigest(),
        "conditions": (
            "Full unprofiled encode API; 3 warmup rounds; paired randomized order; "
            "baseline A/B identical; group4 only for gate/up shapes; uncontrolled thermals"
        ),
        "samples": args.samples,
        "reverse_cases": args.reverse_cases,
        "results": [],
    }
    try:
        with patch("metal_inference.qwen3.MetalRuntime", factory):
            model = EmbeddingModel.load(args.model_dir)
        with model:
            rt = model._backend.runtime
            rt._plans_enabled = False  # These experiments mutate kernel routing between calls.
            selection = install_route(rt)
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            payload["experimental_shader_sha256"] = rt.diagnostics()["shader_sha256"]
            cases = [128, 512]
            if args.reverse_cases:
                cases.reverse()
            for length in cases:
                texts = [" token" * (length - 1)]
                ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert lengths.tolist() == [length]
                timings = {name: [] for name in ("baseline_a", "baseline_b", "group4")}
                rng = np.random.default_rng(1900 + length)
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(timings)):
                        selection["kernel"] = (
                            BASE if name.startswith("baseline") else BASE + "_group4"
                        )
                        started = time.perf_counter()
                        outputs[name] = model.encode(texts)
                        elapsed = time.perf_counter() - started
                        if iteration >= 3:
                            timings[name].append(elapsed)
                    for output in outputs.values():
                        np.testing.assert_array_equal(output, outputs["baseline_a"])
                        difference = max(
                            difference, float(np.max(np.abs(output - outputs["baseline_a"])))
                        )
                a, b, candidate = [float(np.median(timings[k])) for k in timings]
                base = float(np.median(timings["baseline_a"] + timings["baseline_b"]))
                row = {
                    "length": length,
                    "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "max_abs_difference": difference,
                    "timings": {k: summarize(v, 1) for k, v in timings.items()},
                    "speedup": base / candidate,
                    "identical_control_a_over_b": a / b,
                    "control_within_10_percent": 0.9 <= a / b <= 1.1,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print({k: v for k, v in row.items() if k != "timings"}, flush=True)
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
