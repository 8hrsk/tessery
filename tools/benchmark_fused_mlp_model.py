"""Paired full Qwen API check of experimental fused gate/up/SiLU kernels."""

import argparse
import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize
from fused_mlp_experiment import KERNELS, experimental_runtime, hashes, install_model_route

from tessery import EmbeddingModel, MetalRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--kernel", choices=(*KERNELS, "selected"), default="fused_mlp_parallel")
    parser.add_argument("--lengths", type=int, nargs="+", default=[7, 128, 512])
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100 or any(n not in (7, 128, 129, 256, 512) for n in args.lengths):
        parser.error("samples 3..100; lengths from 7,128,129,256,512")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_hashes": hashes(),
        "kernel": args.kernel,
        "conditions": (
            "Full unprofiled encode API; 3 warmup rounds; paired randomized order; "
            "baseline A/B identical; fused aligned gate/up/SiLU; "
            "up allocation retained; uncontrolled thermals"
        ),
        "samples": args.samples,
        "reverse_cases": args.reverse_cases,
        "results": [],
    }
    try:
        with patch(
            "metal_inference.qwen3.MetalRuntime",
            MetalRuntime if args.kernel == "selected" else experimental_runtime,
        ):
            model = EmbeddingModel.load(args.model_dir)
        with model:
            rt = model._backend.runtime
            rt._plans_enabled = False  # These experiments mutate kernel routing between calls.
            selection = install_model_route(model)
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            payload["runtime_shader_sha256"] = rt.diagnostics()["shader_sha256"]
            payload["runtime_library"] = (
                "production" if args.kernel == "selected" else "production_plus_candidate_shaders"
            )
            cases = list(args.lengths)
            if args.reverse_cases:
                cases.reverse()
            for length in cases:
                texts = [" token" * (length - 1)]
                ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert lengths.tolist() == [length]
                timings = {name: [] for name in ("baseline_a", "baseline_b", "candidate")}
                rng = np.random.default_rng(1900 + length)
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(timings)):
                        assert selection["pending"] is None and selection["skip"] is None
                        selection["kernel"] = None if name.startswith("baseline") else args.kernel
                        dispatch_name = (
                            "gated4_16x32_k64" if args.kernel == "selected" else args.kernel
                        )
                        before = rt.diagnostics()["dispatches"]
                        started = time.perf_counter()
                        outputs[name] = model.encode(texts)
                        elapsed = time.perf_counter() - started
                        assert selection["pending"] is None and selection["skip"] is None
                        after = rt.diagnostics()["dispatches"]
                        # These singleton cases have the listed execution heights.
                        # Experimental fused kernels support complete 16-row tiles.
                        expected_rows = (128, 256, 512)
                        expected = 28 if name == "candidate" and length in expected_rows else 0
                        assert (
                            after.get(dispatch_name, 0) - before.get(dispatch_name, 0) == expected
                        )
                        assert (
                            after.get("silu_gate", 0) - before.get("silu_gate", 0) == 28 - expected
                        )
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
        assert payload["experiment_hashes"] == hashes()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
