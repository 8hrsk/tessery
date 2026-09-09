"""Offline Qwen/BGE comparison: each engine runs alone in a fresh process.

Requires an existing MLX interpreter for the development-only reference graph.
The library environment stays independent of MLX. No downloads are performed.
"""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from diagnose_metal import ROOT, save_json, source_hashes, summarize

from metal_inference.api import ModelDescriptor
from metal_inference.batching import execution_batches
from metal_inference.profiles import QWEN3_PROFILE
from metal_inference.tokenizer import QwenTokenizer, validate_qwen_profile
from metal_inference.weights import read_json
from metal_inference.wordpiece import WordPieceTokenizer
from tessery import EmbeddingModel, ModelProfile


def process_memory():
    rss = int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]))
    return {
        "rss_bytes": rss * 1024,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def worker(args):
    profile = ModelProfile.from_file(args.profile_file) if args.profile_file else QWEN3_PROFILE
    bert = profile.architecture == "bert_f32"
    if args.engine == "mlx":
        if bert:
            from benchmark_mlx_bert import MLXBertReference
        else:
            from benchmark_mlx import MLXReference

    before_load = process_memory()
    started = time.perf_counter()
    if args.engine == "mlx":
        data = read_json(args.model_dir, "tokenizer.json", profile=profile)
        if bert:
            backend = MLXBertReference(args.model_dir, profile)
            tokenizer = WordPieceTokenizer(data)
        else:
            assert profile.identity_sha256 == QWEN3_PROFILE.identity_sha256
            validate_qwen_profile(data)
            backend = MLXReference(args.model_dir, causal_fast_path=args.mlx_mask == "causal")
            tokenizer = QwenTokenizer(data)
        model = EmbeddingModel(
            backend,
            tokenizer,
            profile.default_dimensions,
            profile.max_length,
            8,
            ModelDescriptor.from_profile(profile),
        )
    else:
        model = EmbeddingModel.load(args.model_dir, profile=profile)
    payload = {
        "engine": args.engine,
        "model_id": model.descriptor.model_id,
        "compatibility_id": model.descriptor.compatibility_id,
        "profile_sha256": profile.identity_sha256,
        "regex_version": importlib.metadata.version("regex"),
        "memory_before_load": before_load,
        "load_seconds": time.perf_counter() - started,
        "memory_loaded": process_memory(),
        "source_hashes": source_hashes(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": [],
    }
    try:
        if bert:
            import json

            frozen_path = ROOT / "benchmarks/observations/bge-small-en-v1.5/reference.json"
            frozen = json.loads(frozen_path.read_text())
            differences = []
            for row in frozen["batches"]:
                vectors = model.encode(row["texts"])
                expected = np.array(row["vectors"], np.float32)
                np.testing.assert_allclose(vectors, expected, atol=5e-6, rtol=1e-4)
                differences.append(float(np.max(np.abs(vectors - expected))))
            payload["frozen_cpu_validation"] = {
                "sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                "batches": len(differences),
                "max_abs_difference": max(differences),
            }
        special = 2 if bert else 1
        cases = [
            [" token" * (tokens - special)] * batch
            for batch, tokens in [(1, 7), (1, 8), (1, 31), (4, 33)]
        ] + [["Hello world", "Кошки и собаки", "A much longer passage about Paris in France."]]
        if args.lengths:
            cases = [[" token" * (n - special)] for n in args.lengths]
        if args.include_batches:
            cases += [
                [" token" * (33 - special)] * 4,
                [" token" * (n - special) for n in (3, 7, 10)],
            ]
        order = np.random.default_rng(94).permutation(len(cases))
        if args.reverse_cases:
            order = order[::-1]
        for case_id in order:
            texts = cases[case_id]
            ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
            warm_start = time.perf_counter()
            warmups = 0
            while warmups < 3 or (args.paired_controls and time.perf_counter() - warm_start < 0.1):
                model.encode(texts)
                warmups += 1
            controls = {label: [] for label in (["a", "b"] if args.paired_controls else ["a"])}
            reference = None
            rng = np.random.default_rng(1000 + int(case_id))
            for _ in range(args.samples):
                for label in rng.permutation(list(controls)):
                    started = time.perf_counter()
                    output = model.encode(texts)
                    controls[label].append(time.perf_counter() - started)
                    if reference is None:
                        reference = output
                    else:
                        np.testing.assert_array_equal(output, reference)
                    assert np.isfinite(output).all()
            latencies = [value for values in controls.values() for value in values]
            control_ratio = (
                float(np.median(controls["a"]) / np.median(controls["b"]))
                if args.paired_controls
                else None
            )
            payload["results"].append(
                {
                    "case": int(case_id),
                    "warmups": warmups,
                    "control_timings": {k: summarize(v, len(texts)) for k, v in controls.items()},
                    "identical_control_a_over_b": control_ratio,
                    "control_within_10_percent": (
                        0.9 <= control_ratio <= 1.1 if control_ratio is not None else None
                    ),
                    "lengths": lengths.tolist(),
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "plans": [
                        (r.tolist(), w)
                        for r, w in execution_batches(
                            lengths,
                            model._backend.max_padded_tokens,
                            model.max_length,
                            model.descriptor.architecture,
                        )
                    ],
                    **summarize(latencies, len(texts)),
                    "vectors": reference.tolist(),
                    "memory": process_memory(),
                }
            )
            print(f"{args.engine}: case {case_id} complete", flush=True)
        if args.engine == "mlx":
            import mlx.core as mx

            payload["mlx_version"] = importlib.metadata.version("mlx")
            payload["mlx_device"] = mx.device_info()
            payload["allocator"] = {
                "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory(),
                "peak_bytes": mx.get_peak_memory(),
            }
        else:
            payload["allocator"] = model._backend.runtime.diagnostics()
    finally:
        model.close()
    payload["memory_closed"] = process_memory()
    assert payload["source_hashes"] == source_hashes()
    save_json(args.output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--include-batches", action="store_true")
    parser.add_argument(
        "--paired-controls",
        action="store_true",
        help="Randomized identical A/B calls; samples per label",
    )
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--mlx-python", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=30, type=int)
    parser.add_argument(
        "--lengths", nargs="+", type=int, help="Override cases with uniform lengths"
    )
    parser.add_argument("--mlx-mask", choices=["dense", "causal"], default="dense")
    parser.add_argument(
        "--engine-order", choices=["tessery-first", "mlx-first"], default="tessery-first"
    )
    parser.add_argument("--engine", choices=["tessery", "mlx"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 5 <= args.samples <= 100:
        parser.error("samples must be 5..100")
    if args.lengths and not all(3 <= n <= 512 for n in args.lengths):
        parser.error("lengths must be 3..512")
    if args.engine:
        worker(args)
        return
    if args.mlx_python is None:
        parser.error("--mlx-python must point to an existing MLX environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    profile = ModelProfile.from_file(args.profile_file) if args.profile_file else QWEN3_PROFILE
    reference_file = (
        "benchmark_mlx_bert.py" if profile.architecture == "bert_f32" else "benchmark_mlx.py"
    )
    payload = {
        "status": "running",
        "platform": platform.platform(),
        "scope": "isolated_process_full_embedding_public_api",
        "baseline": "independent public MLX F32 graph; not mlx-embeddings",
        "conditions": (
            "sequential fresh processes; >=3 warmups (>=100ms with paired controls); "
            "seeded case order; samples per label; paired labels are identical; "
            "uncontrolled thermals"
        ),
        "memory_note": (
            "RSS and peak RSS belong to one engine process each; peaks include load; "
            "allocator accounting differs; Qwen BF16 metadata is promoted to F32; BGE stays F32; "
            "load_seconds excludes framework imports; file caches are uncontrolled"
        ),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            Path(__file__).with_name(reference_file).read_bytes()
        ).hexdigest(),
        "profile_sha256": profile.identity_sha256,
        "requested_lengths": args.lengths,
        "include_batches": args.include_batches,
        "reference_file": reference_file,
        "mlx_mask": args.mlx_mask,
        "engine_order": args.engine_order,
        "paired_controls": args.paired_controls,
        "reverse_cases": args.reverse_cases,
        "samples_per_label": args.samples,
        "workers": {},
        "results": [],
    }
    try:
        import json

        engines = [("tessery", sys.executable), ("mlx", args.mlx_python)]
        if args.engine_order == "mlx-first":
            engines.reverse()
        for engine, interpreter in engines:
            result = args.output.with_name(args.output.stem + f"-{engine}.json")
            with result.open("x") as stream:
                stream.write('{"status":"starting"}\n')
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            subprocess.run(
                [
                    str(interpreter),
                    str(Path(__file__).resolve()),
                    "--engine",
                    engine,
                    "--model-dir",
                    args.model_dir,
                    "--samples",
                    str(args.samples),
                    "--output",
                    str(result),
                    "--mlx-mask",
                    args.mlx_mask,
                    *(["--profile-file", args.profile_file] if args.profile_file else []),
                    *(["--include-batches"] if args.include_batches else []),
                    *(["--paired-controls"] if args.paired_controls else []),
                    *(["--reverse-cases"] if args.reverse_cases else []),
                    *(["--lengths", *map(str, args.lengths)] if args.lengths else []),
                ],
                env=environment,
                check=True,
            )
            payload["workers"][engine] = json.loads(result.read_text())
            save_json(args.output, payload)
        left, right = [payload["workers"][name] for name in ("tessery", "mlx")]
        assert left["source_hashes"] == right["source_hashes"] == source_hashes()
        assert left["profile_sha256"] == right["profile_sha256"] == profile.identity_sha256
        assert left["compatibility_id"] == right["compatibility_id"]
        for a, b in zip(left["results"], right["results"], strict=True):
            assert a["case"] == b["case"] and a["lengths"] == b["lengths"]
            assert a["plans"] == b["plans"] and a["input_ids_sha256"] == b["input_ids_sha256"]
            x, y = np.array(a["vectors"]), np.array(b["vectors"])
            np.testing.assert_allclose(x, y, atol=5e-6, rtol=1e-4)
            payload["results"].append(
                {
                    "case": a["case"],
                    "lengths": a["lengths"],
                    "max_abs_vector_difference": float(np.max(np.abs(x - y))),
                    "tessery_over_mlx_latency": a["p50_seconds"] / b["p50_seconds"],
                }
            )
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
