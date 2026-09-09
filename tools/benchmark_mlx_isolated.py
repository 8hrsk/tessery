"""Offline Qwen comparison: each engine runs alone in a fresh process.

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
from metal_inference.profiles import QWEN3_PROFILE
from metal_inference.tokenizer import QwenTokenizer, validate_qwen_profile
from metal_inference.weights import read_json
from tessery import EmbeddingModel


def process_memory():
    rss = int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]))
    return {
        "rss_bytes": rss * 1024,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def worker(args):
    if args.engine == "mlx":
        from benchmark_mlx import MLXReference

    before_load = process_memory()
    started = time.perf_counter()
    if args.engine == "mlx":
        data = read_json(args.model_dir, "tokenizer.json")
        validate_qwen_profile(data)
        model = EmbeddingModel(
            MLXReference(args.model_dir),
            QwenTokenizer(data),
            QWEN3_PROFILE.default_dimensions,
            QWEN3_PROFILE.max_length,
            8,
            ModelDescriptor.from_profile(QWEN3_PROFILE),
        )
    else:
        model = EmbeddingModel.load(args.model_dir)
    payload = {
        "engine": args.engine,
        "memory_before_load": before_load,
        "load_seconds": time.perf_counter() - started,
        "memory_loaded": process_memory(),
        "source_hashes": source_hashes(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "results": [],
    }
    try:
        cases = [
            [" token" * (tokens - 1)] * batch
            for batch, tokens in [(1, 7), (1, 8), (1, 31), (4, 33)]
        ] + [["Hello world", "Кошки и собаки", "A much longer passage about Paris in France."]]
        for case_id in np.random.default_rng(94).permutation(len(cases)):
            texts = cases[case_id]
            _, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
            for _ in range(3):
                model.encode(texts)
            latencies = []
            reference = None
            for _ in range(args.samples):
                started = time.perf_counter()
                output = model.encode(texts)
                latencies.append(time.perf_counter() - started)
                if reference is None:
                    reference = output
                else:
                    np.testing.assert_array_equal(output, reference)
                assert np.isfinite(output).all()
            payload["results"].append(
                {
                    "case": int(case_id),
                    "lengths": lengths.tolist(),
                    **summarize(latencies, len(texts)),
                    "vectors": reference.tolist(),
                    "memory": process_memory(),
                }
            )
            print(f"{args.engine}: case {case_id} complete", flush=True)
        if args.engine == "mlx":
            import mlx.core as mx

            payload["mlx_version"] = importlib.metadata.version("mlx")
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
    parser.add_argument("--mlx-python", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=30, type=int)
    parser.add_argument("--engine", choices=["tessery", "mlx"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 5 <= args.samples <= 100:
        parser.error("samples must be 5..100")
    if args.engine:
        worker(args)
        return
    if args.mlx_python is None:
        parser.error("--mlx-python must point to an existing MLX environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "platform": platform.platform(),
        "scope": "isolated_process_full_qwen_public_api",
        "baseline": "independent public MLX F32 graph; not mlx-embeddings",
        "conditions": (
            "sequential fresh processes; 3 warmups; same seeded case order; uncontrolled thermals"
        ),
        "memory_note": (
            "RSS and peak RSS belong to one engine process each; peaks include load; "
            "allocator accounting differs; reference promotes BF16 weights to F32; "
            "load_seconds excludes framework imports; file caches are uncontrolled"
        ),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_mlx.py").read_bytes()
        ).hexdigest(),
        "profile_sha256": QWEN3_PROFILE.identity_sha256,
        "workers": {},
        "results": [],
    }
    try:
        import json

        for engine, interpreter in [("tessery", sys.executable), ("mlx", args.mlx_python)]:
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
                ],
                env=environment,
                check=True,
            )
            payload["workers"][engine] = json.loads(result.read_text())
            save_json(args.output, payload)
        left, right = [payload["workers"][name] for name in ("tessery", "mlx")]
        assert left["source_hashes"] == right["source_hashes"] == source_hashes()
        for a, b in zip(left["results"], right["results"], strict=True):
            assert a["case"] == b["case"] and a["lengths"] == b["lengths"]
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
