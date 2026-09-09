"""Offline resident MLP projection comparison with isolated, counterbalanced engines.

Uses verified first-layer weights and deterministic synthetic F32 activations.
Timing includes submission and synchronization per invocation, excluding host
readback, model loading and CPU validation. No runtime dependency on MLX.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from diagnose_metal import ROOT, save_json, source_hashes, summarize

from metal_inference.profiles import QWEN3_PROFILE
from metal_inference.weights import SafeTensors, read_artifact
from tessery import MetalRuntime, ModelProfile


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_projections(directory, profile):
    quantized = profile.architecture == "qwen3_uint4"
    if quantized:
        prefixes = [f"model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")]
    else:
        assert profile.architecture == "bert_f32"
        prefixes = [f"encoder.layer.0.{name}.dense" for name in ("intermediate", "output")]
    snapshot = SafeTensors(read_artifact(directory, "model.safetensors", profile=profile))
    result = {}
    for prefix in prefixes:
        arrays = []
        for suffix in ("weight", "scales", "biases") if quantized else ("weight",):
            name = prefix + "." + suffix
            info = snapshot.tensors[name]
            dtype = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32}[info.dtype]
            arrays.append(
                snapshot.view(name, shape=info.shape, dtype=info.dtype)
                .view(dtype)
                .reshape(info.shape)
                .copy()
            )
        result[prefix] = arrays
    return result


def dense_weights(arrays):
    if len(arrays) == 1:
        return arrays[0]
    w, scales, biases = arrays
    n, packed = w.shape
    codes = ((w[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(n, packed * 8)
    s = (scales.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
    b = (biases.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
    return codes.astype(np.float32) * s + b


def measure_case(engine, arrays, x, samples, repeats, rng):
    m, k = x.shape
    n = arrays[0].shape[0]
    quantized = len(arrays) == 3
    rt = None
    buffers = []
    last = None
    if engine == "mlx":
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        weights = [
            mx.array(a)
            if a.dtype != np.uint16
            else mx.array((a.astype(np.uint32) << 16).view(np.float32))
            for a in arrays
        ]
        inputs = mx.array(x)
        if not quantized:
            transposed = weights[0].T
            mx.eval(transposed)
        mx.eval(inputs, *weights)

        def invoke():
            nonlocal last
            last = (
                mx.quantized_matmul(inputs, *weights, transpose=True, group_size=64, bits=4)
                if quantized
                else mx.matmul(inputs, transposed)
            )
            mx.eval(last)  # A fresh graph and synchronous evaluation on every invocation.

        def read():
            return np.array(last)
    else:
        rt = MetalRuntime()
        buffers = [rt.buffer(a.nbytes, a) for a in [x, *arrays]]
        buffers.append(rt.buffer(m * n * 4))

        def invoke():
            with rt.command():
                dispatch = rt._linear4 if quantized else rt._matmul_f32
                dispatch(buffers, rows=m, cols=n, k=k)

        def read():
            return rt.read(buffers[-1], (m, n))

    try:
        warm_start = time.perf_counter()
        warmups = 0
        while warmups < 20 or time.perf_counter() - warm_start < 0.1:
            invoke()
            warmups += 1
        timings = {"a": [], "b": []}
        # A and B are the SAME callable. This control detects within-case drift.
        for _ in range(samples):
            for label in rng.permutation(["a", "b"]):
                started = time.perf_counter()
                for _ in range(repeats):
                    invoke()
                timings[label].append((time.perf_counter() - started) / repeats)
        output = read()
        assert output.shape == (m, n) and output.dtype == np.float32
        reference = x.astype(np.float64) @ dense_weights(arrays).astype(np.float64).T
        np.testing.assert_allclose(output, reference, atol=5e-5, rtol=5e-5)
        control = float(np.median(timings["a"]) / np.median(timings["b"]))
        row = {
            "shape_m_n_k": [m, n, k],
            "input_sha256": digest(x),
            "weight_hashes": [digest(a) for a in arrays],
            "warmups": warmups,
            "warm_seconds_minimum": 0.1,
            "timings": summarize(timings["a"] + timings["b"], 1),
            "control_timings": {label: summarize(values, 1) for label, values in timings.items()},
            "identical_control_a_over_b": control,
            "control_within_10_percent": 0.9 <= control <= 1.1,
            "max_abs_error_vs_f64": float(np.max(np.abs(output - reference))),
            "runtime": rt.diagnostics() if rt else {"device": str(mx.default_device())},
        }
        return row, output
    finally:
        if rt:
            for buffer in buffers:
                buffer.close()
            rt.close()
            assert rt.active_bytes == 0


def worker(args, profile):
    if args.engine == "mlx":
        import mlx.core as mx

    projections = load_projections(args.model_dir, profile)
    cases = [(prefix, m) for prefix in projections for m in args.rows]
    order = np.random.default_rng(1501).permutation(len(cases))
    if args.reverse_cases:
        order = order[::-1]
    payload = {
        "engine": args.engine,
        "profile_sha256": profile.identity_sha256,
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mlx_device": mx.device_info() if args.engine == "mlx" else None,
        "mlx_version": importlib.metadata.version("mlx") if args.engine == "mlx" else None,
        "results": {},
    }
    outputs = {}
    for index in order:
        prefix, m = cases[index]
        arrays = projections[prefix]
        k = arrays[0].shape[1] * (8 if len(arrays) == 3 else 1)
        # Case identity, not execution order, determines the inputs.
        x = np.random.default_rng(1500 + int(index)).normal(size=(m, k)).astype(np.float32)
        row, output = measure_case(
            args.engine,
            arrays,
            x,
            args.samples,
            args.repeats,
            np.random.default_rng(1700 + int(index)),
        )
        key = f"{prefix}:{m}"
        row["projection"] = prefix
        payload["results"][key] = row
        outputs[key] = output
        save_json(args.output, payload)
        print(
            f"{args.engine} {key}: {row['timings']['p50_seconds'] * 1000:.3f} ms; "
            f"A/B {row['identical_control_a_over_b']:.3f}",
            flush=True,
        )
    np.savez(args.vectors, **outputs)
    assert payload["source_hashes"] == source_hashes()
    if args.engine == "mlx":
        mx.clear_cache()
    payload["status"] = "passed"
    save_json(args.output, payload)


def stability_summary(workers):
    """Screen noisy timing evidence independently from numerical correctness."""
    results = []
    keys = set(workers[0]["results"])
    assert all(set(worker["results"]) == keys for worker in workers)
    for key in sorted(keys):
        rows = [worker["results"][key] for worker in workers]
        times = [row["timings"]["p50_seconds"] for row in rows]
        assert all(value > 0 and np.isfinite(value) for value in times)
        drifts = {
            "tessery": max(times[0], times[3]) / min(times[0], times[3]),
            "mlx": max(times[1], times[2]) / min(times[1], times[2]),
        }
        controls = all(row["control_within_10_percent"] for row in rows)
        results.append(
            {
                "case": key,
                "tessery_over_mlx_range": [
                    min(times[0] / times[1], times[3] / times[2]),
                    max(times[0] / times[1], times[3] / times[2]),
                ],
                "repeat_max_over_min": drifts,
                "all_identical_controls_within_10_percent": controls,
                "timing_screen_passed": controls
                and all(value <= 1.15 for value in drifts.values()),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--mlx-python", type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[8, 24, 40, 128, 129, 264, 512])
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["tessery", "mlx"], help=argparse.SUPPRESS)
    parser.add_argument("--reverse-cases", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vectors", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        not 3 <= args.samples <= 100
        or not 1 <= args.repeats <= 100
        or not all(1 <= m <= 4096 for m in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("samples 3..100; repeats 1..100; unique rows 1..4096")
    profile = ModelProfile.from_file(args.profile_file) if args.profile_file else QWEN3_PROFILE
    if args.engine:
        worker(args, profile)
        return
    if args.mlx_python is None:
        parser.error("--mlx-python must point to an existing MLX interpreter")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "platform": platform.platform(),
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_sha256": profile.identity_sha256,
        "scope": "first-layer resident matmul only; real weights, synthetic F32 activations; "
        "no bias, activation, full-model graph or readback in timing",
        "conditions": "isolated T/M/M/T processes; reversed case order for second pair; "
        "identical-operation A/B controls; per-invocation synchronization; "
        "uncontrolled thermals and background activity",
        "cpu_reference_thread_limits": {"VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "model_id": profile.model_id,
        "samples_per_control": args.samples,
        "invocations_per_sample": args.repeats,
        "workers": [],
        "comparisons": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="mlp-workers-", dir=args.output.parent) as work:
            paths = []
            for index, engine in enumerate(["tessery", "mlx", "mlx", "tessery"]):
                output = Path(work) / f"{index}.json"
                vectors = Path(work) / f"{index}.npz"
                env = dict(
                    os.environ,
                    PYTHONPATH=str(ROOT / "src"),
                    VECLIB_MAXIMUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                )
                subprocess.run(
                    [
                        str(args.mlx_python if engine == "mlx" else sys.executable),
                        str(Path(__file__).resolve()),
                        "--engine",
                        engine,
                        "--model-dir",
                        args.model_dir,
                        "--output",
                        str(output),
                        "--vectors",
                        str(vectors),
                        "--samples",
                        str(args.samples),
                        "--repeats",
                        str(args.repeats),
                        "--rows",
                        *map(str, args.rows),
                        *(["--profile-file", args.profile_file] if args.profile_file else []),
                        *(["--reverse-cases"] if index >= 2 else []),
                    ],
                    env=env,
                    check=True,
                )
                result = json.loads(output.read_text())
                assert result["status"] == "passed"
                assert result["source_hashes"] == payload["source_hashes"]
                assert result["harness_sha256"] == payload["harness_sha256"]
                assert result["profile_sha256"] == payload["profile_sha256"]
                payload["workers"].append(result)
                paths.append(vectors)
                save_json(args.output, payload)
            for left, right in [(0, 1), (3, 2)]:
                with np.load(paths[left]) as a, np.load(paths[right]) as b:
                    assert set(a.files) == set(b.files)
                    for key in a.files:
                        np.testing.assert_allclose(a[key], b[key], atol=5e-5, rtol=5e-5)
                        x = payload["workers"][left]["results"][key]
                        y = payload["workers"][right]["results"][key]
                        for field in ("input_sha256", "weight_hashes", "shape_m_n_k"):
                            assert x[field] == y[field]
                        payload["comparisons"].append(
                            {
                                "pair": 0 if left == 0 else 1,
                                "case": key,
                                "tessery_over_mlx": x["timings"]["p50_seconds"]
                                / y["timings"]["p50_seconds"],
                                "max_abs_difference": float(np.max(np.abs(a[key] - b[key]))),
                                "both_controls_within_10_percent": x["control_within_10_percent"]
                                and y["control_within_10_percent"],
                            }
                        )
        assert payload["source_hashes"] == source_hashes()
        payload["stability"] = stability_summary(payload["workers"])
        payload["status"] = "passed"  # Numerical/protocol success; timing has a separate screen.
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
