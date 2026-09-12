"""Resident uint4 down projection: old32 versus isolated tile/layout experiments."""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
from benchmark_mlp_isolated import dense_weights, digest, load_projections
from diagnose_metal import save_json, source_hashes, summarize

from metal_inference.profiles import QWEN3_PROFILE
from tessery import MetalRuntime

KERNELS = {
    "linear4_32x32_k64": (32, 256),
    "linear4_32x32_transposed_k64": (32, 256),
    "linear4_64x32_wide_k64": (64, 512),
}


def dispatch(rt, name, buffers, m, n, k):
    tile, group = KERNELS[name]
    assert m % tile == n % 32 == k % 64 == 0
    rt._dispatch(
        name,
        buffers,
        threads=m // tile * (n // 32) * group,
        group_size=group,
        rows=m,
        cols=n,
        k=k,
    )


def measure(rt, arrays, x, kernels, samples=24, repeats=5, validate_only=False):
    m, k = x.shape
    n = arrays[0].shape[0]
    labels = {"baseline_a": "linear4_32x32_k64", "baseline_b": "linear4_32x32_k64"}
    labels.update({name: name for name in kernels})
    buffers = [rt.buffer(a.nbytes, a) for a in (x, *arrays)]
    out = None
    sentinel = np.full((m + 2, n), np.nan, np.float32)
    sentinel[m:] = -918.25
    expected = x.astype(np.float64) @ dense_weights(arrays).astype(np.float64).T
    outputs, errors = {}, {}
    timings = {name: [] for name in labels}
    gpu = {name: [] for name in labels}
    warmups = {}
    rng = np.random.default_rng(91513 + m)
    try:
        for label, kernel in labels.items():
            if out is not None:
                out.close()
            out = rt.buffer(sentinel.nbytes, sentinel)
            with rt.command():
                dispatch(rt, kernel, [*buffers, out], m, n, k)
            result = rt.read(out, sentinel.shape)
            np.testing.assert_array_equal(result[m:], sentinel[m:], err_msg=label)
            np.testing.assert_allclose(result[:m], expected, atol=5e-5, rtol=5e-5)
            outputs[label] = result[:m].copy()
            errors[label] = float(np.max(np.abs(result[:m] - expected)))
        for result in outputs.values():
            np.testing.assert_array_equal(result, outputs["baseline_a"])
        # Also verify read-only inputs/weights after every candidate was exercised.
        for buffer, array in zip(buffers, (x, *arrays), strict=True):
            actual = rt.read(buffer, (array.nbytes // 4,)).view(array.dtype).reshape(array.shape)
            np.testing.assert_array_equal(actual, array)
        if not validate_only:
            for label, kernel in labels.items():
                started, count = time.perf_counter(), 0
                while count < 20 or time.perf_counter() - started < 0.1:
                    with rt.command():
                        dispatch(rt, kernel, [*buffers, out], m, n, k)
                    count += 1
                warmups[label] = count
            orders = []
            for _ in range(samples):
                order = rng.permutation(list(labels)).tolist()
                orders.append(order)
                for label in order:
                    before = rt.diagnostics()
                    start = time.perf_counter()
                    with rt.command():
                        for _ in range(repeats):
                            dispatch(rt, labels[label], [*buffers, out], m, n, k)
                    timings[label].append((time.perf_counter() - start) / repeats)
                    after = rt.diagnostics()
                    assert after["gpu_timed_commands"] == before["gpu_timed_commands"] + 1
                    gpu[label].append((after["gpu_seconds"] - before["gpu_seconds"]) / repeats)
        row = {
            "shape_m_n_k": [m, n, k],
            "input_sha256": digest(x),
            "weight_hashes": [digest(a) for a in arrays],
            "exact_baseline_equality": True,
            "output_guard_untouched": True,
            "inputs_unchanged": True,
            "max_abs_error_vs_f64": errors,
        }
        if not validate_only:
            baseline = np.median(gpu["baseline_a"] + gpu["baseline_b"])
            control = float(np.median(gpu["baseline_a"]) / np.median(gpu["baseline_b"]))
            row.update(
                warmups=warmups,
                orders=orders,
                gpu_timings={n: summarize(t, 1) for n, t in gpu.items()},
                wall_timings={n: summarize(t, 1) for n, t in timings.items()},
                gpu_speedup={n: float(baseline / np.median(gpu[n])) for n in kernels},
                identical_control_a_over_b=control,
                control_within_2_percent=0.98 <= control <= 1.02,
            )
        return row
    finally:
        if out is not None:
            out.close()
        for buffer in buffers:
            buffer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument(
        "--kernels", nargs="+", choices=list(KERNELS)[1:], default=list(KERNELS)[1:]
    )
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 24 <= args.samples <= 96 or any(m not in (64, 128, 256, 512) for m in args.rows):
        parser.error("samples 24..96 and complete rows 64/128/256/512")
    profile = QWEN3_PROFILE
    arrays = load_projections(args.model_dir, profile)["model.layers.0.mlp.down_proj"]
    helpers = [
        Path(__file__),
        Path(__file__).with_name("benchmark_mlp_isolated.py"),
        Path(__file__).with_name("diagnose_metal.py"),
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "helper_hashes": hashes,
        "profile_sha256": profile.identity_sha256,
        "samples": args.samples,
        "dispatch_repeats": 5,
        "reverse": args.reverse,
        "conditions": (
            "same resident buffers; randomized labels; 20 calls/100ms warmup; "
            "sequential GPU; uncontrolled thermals"
        ),
        "thread_limits": {
            k: os.getenv(k) for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    try:
        with MetalRuntime() as rt:
            for m in args.rows[::-1] if args.reverse else args.rows:
                x = np.random.default_rng(9913 + m).normal(size=(m, 3072)).astype(np.float32)
                row = measure(rt, arrays, x, args.kernels, samples=args.samples)
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {
                        k: row[k]
                        for k in ("shape_m_n_k", "gpu_speedup", "identical_control_a_over_b")
                    },
                    flush=True,
                )
        assert rt.active_bytes == rt.cache_bytes == 0
        assert source_hashes() == payload["source_hashes"]
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
        payload["status"] = "passed"
    except BaseException as error:
        payload.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
