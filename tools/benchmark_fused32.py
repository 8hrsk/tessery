"""Compare 32-row fused tiles with the current 16-row gate/up/SiLU kernel."""

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

KERNELS = ("gated4_32x32_k64", "gated4_32x32_wide_k64")


def fused(rt, name, buffers, *, rows, cols, k):
    tile = 32 if name in KERNELS else 16
    group = 512 if name == "gated4_32x32_wide_k64" else 256
    rt._dispatch(
        name,
        buffers,
        threads=(rows // tile) * (cols // 32) * group,
        group_size=group,
        rows=rows,
        cols=cols,
        k=k,
    )


def measure(rt, arrays, x, args, seed):
    m, k = x.shape
    n = arrays[0].shape[0]
    names = ("baseline_a", "baseline_b", *KERNELS)
    buffers = [rt.buffer(a.nbytes, a) for a in (x, *arrays)]
    sentinel = np.full((m + 2, n), np.nan, dtype=np.float32)
    out = rt.buffer(sentinel.nbytes, sentinel)
    up = rt.buffer(sentinel.nbytes, sentinel)
    gpu = {name: [] for name in names}
    wall = {name: [] for name in names}
    warmups = {}

    def invoke(name, repeats):
        with rt.command():
            for _ in range(repeats):
                kernel = "gated4_16x32_k64" if name.startswith("baseline") else name
                fused(rt, kernel, [*buffers, out], rows=m, cols=n, k=k)

    try:
        if not args.validate_only:
            for name in names:
                started = time.perf_counter()
                count = 0
                while count < 20 or time.perf_counter() - started < 0.1:
                    invoke(name, 1)
                    count += 1
                warmups[name] = count
            rng = np.random.default_rng(seed)
            for _ in range(args.samples):
                for name in rng.permutation(names):
                    before = rt.diagnostics()
                    started = time.perf_counter()
                    invoke(name, args.repeats)
                    elapsed = (time.perf_counter() - started) / args.repeats
                    after = rt.diagnostics()
                    assert after["gpu_timed_commands"] == before["gpu_timed_commands"] + 1
                    gpu[name].append((after["gpu_seconds"] - before["gpu_seconds"]) / args.repeats)
                    wall[name].append(elapsed)
        # Independently check both matmuls, then their F64 gated result.
        gate64 = x.astype(np.float64) @ dense_weights(arrays[:3]).astype(np.float64).T
        up64 = x.astype(np.float64) @ dense_weights(arrays[3:]).astype(np.float64).T
        reference = (gate64 / (1 + np.exp(-gate64))) * up64
        with rt.command():
            rt._linear4([*buffers[:4], out], rows=m, cols=n, k=k)
            rt._linear4([buffers[0], *buffers[4:], up], rows=m, cols=n, k=k)
        for buffer, expected in ((out, gate64), (up, up64)):
            np.testing.assert_allclose(
                rt.read(buffer, sentinel.shape)[:m], expected, atol=5e-5, rtol=5e-5
            )
        baseline = None
        errors = {}
        for name in names:
            out.close()
            out = rt.buffer(sentinel.nbytes, sentinel)
            invoke(name, 1)
            actual = rt.read(out, sentinel.shape)
            assert np.isnan(actual[m:]).all()
            np.testing.assert_allclose(actual[:m], reference, atol=5e-5, rtol=5e-5)
            if baseline is None:
                baseline = actual[:m].copy()
            np.testing.assert_array_equal(actual[:m], baseline)
            errors[name] = float(np.max(np.abs(actual[:m] - reference)))
        row = {
            "shape_m_n_k": [m, n, k],
            "input_sha256": digest(x),
            "weight_hashes": [digest(a) for a in arrays],
            "exact_baseline_equality": True,
            "output_guard_untouched": True,
            "max_abs_error_vs_f64": errors,
            "warmups": warmups,
        }
        if not args.validate_only:
            medians = {name: float(np.median(values)) for name, values in gpu.items()}
            assert all(np.isfinite(v) and v > 0 for v in medians.values())
            control = medians["baseline_a"] / medians["baseline_b"]
            baseline_time = float(np.median(gpu["baseline_a"] + gpu["baseline_b"]))
            row.update(
                gpu_timings={k: summarize(v, 1) for k, v in gpu.items()},
                wall_timings={k: summarize(v, 1) for k, v in wall.items()},
                gpu_speedup={name: baseline_time / medians[name] for name in names},
                identical_control_a_over_b=control,
                control_within_10_percent=0.9 <= control <= 1.1,
            )
        return row
    finally:
        for buffer in (*buffers, out, up):
            buffer.close()
        assert rt.active_bytes == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 160, 256, 512])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 3 <= args.samples <= 100
        or not 1 <= args.repeats <= 100
        or any(m < 32 or m > 4096 or m % 32 for m in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("samples 3..100; repeats 1..100; unique rows divisible by 32, up to 4096")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark_mlp_isolated.py", "diagnose_metal.py")
        },
        "profile_sha256": QWEN3_PROFILE.identity_sha256,
        "samples": args.samples,
        "dispatch_repeats_per_command": args.repeats,
        "validate_only": args.validate_only,
        "reverse_cases": args.reverse_cases,
        "shader_validation": os.getenv("MTL_SHADER_VALIDATION"),
        "cpu_reference_thread_limits": {
            k: os.getenv(k) for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "conditions": (
            "20 calls/100ms minimum warmup; randomized paired order; baseline A/B identical; "
            "GPU time per current16/candidate32 fused gate/up/SiLU; "
            "host validation outside timing; uncontrolled thermals"
        ),
        "results": [],
    }
    try:
        weights = load_projections(args.model_dir, QWEN3_PROFILE)
        arrays = weights["model.layers.0.mlp.gate_proj"] + weights["model.layers.0.mlp.up_proj"]
        order = np.random.default_rng(2101).permutation(len(args.rows))
        if args.reverse_cases:
            order = order[::-1]
        with MetalRuntime() as rt:
            for index in order:
                m = args.rows[index]
                x = (
                    np.random.default_rng(2100 + int(index))
                    .normal(size=(m, 1024))
                    .astype(np.float32)
                )
                row = measure(rt, arrays, x, args, 2200 + int(index))
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {
                        k: v
                        for k, v in row.items()
                        if k in ("shape_m_n_k", "gpu_speedup", "identical_control_a_over_b")
                    },
                    flush=True,
                )
            payload["runtime"] = rt.diagnostics()
        assert payload["source_hashes"] == source_hashes() and all(
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == value
            for name, value in payload["helper_sha256"].items()
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
