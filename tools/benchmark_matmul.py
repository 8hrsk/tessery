"""Paired offline comparison of original F32 dot and tiled kernels."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from metal_inference import MetalRuntime


def speedup(baseline, candidate):
    if not baseline or not candidate or min(baseline) <= 0 or min(candidate) <= 0:
        return None
    return float(np.median(baseline) / np.median(candidate))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(83)
    results = []
    for m, n, k in [(8, 384, 384), (32, 384, 384), (128, 1536, 384), (128, 384, 1536)]:
        with MetalRuntime() as rt:
            a = rng.normal(size=(m, k)).astype(np.float32)
            bt = rng.normal(size=(n, k)).astype(np.float32)
            buffers = [rt.buffer(a.nbytes, a), rt.buffer(bt.nbytes, bt), rt.buffer(m * n * 4)]
            reference = a.astype(np.float64) @ bt.astype(np.float64).T
            samples = {name: [] for name in ("matmul_f32", "matmul_f32_tiled")}
            gpu = {name: [] for name in samples}
            errors = {}
            accepted = {}
            for iteration in range(22):
                for name in rng.permutation(list(samples)):
                    tiled = name.endswith("tiled")
                    before = rt.diagnostics()
                    start = time.perf_counter()
                    with rt.command():
                        rt._dispatch(
                            str(name),
                            buffers,
                            threads=(m // 8) * (n // 32) * 128
                            if tiled
                            else ((m + 3) // 4) * n * 32,
                            group_size=128 if tiled else 32,
                            rows=m,
                            cols=n,
                            k=k,
                        )
                    wall = time.perf_counter() - start
                    after = rt.diagnostics()
                    if iteration >= 2:
                        samples[name].append(wall)
                        if after["gpu_timed_commands"] - before["gpu_timed_commands"] == 1:
                            gpu[name].append(after["gpu_seconds"] - before["gpu_seconds"])
                    out = rt.read(buffers[-1], (m, n))
                    # Record rejected candidates as evidence; do not silently
                    # relax the threshold to turn a benchmark into a success.
                    passed = bool(np.allclose(out, reference, atol=5e-5, rtol=5e-5))
                    accepted[name] = accepted.get(name, True) and passed
                    errors[name] = max(errors.get(name, 0), float(np.max(np.abs(out - reference))))
                    if name == "matmul_f32" or k <= 512:
                        assert passed
            results.append(
                {
                    "shape_m_n_k": [m, n, k],
                    "wall_seconds": samples,
                    "gpu_seconds": gpu,
                    "max_abs_error_vs_f64": errors,
                    "within_atol_5e_minus5_rtol_5e_minus5": accepted,
                    "tiled_short_k_route": k <= 512,
                    "gpu_median_speedup": speedup(gpu["matmul_f32"], gpu["matmul_f32_tiled"]),
                    "loaded_runtime": rt.diagnostics(),
                }
            )
            print(results[-1]["shape_m_n_k"], results[-1]["gpu_median_speedup"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shader = Path(__file__).resolve().parents[1] / "src/metal_inference/native/kernels.metal"
    with args.output.open("x") as stream:
        json.dump(
            {
                "scope": "single_host_paired_kernel_diagnostic",
                "platform": platform.platform(),
                "source_sha256": hashlib.sha256(shader.read_bytes()).hexdigest(),
                "conditions": (
                    "randomized order, 2 warmup + 20 paired samples, uncontrolled power/thermal"
                ),
                "results": results,
            },
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
