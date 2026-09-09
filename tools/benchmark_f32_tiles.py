"""Paired previous/selected F32 routes for all BGE projection shapes, checked in F64."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize

from tessery import MetalRuntime


def previous(rt, buffers, *, rows, cols, k):
    tiled = rows >= 8 and rows % 8 == 0 and cols % 32 == 0 and k % 8 == 0 and k <= 512
    rt._dispatch(
        "matmul_f32_tiled" if tiled else "matmul_f32",
        buffers,
        threads=((rows + 7) // 8) * (cols // 32) * 128 if tiled else ((rows + 3) // 4) * cols * 32,
        group_size=128 if tiled else 32,
        rows=rows,
        cols=cols,
        k=k,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--rows", type=int, nargs="+", default=[7, 8, 9, 15, 32, 128, 512])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100 or any(m < 1 or m > 4096 for m in args.rows):
        parser.error("samples 3..100; rows 1..4096")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "baseline_dispatch_commit": "b2fcc2fa8664518a56fee60341cd5690a22a9d8d",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": "3 warmups; randomized pairs; GPU command timing; uncontrolled thermals",
        "results": [],
    }
    rng = np.random.default_rng(753)
    try:
        with MetalRuntime() as rt:
            routes = {
                "previous": lambda *a, **kw: previous(rt, *a, **kw),
                "selected": rt._matmul_f32,
            }
            for m in args.rows:
                for n, k in ((384, 384), (1536, 384), (384, 1536)):
                    x = rng.normal(size=(m, k)).astype(np.float32)
                    w = rng.normal(size=(n, k)).astype(np.float32)
                    expected = x.astype(np.float64) @ w.astype(np.float64).T
                    buffers = [rt.buffer(a.nbytes, a) for a in (x, w)]
                    buffers.append(rt.buffer(m * n * 4))
                    samples = {name: [] for name in routes}
                    gpu_samples = {name: [] for name in routes}
                    errors = {}
                    difference = 0.0
                    try:
                        for iteration in range(args.samples + 3):
                            outputs = {}
                            for name in rng.permutation(list(routes)):
                                before = rt.diagnostics()
                                start = time.perf_counter()
                                with rt.command():
                                    routes[name](buffers, rows=m, cols=n, k=k)
                                elapsed = time.perf_counter() - start
                                after = rt.diagnostics()
                                outputs[name] = rt.read(buffers[-1], (m, n))
                                np.testing.assert_allclose(
                                    outputs[name], expected, atol=5e-5, rtol=5e-5
                                )
                                errors[name] = float(np.max(np.abs(outputs[name] - expected)))
                                if iteration >= 3:
                                    samples[name].append(elapsed)
                                    assert (
                                        after["gpu_timed_commands"]
                                        == before["gpu_timed_commands"] + 1
                                    )
                                    gpu_samples[name].append(
                                        after["gpu_seconds"] - before["gpu_seconds"]
                                    )
                            difference = max(
                                difference,
                                float(np.max(np.abs(outputs["selected"] - outputs["previous"]))),
                            )
                        row = {
                            "shape_m_n_k": [m, n, k],
                            "max_abs_error_vs_f64": errors,
                            "max_abs_difference_vs_previous": difference,
                            "wall_timings": {k: summarize(v, 1) for k, v in samples.items()},
                            "gpu_timings": {k: summarize(v, 1) for k, v in gpu_samples.items()},
                            "gpu_speedup": float(
                                np.median(gpu_samples["previous"])
                                / np.median(gpu_samples["selected"])
                            ),
                        }
                        payload["results"].append(row)
                        save_json(args.output, payload)
                        print({"shape": [m, n, k], "gpu_speedup": row["gpu_speedup"]}, flush=True)
                    finally:
                        for buffer in buffers:
                            buffer.close()
            payload["runtime"] = rt.diagnostics()
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
