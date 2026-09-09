"""Compare full uint4 output tiles against an independent F64 reference."""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
from benchmark_mixed_quantized_model import previous
from diagnose_metal import save_json, source_hashes, summarize

from tessery import MetalRuntime


def bf16(array):
    return (array.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[8, 16, 17, 24, 31, 40, 129, 256, 257, 264, 272]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 3 <= args.samples <= 100
        or not 1 <= args.repeats <= 100
        or any(m < 1 or m > 4096 for m in args.rows)
    ):
        parser.error("samples 3..100; repeats 1..100; rows 1..4096")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "dispatch_repeats_per_command": args.repeats,
        "cpu_reference_thread_limits": {
            key: os.getenv(key) for key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "baseline_dispatch_commit": "7673c19e9f483b5bad6426f3e8258e958529b2af",
        "baseline_harness_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_mixed_quantized_model.py").read_bytes()
        ).hexdigest(),
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": (
            "3 warmups; paired randomized order; validation outside timing loop; "
            "repeated dispatches per command, timings per invocation; uncontrolled thermals"
        ),
        "results": [],
    }
    rng = np.random.default_rng(601)
    try:
        with MetalRuntime() as rt:
            for m in args.rows:
                for n, k in [(1024, 1024), (2048, 1024), (3072, 1024), (1024, 2048), (1024, 3072)]:
                    x = rng.normal(size=(m, k)).astype(np.float32)
                    w = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32)
                    s = bf16(rng.uniform(0.01, 0.2, size=(n, k // 64)))
                    b = bf16(rng.uniform(-1, 0.1, size=s.shape))
                    codes = ((w[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(
                        n, k
                    )
                    weights = codes.astype(np.float32) * (s.astype(np.uint32) << 16).view(
                        np.float32
                    ).repeat(64, axis=1)
                    weights += (b.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
                    expected = x.astype(np.float64) @ weights.astype(np.float64).T
                    buffers = [rt.buffer(a.nbytes, a) for a in (x, w, s, b)]
                    buffers.append(rt.buffer(m * n * 4))
                    names = ["previous", "selected"]
                    samples = {name: [] for name in names}
                    gpu_samples = {name: [] for name in names}
                    errors = {}
                    differences = {}
                    try:
                        for iteration in range(args.samples + 3):
                            for name in rng.permutation(names):
                                before = rt.diagnostics()
                                start = time.perf_counter()
                                with rt.command():
                                    for _ in range(args.repeats):
                                        if name == "previous":
                                            previous(rt, buffers, rows=m, cols=n, k=k)
                                        else:
                                            rt._linear4(buffers, rows=m, cols=n, k=k)
                                elapsed = (time.perf_counter() - start) / args.repeats
                                after = rt.diagnostics()
                                if iteration >= 3:
                                    samples[name].append(elapsed)
                                    assert (
                                        after["gpu_timed_commands"]
                                        == before["gpu_timed_commands"] + 1
                                    )
                                    gpu_samples[name].append(
                                        (after["gpu_seconds"] - before["gpu_seconds"])
                                        / args.repeats
                                    )
                        # Validate each route after timing. Repeated host-side F64
                        # comparisons can disturb GPU clock/thermal state between
                        # paired samples, including identical-dispatch controls.
                        outputs = {}
                        for name in names:
                            with rt.command():
                                if name == "previous":
                                    previous(rt, buffers, rows=m, cols=n, k=k)
                                else:
                                    rt._linear4(buffers, rows=m, cols=n, k=k)
                            outputs[name] = rt.read(buffers[-1], (m, n))
                            np.testing.assert_allclose(
                                outputs[name], expected, atol=5e-5, rtol=5e-5
                            )
                            errors[name] = float(np.max(np.abs(outputs[name] - expected)))
                        for name in names:
                            np.testing.assert_array_equal(outputs[name], outputs["previous"])
                            differences[name] = float(
                                np.max(np.abs(outputs[name] - outputs["previous"]))
                            )
                        row = {
                            "shape_m_n_k": [m, n, k],
                            "max_abs_error_vs_f64": errors,
                            "max_abs_difference_vs_previous": differences,
                            "wall_timings": {k: summarize(v, 1) for k, v in samples.items()},
                            "gpu_timings": {k: summarize(v, 1) for k, v in gpu_samples.items()},
                            "gpu_speedup": {
                                k: float(np.median(gpu_samples["previous"]) / np.median(v))
                                for k, v in gpu_samples.items()
                            },
                        }
                        payload["results"].append(row)
                        save_json(args.output, payload)
                        print(
                            {
                                "shape": [m, n, k],
                                "gpu_speedup": row["gpu_speedup"],
                                "difference": differences,
                            },
                            flush=True,
                        )
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
