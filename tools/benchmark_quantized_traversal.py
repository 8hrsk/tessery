"""Offline experiment: reorder Qwen gate/up threadgroups without changing arithmetic.

Candidate shaders are generated in memory; production sources are never edited.
Resident layer-zero weights, F64 checks, exact baseline equality, and duplicate
baseline timing controls qualify this experiment, not a public runtime selector.
"""

import argparse
import hashlib
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from benchmark_mlp_isolated import dense_weights, digest, load_projections
from diagnose_metal import ROOT, save_json, source_hashes, summarize

from metal_inference.profiles import QWEN3_PROFILE
from tessery import MetalRuntime

BASE = "linear4_16x32_k64"
SHADER = ROOT / "src/metal_inference/native/kernels.metal"
MAPPING = """    uint row = (tile / (p.cols/32))*16 + (sg/4)*8;
    uint channel = (tile % (p.cols/32))*32;"""


def experimental_source():
    source = SHADER.read_text()
    start = source.index(f"kernel void {BASE}(")
    end = source.index("\n}\n", start) + 3
    original = source[start:end]
    assert original.count(MAPPING) == 1
    for width in (2, 4):
        mapping = f"""    uint first = (tile / ({width}*(p.cols/32)))*{width};
    uint height = min({width}u, p.rows/16-first);
    uint within = tile % ({width}*(p.cols/32));
    uint row = (first+within%height)*16 + (sg/4)*8;
    uint channel = (within/height)*32;"""
        source += "\n" + original.replace(BASE, f"{BASE}_group{width}", 1).replace(MAPPING, mapping)
    return source.encode()


def experimental_runtime():
    source = experimental_source()
    read_bytes = Path.read_bytes

    def read(path):
        return source if path == SHADER else read_bytes(path)

    # Only construction is patched, before any commands/model worker threads.
    with patch.object(Path, "read_bytes", read):
        rt = MetalRuntime()
    assert rt.diagnostics()["shader_sha256"] == hashlib.sha256(source).hexdigest()
    return rt


def install_route(rt):
    dispatch = rt._dispatch
    selection = {"kernel": BASE}

    def routed(name, buffers, **kw):
        if name == BASE and (kw["cols"], kw["k"]) == (3072, 1024):
            name = selection["kernel"]
        return dispatch(name, buffers, **kw)

    rt._dispatch = routed
    return selection


def measure(rt, selection, arrays, x, args, rng):
    m, k = x.shape
    n = arrays[0].shape[0]
    names = {"baseline_a": BASE, "baseline_b": BASE}
    names.update({f"group{g}": f"{BASE}_group{g}" for g in (2, 4)})
    buffers = [rt.buffer(a.nbytes, a) for a in [x, *arrays]]
    sentinel = np.full((m + 2, n), np.nan, dtype=np.float32)
    buffers.append(rt.buffer(sentinel.nbytes, sentinel))
    gpu = {name: [] for name in names}
    wall = {name: [] for name in names}
    warmups = {}

    def invoke(repeats):
        with rt.command():
            for _ in range(repeats):
                rt._linear4(buffers, rows=m, cols=n, k=k)

    try:
        if not args.validate_only:
            for label, kernel in names.items():
                selection["kernel"] = kernel
                started = time.perf_counter()
                count = 0
                while count < 20 or time.perf_counter() - started < 0.1:
                    invoke(1)
                    count += 1
                warmups[label] = count
            for _ in range(args.samples):
                for label in rng.permutation(list(names)):
                    selection["kernel"] = names[label]
                    before = rt.diagnostics()
                    started = time.perf_counter()
                    invoke(args.repeats)
                    elapsed = (time.perf_counter() - started) / args.repeats
                    after = rt.diagnostics()
                    assert after["gpu_timed_commands"] == before["gpu_timed_commands"] + 1
                    gpu[label].append((after["gpu_seconds"] - before["gpu_seconds"]) / args.repeats)
                    wall[label].append(elapsed)
        reference = x.astype(np.float64) @ dense_weights(arrays).astype(np.float64).T
        errors = {}
        baseline = None
        for label, kernel in names.items():
            # Reset the entire output and guard on every validation dispatch:
            # stale baseline values cannot hide a missing candidate write.
            buffers[-1].close()
            buffers[-1] = rt.buffer(sentinel.nbytes, sentinel)
            selection["kernel"] = kernel
            invoke(1)
            output = rt.read(buffers[-1], sentinel.shape)
            assert np.isnan(output[m:]).all()
            np.testing.assert_allclose(output[:m], reference, atol=5e-5, rtol=5e-5)
            if baseline is None:
                baseline = output[:m].copy()
            np.testing.assert_array_equal(output[:m], baseline)
            errors[label] = float(np.max(np.abs(output[:m] - reference)))
        row = {
            "shape_m_n_k": [m, n, k],
            "input_sha256": digest(x),
            "weight_hashes": [digest(a) for a in arrays],
            "max_abs_error_vs_f64": errors,
            "exact_baseline_equality": True,
            "output_guard_untouched": True,
            "warmups": warmups,
        }
        if not args.validate_only:
            medians = {label: float(np.median(values)) for label, values in gpu.items()}
            assert all(np.isfinite(v) and v > 0 for v in medians.values())
            control = medians["baseline_a"] / medians["baseline_b"]
            base = float(np.median(gpu["baseline_a"] + gpu["baseline_b"]))
            row.update(
                gpu_timings={k: summarize(v, 1) for k, v in gpu.items()},
                wall_timings={k: summarize(v, 1) for k, v in wall.items()},
                gpu_speedup={label: base / medians[label] for label in names},
                identical_control_a_over_b=control,
                control_within_10_percent=0.9 <= control <= 1.1,
            )
        return row
    finally:
        for buffer in buffers:
            buffer.close()
        assert rt.active_bytes == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 129, 264, 512])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not 3 <= args.samples <= 100
        or not 1 <= args.repeats <= 100
        or any(not 1 <= m <= 4096 for m in args.rows)
        or len(set(args.rows)) != len(args.rows)
    ):
        parser.error("samples 3..100; repeats 1..100; unique rows 1..4096")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": digest(
            np.frombuffer(
                Path(__file__).with_name("benchmark_mlp_isolated.py").read_bytes(), dtype=np.uint8
            )
        ),
        "experimental_shader_sha256": hashlib.sha256(experimental_source()).hexdigest(),
        "profile_sha256": QWEN3_PROFILE.identity_sha256,
        "cpu_reference_thread_limits": {
            key: os.getenv(key) for key in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "shader_validation": os.getenv("MTL_SHADER_VALIDATION"),
        "samples": args.samples,
        "dispatch_repeats_per_command": args.repeats,
        "validate_only": args.validate_only,
        "reverse_cases": args.reverse_cases,
        "conditions": (
            "20 calls and 100ms minimum warmup per route; random paired order; "
            "baseline A/B are identical; GPU command timestamps per invocation; "
            "host validation outside timing; uncontrolled thermals; layer-zero weights"
        ),
        "results": [],
    }
    try:
        projections = load_projections(args.model_dir, QWEN3_PROFILE)
        cases = [
            (name, m) for name in projections if not name.endswith("down_proj") for m in args.rows
        ]
        order = np.random.default_rng(1601).permutation(len(cases))
        if args.reverse_cases:
            order = order[::-1]
        with experimental_runtime() as rt:
            selection = install_route(rt)
            for index in order:
                name, m = cases[index]
                x = (
                    np.random.default_rng(1600 + int(index))
                    .normal(size=(m, 1024))
                    .astype(np.float32)
                )
                row = measure(
                    rt,
                    selection,
                    projections[name],
                    x,
                    args,
                    np.random.default_rng(1800 + int(index)),
                )
                row["projection"] = name
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {
                        k: v
                        for k, v in row.items()
                        if k
                        in (
                            "projection",
                            "shape_m_n_k",
                            "gpu_speedup",
                            "identical_control_a_over_b",
                        )
                    },
                    flush=True,
                )
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
