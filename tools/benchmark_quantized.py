"""Paired uint4 kernel and full Qwen diagnostics using existing local weights."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from metal_inference import EmbeddingModel, MetalRuntime


def bf16(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def kernel_samples():
    rng = np.random.default_rng(914)
    results = []
    with MetalRuntime() as rt:
        for m, n, k in [(m, 1024, k) for m in (1, 7, 8, 9, 31, 32) for k in (1024, 3072)]:
            x = rng.normal(size=(m, k)).astype(np.float32)
            w = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32)
            s = bf16(rng.uniform(0.01, 0.2, size=(n, k // 64)))
            b = bf16(rng.uniform(-1, 0.1, size=s.shape))
            codes = ((w[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(n, k)
            scales = (s.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
            biases = (b.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
            weights = codes.astype(np.float32) * scales + biases
            reference = x.astype(np.float64) @ weights.astype(np.float64).T
            bufs = [rt.buffer(a.nbytes, a) for a in (x, w, s, b)] + [rt.buffer(m * n * 4)]
            samples = {
                name: [] for name in ("linear4", "linear4_tiled" if m % 8 == 0 else "linear4_tail")
            }
            errors = {}
            for iteration in range(22):
                for name in rng.permutation(list(samples)):
                    tiled = name != "linear4"
                    before = rt.diagnostics()
                    with rt.command():
                        rt._dispatch(
                            str(name),
                            bufs,
                            threads=((m + 7) // 8) * (n // 32) * 128
                            if tiled
                            else ((m + 3) // 4) * n * 32,
                            group_size=128 if tiled else 32,
                            rows=m,
                            cols=n,
                            k=k,
                        )
                    after = rt.diagnostics()
                    if (
                        iteration >= 2
                        and after["gpu_timed_commands"] > before["gpu_timed_commands"]
                    ):
                        samples[name].append(after["gpu_seconds"] - before["gpu_seconds"])
                    out = rt.read(bufs[-1], (m, n))
                    np.testing.assert_allclose(out, reference, atol=5e-5, rtol=5e-5)
                    errors[name] = float(np.max(np.abs(out - reference)))
            results.append(
                {"shape_m_n_k": [m, n, k], "gpu_seconds": samples, "max_abs_error_vs_f64": errors}
            )
            for buf in bufs:
                buf.close()
        return {"results": results, "runtime": rt.diagnostics()}


def model_samples(model_dir):
    rng = np.random.default_rng(14)
    results = []
    with EmbeddingModel.load(model_dir) as model:
        rt = model._backend.runtime
        rt._plans_enabled = False  # These experiments mutate kernel routing between calls.
        selected = rt._linear4

        def baseline(buffers, *, rows, cols, k):
            rt._dispatch(
                "linear4",
                buffers,
                threads=((rows + 3) // 4) * cols * 32,
                group_size=32,
                rows=rows,
                cols=cols,
                k=k,
            )

        for batch, tokens in [(1, 2), (1, 7), (1, 8), (1, 9), (1, 31), (1, 32), (4, 33)]:
            texts = [" token" * (tokens - 1)] * batch
            ids, _ = model._tokenizer.batch(texts, max_length=model.max_length)
            assert ids.shape == (batch, tokens)
            samples = {name: [] for name in ("baseline", "tiled")}
            outputs = {}
            for iteration in range(7):
                for name in rng.permutation(list(samples)):
                    rt._linear4 = baseline if name == "baseline" else selected
                    started = time.perf_counter()
                    outputs[name] = model.encode(texts)
                    if iteration >= 2:
                        samples[name].append(time.perf_counter() - started)
                np.testing.assert_allclose(
                    outputs["tiled"], outputs["baseline"], atol=5e-6, rtol=1e-4
                )
            results.append(
                {
                    "batch": batch,
                    "tokens": tokens,
                    "wall_seconds": samples,
                    "max_abs_vector_difference": float(
                        np.max(np.abs(outputs["tiled"] - outputs["baseline"]))
                    ),
                    "speedup": float(np.median(samples["baseline"]) / np.median(samples["tiled"])),
                }
            )
            print(results[-1], flush=True)
        rt._linear4 = selected
        return {
            "results": results,
            "runtime": rt.diagnostics(),
            "compatibility_id": model.descriptor.compatibility_id,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Reserve output before expensive work; incomplete/failed runs are not successes.
    with args.output.open("x") as stream:
        payload = {
            "scope": "single_host_paired_diagnostic",
            "platform": platform.platform(),
            "conditions": "randomized order; uncontrolled thermal/power/background",
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "kernel": kernel_samples(),
            "model": model_samples(args.model_dir),
        }
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
