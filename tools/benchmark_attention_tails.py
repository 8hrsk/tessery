"""Paired partial-tile attention kernel benchmark against an independent float64 reference."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize

from tessery import MetalRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default=15, type=int)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100:
        parser.error("samples must be 3..100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "baseline_dispatch_commit": "9b7bb1e5015c3d6731a4803bf050dde5ef6f4869",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": (
            "paired randomized order; 3 warmups; normal command timing; uncontrolled thermals"
        ),
        "results": [],
    }
    rng = np.random.default_rng(129)
    try:
        with MetalRuntime() as rt:
            for dim, heads, kv, bidirectional in [(32, 12, 12, True), (128, 16, 8, False)]:
                for seq in (64, 65, 71, 127, 128, 129, 136, 255, 511, 512):
                    q = rng.normal(size=(1, seq, heads, dim)).astype(np.float32)
                    k, v = [rng.normal(size=(1, seq, kv, dim)).astype(np.float32) for _ in range(2)]
                    lengths = np.array([seq], np.uint32)
                    expected = np.empty_like(q, dtype=np.float64)
                    for h in range(heads):
                        kh = h // (heads // kv)
                        scores = q[0, :, h].astype(np.float64) @ k[0, :, kh].astype(np.float64).T
                        scores *= dim**-0.5
                        if not bidirectional:
                            scores[np.triu_indices(seq, 1)] = -np.inf
                        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
                        probs /= probs.sum(axis=1, keepdims=True)
                        expected[0, :, h] = probs @ v[0, :, kh].astype(np.float64)
                    buffers = [rt.buffer(a.nbytes, a) for a in (q, k, v, lengths)]
                    buffers.append(rt.buffer(q.nbytes))
                    baseline = "attention_tiled" if seq % 32 == 0 else "attention"
                    selected = "attention_tiled" if seq % 32 == 0 else f"attention_tail_{dim}"
                    kernels = {"previous": baseline, "selected": selected}
                    samples = {name: [] for name in kernels}
                    gpu_samples = {name: [] for name in kernels}
                    errors = {}
                    try:
                        for iteration in range(args.samples + 3):
                            for name in rng.permutation(list(samples)):
                                kernel = kernels[name]
                                tiled = kernel != "attention"
                                group_size = 128 if tiled else 32
                                before = rt.diagnostics()
                                start = time.perf_counter()
                                with rt.command():
                                    rt._dispatch(
                                        kernel,
                                        buffers,
                                        threads=((seq + 7) // 8 if tiled else seq)
                                        * heads
                                        * group_size,
                                        group_size=group_size,
                                        seq=seq,
                                        heads=heads,
                                        kv_heads=kv,
                                        dim=dim,
                                        scale=dim**-0.5,
                                        bidirectional=bidirectional,
                                    )
                                wall = time.perf_counter() - start
                                after = rt.diagnostics()
                                result = rt.read(buffers[-1], q.shape)
                                np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)
                                errors[str(name)] = float(np.max(np.abs(result - expected)))
                                if iteration >= 3:
                                    samples[str(name)].append(wall)
                                    assert (
                                        after["gpu_timed_commands"]
                                        == before["gpu_timed_commands"] + 1
                                    )
                                    gpu_samples[str(name)].append(
                                        after["gpu_seconds"] - before["gpu_seconds"]
                                    )
                        row = {
                            "seq": seq,
                            "kernels": kernels,
                            "heads": heads,
                            "dim": dim,
                            "bidirectional": bidirectional,
                            "max_abs_errors": errors,
                            "timings": {k: summarize(v, 1) for k, v in samples.items()},
                            "gpu_timings": {k: summarize(v, 1) for k, v in gpu_samples.items()},
                            "gpu_speedup": float(
                                np.median(gpu_samples["previous"])
                                / np.median(gpu_samples["selected"])
                            ),
                            "speedup": float(
                                np.median(samples["previous"]) / np.median(samples["selected"])
                            ),
                        }
                        payload["results"].append(row)
                        save_json(args.output, payload)
                        print(
                            {k: v for k, v in row.items() if k not in {"timings", "gpu_timings"}},
                            flush=True,
                        )
                    finally:
                        for b in buffers:
                            b.close()
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
