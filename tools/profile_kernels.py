"""Intrusive per-dispatch GPU stage timestamps with separate normal-path controls."""

import argparse
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes

from tessery import EmbeddingModel, ModelProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--lengths", nargs="+", type=int, default=[7, 8, 9, 31, 32, 33, 128, 512])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 30 or not all(3 <= n <= 512 for n in args.lengths):
        parser.error("Invalid sample count or lengths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "mode": "intrusive_stage_timestamps",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [],
        "limitations": (
            "One encoder per dispatch; calibration and encoder costs perturb timing. "
            "Normal-path timings are separate. Counter storage excluded from "
            "runtime buffer counters."
        ),
    }
    try:
        profile = (
            ModelProfile.from_file(args.profile_file)
            if args.profile_file
            else "qwen3-embedding-0.6b-dwq"
        )
        with EmbeddingModel.load(args.model_dir, profile=profile) as model:
            rt = model._backend.runtime
            for n in args.lengths:
                special = 2 if model.descriptor.architecture == "bert_f32" else 1
                ids, lengths = model._tokenizer.batch(
                    [" token" * (n - special)], max_length=model.max_length
                )
                assert ids.shape == (1, n)
                forward = model._backend.forward
                for _ in range(2):
                    reference = forward(ids, lengths, dimensions=model.dimensions)
                normal, profiled, summaries = [], [], []
                for _ in range(args.samples):
                    start = time.perf_counter()
                    output = forward(ids, lengths, dimensions=model.dimensions)
                    normal.append(time.perf_counter() - start)
                    np.testing.assert_array_equal(output, reference)
                    with rt.profile_kernels() as records:
                        start = time.perf_counter()
                        output = forward(ids, lengths, dimensions=model.dimensions)
                        profiled.append(time.perf_counter() - start)
                    np.testing.assert_allclose(output, reference, atol=5e-6, rtol=1e-4)
                    sums = defaultdict(lambda: {"calls": 0, "gpu_seconds": 0.0})
                    for row in records:
                        assert np.isfinite(row["gpu_seconds"]) and row["gpu_seconds"] >= 0
                        sums[row["kernel"]]["calls"] += 1
                        sums[row["kernel"]]["gpu_seconds"] += row["gpu_seconds"]
                    assert 0 < sum(r["gpu_seconds"] for r in records) <= profiled[-1] * 1.1
                    summaries.append(dict(sums))
                ranking = sorted(
                    [
                        {
                            "kernel": name,
                            "calls": summaries[0][name]["calls"],
                            "median_gpu_seconds": float(
                                np.median([s[name]["gpu_seconds"] for s in summaries])
                            ),
                        }
                        for name in summaries[0]
                    ],
                    key=lambda x: -x["median_gpu_seconds"],
                )
                row = {
                    "tokens": n,
                    "normal_wall_seconds": normal,
                    "profiled_wall_seconds": profiled,
                    "samples": summaries,
                    "ranking": ranking,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print({"tokens": n, "top": ranking[:4]}, flush=True)
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
