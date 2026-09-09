"""Intrusive per-dispatch GPU stage timestamps with separate normal-path controls."""

import argparse
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel, ModelProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--lengths", nargs="+", type=int, default=[7, 8, 9, 31, 32, 33, 128, 512])
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--execution-policy", choices=["selected", "raw"], default="selected")
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
        "execution_policy": args.execution_policy,
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
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            for n in args.lengths:
                special = 2 if model.descriptor.architecture == "bert_f32" else 1
                ids, lengths = model._tokenizer.batch(
                    [" token" * (n - special)], max_length=model.max_length
                )
                assert ids.shape == (1, n)
                public_reference = model.encode([" token" * (n - special)])
                width = n
                if args.execution_policy == "selected":
                    plan = list(
                        execution_batches(
                            lengths,
                            model._backend.max_padded_tokens,
                            model.max_length,
                            model.descriptor.architecture,
                        )
                    )
                    assert len(plan) == 1 and plan[0][0].tolist() == [0]
                    width = plan[0][1]
                    padded = np.full((1, width), model._tokenizer.pad_id, np.uint32)
                    padded[:, :n] = ids
                    ids = padded
                forward = model._backend.forward
                for _ in range(2):
                    reference = forward(ids, lengths, dimensions=model.dimensions)
                np.testing.assert_allclose(reference, public_reference, atol=5e-6, rtol=1e-4)
                normal, profiled, summaries, shape_summaries = [], [], [], []
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
                    shapes = defaultdict(lambda: {"calls": 0, "gpu_seconds": 0.0})
                    for row in records:
                        assert np.isfinite(row["gpu_seconds"]) and row["gpu_seconds"] >= 0
                        sums[row["kernel"]]["calls"] += 1
                        sums[row["kernel"]]["gpu_seconds"] += row["gpu_seconds"]
                        key = (row["kernel"], row["rows"], row["cols"], row["k"])
                        shapes[key]["calls"] += 1
                        shapes[key]["gpu_seconds"] += row["gpu_seconds"]
                    assert 0 < sum(r["gpu_seconds"] for r in records) <= profiled[-1] * 1.1
                    summaries.append(dict(sums))
                    shape_summaries.append(dict(shapes))
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
                    "execution_width": width,
                    "max_abs_difference_vs_public_api": float(
                        np.max(np.abs(reference - public_reference))
                    ),
                    "shape_ranking": sorted(
                        [
                            {
                                "kernel": key[0],
                                "rows": key[1],
                                "cols": key[2],
                                "k": key[3],
                                "calls": shape_summaries[0][key]["calls"],
                                "median_gpu_seconds": float(
                                    np.median(
                                        [sample[key]["gpu_seconds"] for sample in shape_summaries]
                                    )
                                ),
                            }
                            for key in shape_summaries[0]
                        ],
                        key=lambda row: -row["median_gpu_seconds"],
                    ),
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
