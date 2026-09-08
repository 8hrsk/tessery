"""Paired comparison of historical global padding and length buckets.

Both plans call the same final backend with the same tokenized input. Timings
cover plan construction, forward and output restoration, not tokenization or
public API admission. This isolates padding from simultaneous kernel changes.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from metal_inference import EmbeddingModel, ModelProfile, get_profile
from metal_inference.batching import length_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", default="qwen3-embedding-0.6b-dwq")
    group.add_argument("--profile-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--long-tokens", type=int, default=512)
    args = parser.parse_args()
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile = (
        ModelProfile.from_file(args.profile_file)
        if args.profile_file
        else get_profile(args.profile)
    )
    if not (2 <= args.batch_size <= 32 and 3 <= args.long_tokens <= profile.max_length):
        parser.error("batch or sequence length outside supported bounds")
    with EmbeddingModel.load(args.model_dir, profile=profile) as model:
        texts = [" token" * (args.long_tokens - profile.min_length)] + ["short text"] * (
            args.batch_size - 1
        )
        ids, lengths = model._tokenizer.batch(texts, max_length=512)
        budget = model._backend.max_padded_tokens
        step = max(1, budget // ids.shape[1])

        def plan(name):
            if name == "buckets":
                return list(length_batches(lengths, budget))
            return [
                (np.arange(i, min(i + step, len(texts))), ids.shape[1])
                for i in range(0, len(texts), step)
            ]

        def forward(name):
            output = np.empty((len(texts), model.dimensions), np.float32)
            for rows, width in plan(name):
                output[rows] = model._backend.forward(
                    np.ascontiguousarray(ids[rows, :width]),
                    lengths[rows],
                    dimensions=model.dimensions,
                )
            return output

        reference = forward("global_padding")
        actual = forward("buckets")
        np.testing.assert_allclose(actual, reference, atol=5e-6, rtol=1e-4)
        np.testing.assert_array_equal(model.encode(texts), actual)
        samples = {key: [] for key in ("global_padding", "buckets")}
        rng = np.random.default_rng(37)
        for _ in range(3):
            for name in rng.permutation(list(samples)):
                start = time.perf_counter()
                vectors = forward(name)
                samples[name].append(time.perf_counter() - start)
                np.testing.assert_allclose(vectors, reference, atol=5e-6, rtol=1e-4)
                print(name, samples[name][-1], flush=True)
        output = {
            "harness_sha256": harness_sha256,
            "loaded_runtime": model._backend.runtime.diagnostics(),
            "scope": "single_host_paired_batch_plan_diagnostic",
            "model": profile.model_id,
            "compatibility_id": profile.compatibility_id,
            "lengths": lengths.tolist(),
            "conditions": "1 warmup + 3 paired samples; uncontrolled power/thermal/background",
            "latency_seconds": samples,
            "padded_positions": {key: sum(len(r) * w for r, w in plan(key)) for key in samples},
            "maximum_vector_difference": float(np.max(np.abs(actual - reference))),
            "bucketed_median_speedup": float(
                np.median(samples["global_padding"]) / np.median(samples["buckets"])
            ),
            "source_hashes": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (Path(__file__).resolve().parents[1] / "src/metal_inference").rglob("*")
                if path.suffix in {".py", ".mm", ".metal"}
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
