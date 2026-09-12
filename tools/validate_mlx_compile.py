"""Check compiled MLX against its uncompiled graph with changing values and masks.

Uses only existing model files. Run sequentially with other GPU workloads.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json

from metal_inference.profiles import QWEN3_PROFILE, ModelProfile
from metal_inference.tokenizer import QwenTokenizer
from metal_inference.weights import read_json
from metal_inference.wordpiece import WordPieceTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = ModelProfile.from_file(args.profile_file) if args.profile_file else QWEN3_PROFILE
    data = read_json(args.model_dir, "tokenizer.json", profile=profile)
    if profile.architecture == "bert_f32":
        from benchmark_mlx_bert import MLXBertReference

        backend = MLXBertReference(args.model_dir, profile, compiled=True)
        tokenizer = WordPieceTokenizer(data)
    else:
        from benchmark_mlx import MLXReference

        backend = MLXReference(args.model_dir, causal_fast_path=True, compiled=True)
        tokenizer = QwenTokenizer(data)
    payload = {"status": "running", "profile_sha256": profile.identity_sha256, "checks": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    try:
        # Each pair has equal static shapes but different IDs, pooling lengths and masks.
        for width, texts in [
            (16, ["Hello world", "A somewhat longer example for retrieval."]),
            (16, ["Short changed words", "Another document."]),
            (24, ["A third changing example", "A final document about cats and dogs"]),
            (16, ["Hello world", "A somewhat longer example for retrieval."]),
        ]:
            ids, lengths = tokenizer.batch(texts, max_length=width)
            padded = np.full((len(texts), width), tokenizer.pad_id, np.uint32)
            padded[:, : ids.shape[1]] = ids
            backend.runner.compiled = False
            expected = backend.forward(padded, lengths, dimensions=profile.default_dimensions)
            backend.runner.compiled = True
            before = backend.runner.traces
            started = time.perf_counter()
            actual = backend.forward(padded, lengths, dimensions=profile.default_dimensions)
            elapsed = time.perf_counter() - started
            np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-4)
            repeat = backend.forward(padded, lengths, dimensions=profile.default_dimensions)
            np.testing.assert_array_equal(actual, repeat)
            payload["checks"].append(
                {
                    "shape": list(padded.shape),
                    "lengths": lengths.tolist(),
                    "ids_sha256": hashlib.sha256(padded.tobytes()).hexdigest(),
                    "max_abs_difference": float(np.max(np.abs(actual - expected))),
                    "traces_added": backend.runner.traces - before,
                    "call_seconds": elapsed,
                }
            )
        assert [row["traces_added"] for row in payload["checks"]] == [1, 0, 1, 0]
        payload["compilation"] = backend.runner.diagnostics()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        backend.close()
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
