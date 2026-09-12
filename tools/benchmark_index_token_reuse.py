"""Measure index token reuse with real local tokenizers and a CPU-only fake encoder."""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_inference.api import EmbeddingModel, ModelDescriptor  # noqa: E402
from metal_inference.index import DocumentIndex  # noqa: E402
from metal_inference.profiles import QWEN3_PROFILE, ModelProfile  # noqa: E402
from metal_inference.tokenizer import QwenTokenizer  # noqa: E402
from metal_inference.weights import read_json  # noqa: E402
from metal_inference.wordpiece import WordPieceTokenizer  # noqa: E402


class CountingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id
        self.text_count = 0

    def batch(self, texts, *, max_length, canceled=None):
        self.text_count += len(texts)
        return self.tokenizer.batch(texts, max_length=max_length, canceled=canceled)


class CPUBackend:
    max_padded_tokens = 4096
    runtime = SimpleNamespace(active_bytes=0, peak_bytes=0)

    def forward(self, ids, lengths, *, dimensions):
        self.token_digest.update(ids.tobytes())
        self.token_digest.update(lengths.tobytes())
        output = np.zeros((len(ids), dimensions), np.float32)
        for i, length in enumerate(lengths):
            output[i, int(np.sum(ids[i, :length])) % dimensions] = 1
        return output

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2:
        parser.error("samples must be an even number >=4")
    cases = [(256, 512), (600, 512), (600, 128)]
    if args.reverse:
        cases.reverse()
    documents = {
        "english": ("Paris is in France. Metal computes embeddings for local retrieval. " * 60),
        "russian": ("Локальный поиск документов. Векторные представления и память проекта. " * 60),
        "unicode": ("Semantic search 世界 café Привет 🙂 retrieves useful passages. " * 60),
    }
    bge_profile = ModelProfile.from_file(
        os.getenv(
            "METAL_INFERENCE_BGE_PROFILE_FILE",
            ROOT / "model-manifests/bge-small-en-v1.5-hf-cache.json",
        )
    )
    configurations = [
        (
            "qwen",
            QWEN3_PROFILE,
            os.getenv(
                "METAL_INFERENCE_MODEL_DIR",
                Path.home() / ".mlx-serve/models/Qwen3-Embedding-0.6B-4bit-DWQ",
            ),
        ),
        (
            "bert",
            bge_profile,
            os.getenv(
                "METAL_INFERENCE_BGE_MODEL_DIR",
                Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs",
            ),
        ),
    ]
    if args.reverse:
        configurations.reverse()
    results = []
    for kind, profile, directory in configurations:
        data = read_json(str(directory), "tokenizer.json", profile=profile)
        tokenizer = CountingTokenizer(
            QwenTokenizer(data) if kind == "qwen" else WordPieceTokenizer(data)
        )
        for chunk_chars, max_length in cases:
            with EmbeddingModel(
                CPUBackend(), tokenizer, 384, max_length, 2, ModelDescriptor.from_profile(profile)
            ) as model:

                def build(reuse, chunk_chars=chunk_chars):
                    model._reuse_index_tokens = reuse
                    model._backend.token_digest = hashlib.sha256()
                    return DocumentIndex.build(
                        model,
                        documents,
                        chunk_chars=chunk_chars,
                        overlap_chars=32,
                        document_prefix="document: ",
                        query_prefix="query: ",
                    )

                original = build(False)
                expected_tokens = model._backend.token_digest.hexdigest()
                actual = build(True)
                assert model._backend.token_digest.hexdigest() == expected_tokens
                assert original.chunks == actual.chunks
                np.testing.assert_array_equal(original._vectors, actual._vectors)
                timings = {False: [], True: []}
                counts = {False: [], True: []}
                savings = []
                for i in range(args.samples):
                    order = [False, True] if (i + args.reverse) % 2 == 0 else [True, False]
                    elapsed = {}
                    for reuse in order:
                        before = tokenizer.text_count
                        start = time.perf_counter_ns()
                        result = build(reuse)
                        elapsed[reuse] = (time.perf_counter_ns() - start) / 1e6
                        assert model._backend.token_digest.hexdigest() == expected_tokens
                        assert result.chunks == original.chunks
                        np.testing.assert_array_equal(result._vectors, original._vectors)
                        timings[reuse].append(elapsed[reuse])
                        counts[reuse].append(tokenizer.text_count - before)
                    savings.append(elapsed[False] - elapsed[True])
                assert all(
                    a - b == len(original.chunks)
                    for a, b in zip(counts[False], counts[True], strict=True)
                )
                peaks = {}
                for reuse in [False, True]:
                    tracemalloc.start()
                    build(reuse)
                    peaks[reuse] = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                blocks = np.asarray(savings).reshape(-1, 2).mean(axis=1)
                bootstrap = (
                    np.random.default_rng(11)
                    .choice(blocks, (2000, len(blocks)), replace=True)
                    .mean(axis=1)
                )
                row = {
                    "tokenizer": kind,
                    "chunk_chars": chunk_chars,
                    "max_length": max_length,
                    "chunks": len(original.chunks),
                    "exact_chunks_vectors": True,
                    "backend_token_sha256": expected_tokens,
                    "timings_ms": {
                        str(k): {
                            "p50": float(np.median(v)),
                            "p95": float(np.percentile(v, 95)),
                            "samples": v,
                        }
                        for k, v in timings.items()
                    },
                    "old_over_reuse": float(np.median(timings[False]) / np.median(timings[True])),
                    "tokenized_texts": {str(k): v[0] for k, v in counts.items()},
                    "peak_traced_bytes": {str(k): v for k, v in peaks.items()},
                    "paired_saving_ms_ci95": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                }
                results.append(row)
                print(
                    kind,
                    chunk_chars,
                    max_length,
                    len(original.chunks),
                    row["old_over_reuse"],
                    flush=True,
                )
    payload = {
        "scope": (
            "Index build with actual local tokenizers and deterministic CPU-only fake backend; "
            "no model inference"
        ),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "reverse": args.reverse,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "src/metal_inference/api.py",
                ROOT / "src/metal_inference/index.py",
                Path(__file__),
            ]
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
