"""Compare exact CPU retrieval with the old full-scan normalization/sort baseline.

No model is loaded: a deterministic fake encoder returns one fixed query vector.
The public DocumentIndex.search timings include its usual contract and input checks,
but exclude real tokenization and model inference. Run with BLAS thread counts = 1.
"""

import argparse
import hashlib
import itertools
import json
import platform
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metal_inference import ModelDescriptor  # noqa: E402
from metal_inference.index import Chunk, DocumentIndex, RetrievalHit, _contract  # noqa: E402
from metal_inference.retrieval import SearchHit, cosine_search  # noqa: E402


def baseline_search(index, model, query, k):
    """Original search formula/sort on the benchmark's finite ordinary embeddings."""
    if _contract(model) != index._metadata["contract"]:
        raise AssertionError("contract mismatch")
    if not isinstance(query, str) or not query.strip() or type(k) is not int or not 1 <= k <= 100:
        raise AssertionError("invalid query")
    vector = model.encode([index._metadata["query_prefix"] + query])[0]
    documents = index._vectors
    if (
        vector.ndim != 1
        or documents.ndim != 2
        or documents.shape[1] != vector.shape[0]
        or type(k) is not int
        or k < 1
        or not np.isfinite(vector).all()
        or not np.isfinite(documents).all()
    ):
        raise AssertionError("benchmark expects finite vectors")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        qnorm = np.linalg.norm(vector)
        norms = np.linalg.norm(documents, axis=1)
        scores = (documents @ vector) / (norms * qnorm)
    floor = np.sqrt(np.finfo(np.float32).tiny)
    if (
        not np.isfinite(qnorm)
        or not np.isfinite(norms).all()
        or qnorm < floor
        or np.any(norms < floor)
        or not np.isfinite(scores).all()
    ):
        raise AssertionError("benchmark excludes extreme finite fallback")
    order = np.argsort(-scores, kind="stable")[:k]
    hits = [SearchHit(int(i), float(scores[i])) for i in order]
    return [RetrievalHit(index.chunks[h.index], h.score) for h in hits]


def generic_search(index, model, query, k):
    if _contract(model) != index._metadata["contract"]:
        raise AssertionError("contract mismatch")
    if not isinstance(query, str) or not query.strip() or type(k) is not int or not 1 <= k <= 100:
        raise AssertionError("invalid query")
    vector = model.encode([query])[0]
    return [
        RetrievalHit(index.chunks[h.index], h.score)
        for h in cosine_search(vector, index._vectors, k=k)
    ]


def peak_bytes(call):
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak - before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.samples < 6 or args.samples % 6:
        parser.error("samples must be a positive multiple of six")
    cases = list(itertools.product([10, 100, 1000, 10000], [64, 384, 1024], [1, 5, 100]))
    if args.reverse_cases:
        cases.reverse()
    output = []
    for count, dimensions, k in cases:
        rng = np.random.default_rng(1301 + count + dimensions)
        docs = rng.standard_normal((count, dimensions), dtype=np.float32)
        docs /= np.linalg.norm(docs, axis=1)[:, None]
        query = rng.standard_normal(dimensions, dtype=np.float32)
        query /= np.linalg.norm(query)
        # Include a tie plateau crossing the k=5 boundary.
        docs[: min(12, count)] = query
        model = SimpleNamespace(
            descriptor=ModelDescriptor(),
            dimensions=dimensions,
            max_length=512,
            encode=lambda texts, vector=query: vector[None, :],
        )
        chunks = tuple(Chunk(str(i), 0, 1, "x") for i in range(count))
        metadata = {"contract": _contract(model), "query_prefix": ""}
        start = time.perf_counter_ns()
        index = DocumentIndex(metadata, chunks, docs)
        build_ms = (time.perf_counter_ns() - start) / 1e6
        funcs = {
            "baseline": partial(baseline_search, index, model, "query", k),
            "generic_partial": partial(generic_search, index, model, "query", k),
            "cached_index": partial(index.search, model, "query", k=k),
        }
        expected = funcs["baseline"]()
        for func in funcs.values():
            assert func() == expected
            for _ in range(3):
                func()
        timings = {name: [] for name in funcs}
        paired_savings = []
        orders = list(itertools.permutations(funcs))
        for sample in range(args.samples):
            block = {}
            for name in orders[sample % 6]:
                start = time.perf_counter_ns()
                result = funcs[name]()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                assert result == expected
                timings[name].append(elapsed)
                block[name] = elapsed
            paired_savings.append(block["baseline"] - block["cached_index"])
        boot = (
            np.random.default_rng(19)
            .choice(paired_savings, size=(2000, len(paired_savings)), replace=True)
            .mean(axis=1)
        )
        stats = {
            name: {
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "peak_traced_bytes": peak_bytes(funcs[name]),
                "samples_ms": times,
            }
            for name, times in timings.items()
        }
        row = {
            "count": count,
            "dimensions": dimensions,
            "k": k,
            "snapshot_build_ms": build_ms,
            "norm_cache_bytes": index._document_norms.nbytes,
            "vector_bytes": index._vectors.nbytes,
            "exact_scores_and_order": True,
            "baseline_over_cached": stats["baseline"]["p50_ms"] / stats["cached_index"]["p50_ms"],
            "paired_mean_saving_ms_ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
            "timings": stats,
        }
        output.append(row)
    payload = {
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scope": "CPU retrieval only; fake query encoder; no tokenization/model inference",
        "samples_per_case": args.samples,
        "reverse_cases": args.reverse_cases,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "src/metal_inference/retrieval.py",
                ROOT / "src/metal_inference/index.py",
                Path(__file__),
            ]
        },
        "cases": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in output:
        print(row["count"], row["dimensions"], row["k"], round(row["baseline_over_cached"], 3))


if __name__ == "__main__":
    main()
