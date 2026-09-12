# Exact CPU retrieval: norm cache and stable top-k

Measured on Apple M1 on 2026-09-12 using NumPy 2.5.2 and Python 3.12.13.
The experiment started from `368d00a` in its own `perf/rag-search` worktree.

`DocumentIndex` now computes row norms once when taking ownership of its private,
immutable float32 matrix. Each query uses the original `(documents @ query) /
(norms * query_norm)` formula, preserving score rounding. It does not assume that
vectors have unit norm. The cache costs four bytes per row: 40,000 bytes at the
current 10,000-chunk bound. Snapshot files and hashes do not change; loading
reconstructs the cache from the validated matrix.

Large narrow top-k selections partition scores, retain every strictly better row,
then take cutoff ties in original document order. Sorting just those selected
rows by score and original index preserves the stable full-sort contract.
Small matrices (<128 rows) and wide selections (k >= one quarter of rows) retain
full stable sorting. This selection also benefits standalone `cosine_search`.

The generic API still checks every document on each call. Both paths retain the
scaled float64 fallback for overflow/underflow, rejection of zero/non-finite
vectors, and stable original-index ties. Invalid direct-constructor matrices
continue through generic validation instead of silently trusting a norm cache.

## Measurements

The checked-in [summary](../benchmarks/retrieval/cached-search-20260912.json)
contains two fresh-process runs, the second reversing case order. Each covers
10/100/1,000/10,000 documents, 64/384/1,024 dimensions and k=1/5/100 (36 cases).
Each variant receives 60 samples, ordered in six balanced permutations. A tie
plateau crosses the k=5 boundary. Every measured result exactly matches the old
score formula and stable sorting, including Python float scores. The harness
records source hashes, p50/p95, transient allocation peaks and bootstrap intervals
of mean paired savings, resampling whole six-order blocks.

These timings call public `DocumentIndex.search` with a deterministic fake encoder.
They measure CPU retrieval, including its ordinary contract/input checks, but
**exclude real tokenization and model inference**. They are not total RAG latency
or an MLX comparison. Baseline performs the previous validation, norm calculation,
full stable sort and both hit-object conversion steps.

Representative k=5 results, ranges across the two processes:

| Documents × dimensions | Old p50 (ms) | Cached p50 (ms) | Old / cached |
|---|---:|---:|---:|
| 10 × 64 | 0.0201–0.0202 | 0.0174–0.0175 | 1.16× |
| 100 × 384 | 0.0407–0.0416 | 0.0217–0.0222 | 1.87× |
| 1,000 × 384 | 0.1834–0.1835 | 0.0394–0.0396 | 4.64–4.65× |
| 10,000 × 64 | 0.9168–0.9379 | 0.0894–0.0964 | 9.73–10.26× |
| 10,000 × 384 | 2.7096–2.8504 | 0.4460–0.4924 | 5.79–6.08× |
| 10,000 × 1,024 | 7.4605–7.6460 | 1.6790–1.6900 | 4.44–4.52× |

All 72 paired confidence intervals for cached-search mean savings were above zero.
Small absolute savings on tiny indexes should not be extrapolated to inference.
For 10,000 × 1,024, k=5, traced transient allocation peak falls from 41,040,976 to
121,128 bytes. This is Python/NumPy traced memory, not a total process RSS bound;
the owned 40.96 MB matrix still exists, and index construction temporarily computes
its norms. The additional persistent cache is 40 KB. No large model was loaded.

Reproduce from the repository root with the active environment's Python:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python tools/benchmark_retrieval.py --samples 60 --output artifacts/retrieval-first.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python tools/benchmark_retrieval.py --samples 60 --reverse-cases \
  --output artifacts/retrieval-reverse.json
```

Validation: 587 portable tests passed, 455 native tests deselected; 51 new tests
cover cutoff ties, signed zero, full/partial boundaries, random scores, extreme
finite magnitudes, invalid queries/documents, input ownership and SQLite reload.
Ruff and strict mypy passed for changed production sources. No native/GPU behavior
changed. Tokenizer reuse, asynchronous ingestion and GPU/ANN search remain separate
experiments requiring their own lifecycle and quality contracts.
