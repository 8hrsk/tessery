# Reusing validated tokenizer rows during index ingestion

The experiment branches from retrieval commit `273bcc6` in a separate
`perf/rag-token-reuse` worktree. It removes the second tokenization of an accepted
chunk in `DocumentIndex.build`; failed saturation probes still run normally.

The builder retains accepted uint32 token rows with a **16 MiB token-payload cap**.
At most 10,000 row objects and list entries add bounded Python metadata overhead;
16 MiB is not a total RSS limit. Once the payload cap is reached, batches without
all prepared rows use ordinary `encode`. Rows are released after their batch
finishes. The existing chunk, text, vector and snapshot limits remain in force,
and the complete index payload is checked before embedding begins.

A private `_encode_prepared` method reuses the normal text validation, validates
uint32 row shape, vocabulary range and token limits, and copies rows into owned,
read-only padded arrays. Those arrays then enter the same single executor,
admission semaphore, cancellation checks and execution batching as `encode`.
A queued request cannot observe mutations of the original rows or text list.
There is no new public token-ID entry point, tokenizer cache on the model or
persistent token data in index files.

Every probe captures the exact cap it passes to the tokenizer. A cap change during
chunk probing permanently disables reuse for that build, including a change that
later returns to its initial value. A cap change after preparation causes ordinary
worker tokenization. Tests cover the 64→32→64 probe transition. The route assumes
the model's private tokenizer object remains the same, as other model operations do.

## CPU measurements with the actual local tokenizers

[Two-process summary](../benchmarks/retrieval/token-reuse-20260912.json) records
six cases per process: Qwen and BGE tokenizers, 256/600-character chunks and
128/512-token caps. The corpus contains English, Russian and mixed Unicode texts,
with document/query prefixes. Each variant receives 12 samples with alternating
order; the second fresh process reverses model, case and pair order. Uncertainty
intervals resample whole two-order paired blocks.

The tokenizers and their existing local, pinned artifacts are real. The encoder
is a deterministic CPU-only fake backend. **These are CPU index-building results,
not end-to-end Metal ingestion throughput or an MLX comparison.** The measurement
includes chunk construction, tokenizer probes, API admission/batching, deterministic
fake encoding and final index construction. No model weights were loaded or downloaded.

Every measured build preserves exact chunk records, vector output and the SHA-256
of every backend token-ID/length batch. Original and prepared paths therefore send
the same integer inputs to model execution. Each accepted chunk removes exactly
one tokenization; failed saturation probes explain the smaller gain at cap 128.

| Tokenizer | Chunk chars / token cap | Chunks | Tokenized texts old → reused | Old / reused p50 |
|---|---:|---:|---:|---:|
| Qwen | 256 / 512 | 53 | 106 → 53 | 1.951–1.952× |
| Qwen | 600 / 512 | 23 | 46 → 23 | 1.978–1.979× |
| Qwen | 600 / 128 | 36 | 98 → 62 | 1.452–1.458× |
| BGE | 256 / 512 | 53 | 106 → 53 | 1.966–1.969× |
| BGE | 600 / 512 | 23 | 46 → 23 | 1.982–1.986× |
| BGE | 600 / 128 | 120 | 531 → 411 | 1.148–1.149× |

At 600 characters / cap 512, Qwen's first-run p50 drops from 31.80 to 16.07 ms;
BGE drops from 35.74 to 18.03 ms. All 12 paired confidence intervals for CPU time
saved are above zero. Real Metal inference adds its unchanged cost to both paths,
so it will reduce these end-to-end ratios. Shared-host measurements should not be
interpreted as fixed latency guarantees.

For these small corpora, traced transient peak memory rises by roughly 2–27 KiB
in the first process (for example BGE 600/cap512: 209,277 → 235,945 bytes).
The 16 MiB payload cap and ordinary fallback protect larger builds. Existing
unit tests plus 16 new prepared-input tests pass: **603 portable tests passed,
455 native tests deselected**. New cases cover truncation/prefixes, complete
backend input equivalence, >32 chunks, cap fallback, malformed inputs, changed
configuration, owned queued snapshots, FIFO/admission, canceled and closed work.
Ruff, mypy and whitespace checks pass. Native model timing is still a separate gate.

Reproduce with the repository's environment; no Metal opt-in is needed:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python tools/benchmark_index_token_reuse.py --samples 12 \
  --output artifacts/token-reuse-first.json
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python tools/benchmark_index_token_reuse.py --samples 12 --reverse \
  --output artifacts/token-reuse-reverse.json
```
