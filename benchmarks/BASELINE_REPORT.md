# Local baseline observation — 2026-09-05

**Status: unreviewed observation. No compatibility or release approval.**

Data source: existing local Yuri runtime environment, CPython 3.12.13,
mlx-embeddings 0.1.0, MLX/Metal 0.32.2, transformers 5.16.1, tokenizers 0.23.2,
NumPy 2.5.2, macOS 26.3 arm64. Full installed inventory is recorded; there is
no reproducible baseline lock yet. This run used actual Metal after the sandbox
reported that no Metal device was available. Python socket access was denied.

All five required model hashes matched the Yuri Agent constants at commit
`91b8ae81f58160f699a2378e1b38d4d449f4544e`. Exact sizes and intended 0600 modes are
in `model-manifests/qwen3-0.6b-dwq.json`. The installed source model includes
additional tokenizer/support files; the observation records those hashes too.
Future tokenizer loading must establish every byte it reads and use the reviewed
exact production file set. This capture does not prove five files suffice.

## Observed preprocessing

* Padding and truncation sides: **right**.
* Plain text; no query/document instruction supplied by the Yuri caller.
* `Hello` without special injection: `[9707]`; with injection: `[9707, 151643]`.
* Tokenizer metadata reports `eos_token_id=151645`, `pad_token_id=151643`, no BOS.
  Thus the actual injected suffix **differs from tokenizer EOS metadata**.
  Future code must not infer injection from `eos_token_id` alone.
* Input IDs and masks are captured exactly (NumPy int64 in this run).
* Boundary inputs of 511/512/513 untruncated tokens yield lengths 511/512/512.
  In each case the suffix remains 151643; truncation reserves the special token.
* RoPE config: theta 1,000,000; scaling null. Position IDs are not provided by
  Yuri; backend-internal position construction and pooling remain unobserved.
* Public `text_embeds` returns 1024D **bfloat16** in this environment. The Yuri
  caller takes its first 384 components and normalizes using Python float math.
  This is recorded exactly; a new float32 implementation needs differential tests.

## Batch sensitivity

18 batches were captured: 15 single cases plus mixed batches of 15, 4 and 32.
Each has full input IDs, masks, 1024D output and projected 384D vectors.

Same-text single-vs-mixed baseline cosine minima:

| Batch | Minimum cosine | Rows below 0.99999 |
|---|---:|---:|
| 15 | 0.9993776664467899 | 12 |
| 4 | 0.9996287119310501 | 2 |
| 32 | 0.9993776664467899 | 26 |

These are baseline self-comparisons, **not candidate regressions**. They establish
that differential comparisons must pin identical batch composition and order.
They do not justify lowering the 0.99999 candidate threshold. The cause has not
been isolated; no claim about numerical kernels or pooling is made.

## Blocking evidence still required

Independent approval of corpus and retrieval methodology; baseline lock with
wheel hashes; complete observed position/pooling semantics; canonical tagged Go
fixtures; repeat determinism and same-batch differential tests; old/new query/doc
retrieval; required two-host sample populations and soak; signed compatibility
decision. No legacy engine/pack ID has been assigned to the new implementation.
