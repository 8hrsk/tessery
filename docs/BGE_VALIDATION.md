# BGE-small-en-v1.5 validation, 0.3 alpha

On 2026-09-08 the complete BGE-small-en-v1.5 checkpoint was found in the existing
local Hugging Face cache. It was reused directly; **zero new model bytes were
downloaded or copied**. Qwen3 remains in its original local directory.

## Inputs and reference

* Model: `BAAI/bge-small-en-v1.5`, revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
* Weights: 133,466,304 bytes, SHA-256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`.
* Tokenizer: 711,396 bytes, SHA-256 `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66`.
* Configuration: 743 bytes, SHA-256 `094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750`.
* Reference: existing Python 3.12.13, torch 2.14.0, transformers 5.16.1,
  tokenizers 0.23.2, CPU eager attention, evaluation/inference mode.
* Engine: macOS 26.3 arm64, original Metal kernels and runtime, Python float32 outputs.

The [collector](../benchmarks/reference/capture_bert.py) sets offline mode and
loads only the local checkpoint using public APIs. It adds no runtime dependency
to Tessery. [Saved reference data](../benchmarks/observations/bge-small-en-v1.5/reference.json)
contains synthetic texts, complete token IDs/masks, vectors, versions and hashes.
Neither the collector nor baseline data is shipped in the wheel or sdist.

## Results

* Exact token IDs matched **92 cases**, spanning lengths 2/3/16/512, Unicode,
  accents, CJK, control/private-use/unassigned characters, added tokens, unknown
  words and truncation. All four reference batch masks also matched.
* **17 embeddings in four identically composed batches** matched the independent
  CPU reference with maximum absolute component error **2.20e-7** and minimum
  float32 cosine **0.99999988** in the observed local run.
* Real-model tests enforce coordinate tolerances (`atol=5e-6`, `rtol=1e-4`) and
  cosine above `0.99999`, plus a simple English retrieval margin, repeatability,
  batch 32, 511/512/513 boundaries, dimension rejection and stable resident buffers.
* Owned resident GPU buffers: **132,848,640 bytes**. Unused pooler and saved position
  IDs are validated but not uploaded. This count excludes process/driver overhead.
* Small generated BERT and Qwen3 packs with different vocabulary/hidden/head/layer
  sizes run through the same adapters using caller-supplied profiles and hashes.
  They verify extensibility and failure handling; they are not trained models.

The complete 0.3 implementation passed **280 tests** with **95.03%**
combined Python statement/branch coverage, including the existing Qwen3 suite.
Native shader correctness is checked numerically; Python coverage does not
measure C++/Metal branches. GELU uses a numerical erf approximation, checked
against `math.erf` at 10,001 points from -12 to 12 with absolute tolerance 2e-6.

These results qualify the listed artifacts and corpus on one device. They are
not a multilingual BGE quality claim, MTEB evaluation, performance parity claim,
two-device release gate or general proof for every BERT model. BGE is the English
model described by its [publisher](https://huggingface.co/BAAI/bge-small-en-v1.5).
