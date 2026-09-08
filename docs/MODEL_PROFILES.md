# Model profiles

Tessery separates model data from architecture implementation. A `ModelProfile`
describes three local artifacts and a supported architecture/tokenizer/pooling
combination. Files must match its byte sizes and SHA-256 digests. No model Python
files, plugin imports, automatic conversions or download hooks are executed.

| Built-in name | Adapter | Storage | Tokenizer / pooling | Output | Sequence length |
| --- | --- | --- | --- | --- | --- |
| `qwen3-embedding-0.6b-dwq` | Qwen3 | affine uint4, group 64, BF16 scales/biases | NFC byte BPE / last nonpadding token | 32..1024 | 1..512 |
| `bge-small-en-v1.5` | BERT | float32 | BERT WordPiece / CLS | 384 | 2..512 |

Both normalize output with L2 and use the same Metal runtime. The Qwen3 model
supports prefix projection; BGE is not registered as a Matryoshka model and its
output is not truncated. No query instruction is injected. If an application
uses BGE's query instruction, it should prepend it explicitly and keep that
policy stable when indexing/querying. See the [BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5).

## Load a named profile

```python
from metal_inference import EmbeddingModel, get_profile, list_profiles

print(list_profiles())
profile = get_profile("bge-small-en-v1.5")
with EmbeddingModel.load("/absolute/local/bge-small-en-v1.5", profile=profile) as model:
    embeddings = model.encode(["A question", "A passage"])
    print(model.descriptor)
```

The standard directory contains regular `config.json`, `tokenizer.json`, and
`model.safetensors` files. Other files are ignored. Symlinks, hardlinks, special
files and symlink ancestors in the artifact directory are rejected. Hashes are
checked on the same byte snapshots used for tokenization and GPU uploads.

## Reuse existing Hugging Face cache blobs without copying weights

Snapshot entries in the Hugging Face cache are usually symlinks. Point the engine
at the regular files in `blobs` using an explicit filename map. The repository
includes [a profile for the verified BGE revision](../model-manifests/bge-small-en-v1.5-hf-cache.json).
It contains no absolute machine paths and does not modify the cache.

Run from the checkout:

```python
from pathlib import Path
from metal_inference import EmbeddingModel, ModelProfile

profile = ModelProfile.from_file("model-manifests/bge-small-en-v1.5-hf-cache.json")
blobs = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs"
with EmbeddingModel.load(blobs, profile=profile) as model:
    vectors = model.encode(["What is the capital of France?", "Paris is the capital of France."])
```

```sh
metal-inference inspect \
  --model-dir "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json
```

Missing blobs fail validation; they are not downloaded. The built-in profile
and its cache filename mapping have the same compatibility ID.

## Describe another compatible model

Export a profile with `get_profile(name).to_dict()`, or use
`metal-inference profiles`. Its JSON schema is strict, version 1, with these fields:

* `model_id`, `revision`: model identity and immutable revision metadata.
* `architecture`, `tokenizer`, `pooling`: one of the supported table combinations.
* `native_dimensions`, `min_dimensions`, `default_dimensions`, `max_length`:
  bounds matching the model configuration and pooling contract.
* `artifacts`: exactly three objects, each with `name`, `filename`, `size`, `sha256`.
  Logical names are the three standard artifact names. Physical filenames are
  single basenames within the selected directory; paths and traversal are refused.

```python
from metal_inference import ModelProfile, get_profile

# Supply sizes/digests from your reviewed model artifacts, not arbitrary values.
base = get_profile("bge-small-en-v1.5")
manifest = base.to_dict()
# Edit manifest metadata and artifact records for the compatible model, then:
profile = ModelProfile.from_dict(manifest)
# Or load the complete JSON written by your model packaging process:
profile = ModelProfile.from_file("my-model-profile.json")
```

Profiles are immutable. A caller-selected profile is an explicit trust decision:
its hashes establish byte integrity, not the publisher's identity, licensing or
embedding quality. The loader validates configuration bounds, tokenizer behavior,
complete tensor names/shapes/dtypes and offsets before GPU allocation. Weights
are a single SafeTensors file; pickle and sharded checkpoints are unsupported.

Qwen3 supports the existing no-bias SiLU/RMSNorm/GQA/RoPE architecture, no sliding
window or RoPE scaling, and the fixed Qwen split/template behavior. BERT supports
absolute positions, erf-form GELU, no decoder/cross attention or pruned heads,
single-sequence token type zero, and normalized encoder CLS pooling. Optional
saved BERT position IDs and pooler tensors are validated but not uploaded; the
tanh pooler is not part of the embedding result. Configurable sizes are bounded
(up to 64 layers, hidden width 4096, intermediate width 16384, head width 256).
Only the two real packs above are qualified here; synthetic small packs verify
other dimensions without claiming model quality.

Profile fingerprints include artifact contents and the declared behavior.
Filename changes and artifact ordering do not alter them. Store output dimensions,
configured max length and input-prefix policy alongside the profile ID in a vector
index. A new hash or architecture is not compatible just because dimensions match.

## Native test inputs

`METAL_INFERENCE_TEST=1` enables all GPU tests, including both real models and
synthetic packs. Defaults use the local Qwen directory and BGE blob directory.
Override with:

* `METAL_INFERENCE_MODEL_DIR`: the verified Qwen3 model directory.
* `METAL_INFERENCE_BGE_MODEL_DIR`: BGE artifact directory.
* `METAL_INFERENCE_BGE_PROFILE_FILE`: a trusted BGE filename-map JSON.

Missing model data fails an explicitly enabled native run; it is never fetched.
`pytest -m 'not metal'` runs portable contract/parser/tokenizer tests without model
weights or Metal. CPU-reference collection runs separately using existing tools;
those frameworks are not dependencies of the distributable engine.
