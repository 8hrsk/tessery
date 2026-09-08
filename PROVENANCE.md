# Provenance and implementation boundary

Status: independent native Metal implementation; **not a clean-room
certification or legal sign-off**. License: Apache-2.0.

## Sources consulted

* User-supplied technical specification, draft 1.0 (verbatim copy in
  `docs/specification.ru.md`).
* [Official Qwen3-Embedding-0.6B model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B):
  architecture family, native dimensions, pooling/projection specification.
* [Pinned model artifacts](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ/tree/6c3ae70858513f1a78e9cdca3cae330d9075cd2a):
  JSON configuration and tokenizer data; no executable model code was consulted.
* [Official MLX platform documentation](https://ml-explore.github.io/mlx/build/html/install.html).
* Local Yuri Agent commit `91b8ae81f58160f699a2378e1b38d4d449f4544e`:
  `scripts/install-macos-rag-model.sh`,
  `internal/retrieval/modelpack/macos.go`,
  `internal/retrieval/modelpack/owned_runtime.go` (call-site/protocol observations),
  and `docs/specs/GO_EMBEDDING_INTEGRATION_TZ.md` (untracked draft at inspection).
* Installed **metadata only** for the existing baseline environment:
  mlx-embeddings 0.1.0, MLX 0.32.2, transformers 5.16.1, tokenizers 0.23.2,
  NumPy 2.5.2, CPython 3.12.13. The caller pins mlx-serve commit
  `4f10f812356a766c49bc73efa1a39eacafbfb0dc`; that does not transitively pin
  mlx-embeddings or its dependencies.

## Separation

No mlx-embeddings implementation source was opened, copied or adapted in this
work. `benchmarks/isolated/capture.py` invokes its public API in the pre-existing
external environment and is excluded from the candidate wheel and sdist.
The earlier foundation imported only the standard library; the native engine adds NumPy and regex. Synthetic observed
vectors and token IDs are data, not implementation source.

On 2026-09-08 the user changed the scope to a reusable independent Metal engine
and authorized API/implementation work without Go fixture gates. The same agent
then wrote the original native kernels, runtime, BPE and Qwen adapter. This is
documented independent development, not the original role-separated clean-room
process. Prior GPL-source exposure must still be disclosed by future contributors;
no clean-room certification or legal sign-off is claimed.

Additional public sources consulted:

* [Apple Metal compute documentation](https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu)
  and the installed Apple SDK interfaces.
* [Qwen3 public reference architecture](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py),
  Apache-2.0, as architecture documentation. Transformers is not imported.
* [Published affine quantization format](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html).
  MLX is consulted only for weight format documentation, not used as a backend.
* The pinned configuration, tokenizer JSON and SafeTensors tensor metadata.
  BPE merge scheduling and numerical kernels were written for this project.

No mlx-embeddings GPL implementation source was consulted for this implementation.
NumPy and regex license notices are preserved in `third_party/` from the exact
installed distributions. This license inventory is not a legal approval.

## Required PR declaration

* List specification, API documentation and model artifact sources consulted.
* Declare whether GPL implementation source was previously accessed.
* Confirm no GPL code was copied/adapted and no GPL dependency was introduced.
* Identify corpus/spec reviewer separately from model implementer.
* Include relevant artifact hashes, test evidence and compatibility ID effects.

These declarations are review evidence, not automatic legal approval. Dependency
allowlisting is not a substitute for a release similarity scan, complete
third-party notices, model redistribution review or signed compatibility report.
