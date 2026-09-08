# Local retrieval with persisted indexes

Tessery can build and reload a small exact cosine index. It returns passages with
source names and character offsets, ready to give to a generator chosen by the
calling project. Tessery does not generate an answer or load a second model.

## Existing local BGE model

From the repository, using the already cached pack without copying weights:

```sh
python examples/rag.py \
  --model-dir "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs" \
  --profile-file model-manifests/bge-small-en-v1.5-hf-cache.json \
  --index /tmp/tessery-demo.sqlite \
  --query 'What is the capital of France?'
```

The example creates a snapshot only if it does not exist, reloads it, and prints
ranked context. It reads the small English documents supplied in `examples/documents`.
BGE is an English model. Use the existing Qwen profile for multilingual documents.

## Reuse in another project

```python
from metal_inference import DocumentIndex, EmbeddingModel, read_documents

with EmbeddingModel.load('/absolute/model') as model:
    index = DocumentIndex.build(model, read_documents('/absolute/documents'))
    index.save('/absolute/new-index.sqlite')
    index = DocumentIndex.load('/absolute/new-index.sqlite')
    for hit in index.search(model, 'Your question', k=5):
        print(hit.chunk.source, hit.chunk.start, hit.chunk.end, hit.score)
        print(hit.chunk.text)
```

`read_documents` reads UTF-8 `.txt` and `.md` files in sorted path order; empty
files and file symlinks are skipped. `build` also accepts a mapping of source
names to document strings, preserving its iteration order.

Chunks start with a 600-character window and 80-character requested overlap.
A window is repeatedly shortened if its tokenized length reaches the model cap.
This conservative check avoids silently indexing truncated passages (even an
exact-fit window is shortened). Effective overlap is at most half the accepted
window, so progress is guaranteed. Offsets are Python Unicode character indices,
not byte offsets. Whitespace-only spans may be omitted. This is a text splitter,
not a Markdown/PDF parser or a semantic chunker.

Options: `chunk_chars=1..8192`, `overlap_chars=0..chunk_chars-1`, explicit
`document_prefix=''`, and `query_prefix=''`. Prefix policy is saved with the index;
no model-specific instruction is inserted implicitly. A prefix or unusually small
token limit that leaves no room for a character causes `InvalidInputError`.
Queries use the model's regular truncation policy.

## Snapshot contract and bounds

SQLite stores normalized float32 vectors, original chunk text/offsets, prefixes,
chunk parameters, the complete model descriptor, dimensions, and max length.
Search rejects a different embedding contract with `ManifestError`. Use a new
snapshot when changing models or preprocessing. The checksum detects accidental
metadata/text/vector corruption; it is not a signature or authenticity proof.

`save` atomically publishes a new file from a temporary file in the same directory.
An existing destination is never overwritten. `load` opens SQLite read-only and
validates the bounded records and checksum. Only load snapshots you trust; this
is a local data format, not an untrusted database sandbox.

Bounds: 1,000 source documents, 16 MiB source UTF-8, 10,000 chunks, 128 MiB index
file, at most 64 MiB combined chunk/vector payload before SQLite overhead, and
`k=1..100`. Directory scanning keeps at most 1,000 eligible paths before sorting. The matrix and passages are loaded into RAM. Search is exact
NumPy cosine ranking with stable ties, not ANN; there is no incremental mutation,
background indexing, PDF extraction, or multi-process writer coordination.
Index files contain your document text and should be stored accordingly.

## CLI

```sh
metal-inference index --model-dir /absolute/model \
  --documents /absolute/documents --index /absolute/new-index.sqlite
metal-inference search --model-dir /absolute/model \
  --index /absolute/new-index.sqlite --query 'Your question' --top-k 5
```

Both commands accept the usual `--profile` or `--profile-file`, `--dimensions`
and `--max-length`. Index creation also accepts `--chunk-chars`, `--overlap-chars`,
`--document-prefix`, and `--query-prefix`. Search must use the same embedding
contract as creation. JSON results include chunk sources, offsets, text and scores.
