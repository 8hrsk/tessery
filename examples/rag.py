"""Create an index, reload it, and retrieve cited context for another project's LLM."""

import argparse
from pathlib import Path

from metal_inference import DocumentIndex, EmbeddingModel, ModelProfile, read_documents

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--profile-file", required=True)
parser.add_argument("--index", type=Path, required=True)
parser.add_argument("--query", default="What is the capital of France?")
args = parser.parse_args()
with EmbeddingModel.load(
    args.model_dir, profile=ModelProfile.from_file(args.profile_file)
) as model:
    if not args.index.exists():
        documents = read_documents(Path(__file__).parent / "documents")
        DocumentIndex.build(model, documents).save(args.index)
    hits = DocumentIndex.load(args.index).search(model, args.query, k=2)
    for hit in hits:
        print(f"[{hit.chunk.source}:{hit.chunk.start}-{hit.chunk.end}] {hit.score:.4f}")
        print(hit.chunk.text)
# Pass these source-labelled passages and the question to your chosen generator.
