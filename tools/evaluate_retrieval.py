"""Small hand-authored retrieval regression, without model downloads."""

import argparse
import hashlib
import json
from pathlib import Path

from diagnose_metal import save_json, source_hashes

from tessery import DocumentIndex, EmbeddingModel, ModelProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Reject rank regressions against a saved run")
    args = parser.parse_args()
    fixture = Path(__file__).parent / "fixtures/retrieval-mini.json"
    corpus = json.loads(fixture.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "scope": corpus["description"],
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "results": [],
    }
    try:
        profile = (
            ModelProfile.from_file(args.profile_file)
            if args.profile_file
            else "qwen3-embedding-0.6b-dwq"
        )
        with EmbeddingModel.load(args.model_dir, profile=profile) as model:
            payload["contract"] = {
                "compatibility_id": model.descriptor.compatibility_id,
                "manifest_sha256": model.descriptor.manifest_sha256,
                "dimensions": model.dimensions,
                "max_length": model.max_length,
            }
            # BGE-small-en is evaluated only on English. Keep all texts unprefixed,
            # matching the default library API; record this preprocessing choice.
            cases = [
                c
                for c in corpus["cases"]
                if c["language"] == "en" or model.descriptor.architecture == "qwen3_uint4"
            ]
            payload["preprocessing"] = "default API; empty document/query prefixes"
            index = DocumentIndex.build(
                model, {c["id"]: c["document"] for c in cases}, chunk_chars=600, overlap_chars=0
            )
            for case in cases:
                hits = index.search(model, case["query"], k=len(cases))
                rank = next(i + 1 for i, h in enumerate(hits) if h.chunk.source == case["id"])
                payload["results"].append(
                    {
                        "id": case["id"],
                        "language": case["language"],
                        "rank": rank,
                        "top_id": hits[0].chunk.source,
                    }
                )
            payload["recall_at_1"] = sum(c["rank"] == 1 for c in payload["results"]) / len(cases)
            payload["mrr"] = sum(1 / c["rank"] for c in payload["results"]) / len(cases)
        assert payload["source_hashes"] == source_hashes()
        payload["status"] = "measured"  # Scores, not an invented release-quality threshold.
        if args.baseline:
            baseline = json.loads(args.baseline.read_text())
            assert baseline["status"] in {"measured", "passed"}
            for key in ("fixture_sha256", "preprocessing", "contract"):
                assert baseline[key] == payload[key], f"Incompatible baseline: {key}"
            expected = {r["id"]: r["rank"] for r in baseline["results"]}
            assert set(expected) == {r["id"] for r in payload["results"]}
            assert all(r["rank"] <= expected[r["id"]] for r in payload["results"])
            payload["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
            payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)
        print({k: v for k, v in payload.items() if k not in ("source_hashes", "results")})


if __name__ == "__main__":
    main()
