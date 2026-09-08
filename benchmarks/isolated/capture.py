"""Optional black-box observer. Run ONLY in the existing external baseline env.

This file is excluded from wheels and sdists. It calls public APIs and does not
inspect, adapt, vendor or import baseline source into the candidate environment.
Only the synthetic corpus is accepted; no user data is read or persisted.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path


def write_new(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-corpus-sha256", required=True)
    args = parser.parse_args()
    if not args.model_dir.is_absolute() or args.out.exists():
        raise SystemExit("absolute local model and fresh output directory required")
    if sha(args.corpus) != args.expected_corpus_sha256:
        raise SystemExit("corpus digest mismatch")
    if importlib.metadata.version("mlx-embeddings") != "0.1.0":
        raise SystemExit("baseline version mismatch")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    # Additional Python network guard; this is not an OS-level offline proof.
    import socket

    def deny_network(*_args, **_kwargs):
        raise RuntimeError("baseline network access denied")

    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.getaddrinfo = deny_network

    import mlx.core as mx
    from mlx_embeddings import load

    corpus = json.loads(args.corpus.read_text())
    installed = sorted(
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}.items()
    )
    artifacts = [
        {"path": p.name, "size": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(args.model_dir.iterdir())
        if p.is_file() and not p.is_symlink()
    ]
    started = time.perf_counter()
    model, wrapper = load(str(args.model_dir))
    load_seconds = time.perf_counter() - started
    tokenizer = getattr(wrapper, "_tokenizer", wrapper)
    args.out.mkdir(parents=True)
    config = json.loads((args.model_dir / "config.json").read_text())
    profile = {
        "schema_version": 1,
        "status": "observed_not_reviewed",
        "corpus_sha256": sha(args.corpus),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "special_tokens": {
            name: getattr(tokenizer, name, None)
            for name in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]
        },
        "special_token_injection_probe": {
            "with_specials": tokenizer("Hello", add_special_tokens=True)["input_ids"],
            "without_specials": tokenizer("Hello", add_special_tokens=False)["input_ids"],
        },
        "roles": "plain_text",
        "max_length": 512,
        "rope_config": {k: v for k, v in config.items() if k.startswith("rope_")},
        "position_ids": "not supplied by Yuri; backend internal behavior unobserved",
        "pooling": "backend text_embeds; internal pooling unobserved",
        "projection": "first 384 components of text_embeds, then Python float unit L2",
        "artifacts": artifacts,
    }
    cases = {row["id"]: row for row in corpus["cases"]}
    # Check boundaries against actual tokenization instead of assuming word counts.
    for row in corpus["cases"]:
        if "expected_untruncated_tokens" in row:
            count = len(tokenizer(row["text"], truncation=False)["input_ids"])
            if count != row["expected_untruncated_tokens"]:
                raise SystemExit("frozen corpus token boundary mismatch")
    observations = []
    for batch in corpus["batches"]:
        texts = [cases[name]["text"] for name in batch]
        start = time.perf_counter()
        inputs = tokenizer(
            texts, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        output = model(
            mx.array(inputs["input_ids"]), attention_mask=mx.array(inputs["attention_mask"])
        )
        mx.eval(output.text_embeds)
        native = output.text_embeds.tolist()
        projected = []
        for vector in native:
            values = [float(value) for value in vector[:384]]
            norm = math.sqrt(sum(value * value for value in values))
            projected.append([value / norm for value in values])
        observations.append(
            {
                "case_ids": batch,
                "input_ids": inputs["input_ids"].tolist(),
                "attention_mask": inputs["attention_mask"].tolist(),
                "input_dtype": str(inputs["input_ids"].dtype),
                "mask_dtype": str(inputs["attention_mask"].dtype),
                "shape": list(inputs["input_ids"].shape),
                "native_dtype": str(output.text_embeds.dtype),
                "native_1024": native,
                "legacy_384": projected,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(f"captured batch {len(observations)}/{len(corpus['batches'])}", flush=True)
    write_new(args.out / "preprocessing-profile.json", profile)
    write_new(args.out / "vectors.json", observations)
    write_new(
        args.out / "environment.json",
        {
            "schema_version": 1,
            "status": "observation_only_not_release_baseline",
            "python": platform.python_version(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "packages": dict(installed),
            "installed_inventory_sha256": hashlib.sha256(
                json.dumps(installed, separators=(",", ":")).encode()
            ).hexdigest(),
            "lock_sha256": None,
            "lock_status": "installed metadata is not a reproducible dependency lock",
            "model_load_seconds": load_seconds,
            "network_guard": "Python sockets denied; OS sandbox proof outstanding",
            "corpus_sha256": sha(args.corpus),
            "outputs": {
                name: sha(args.out / name)
                for name in ["preprocessing-profile.json", "vectors.json"]
            },
        },
    )


if __name__ == "__main__":
    main()
