"""Offline reference capture in an existing, separate torch/transformers environment.

Not a runtime dependency. Synthetic inputs only; no downloads or remote code.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = (
        AutoModel.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, attn_implementation="eager"
        )
        .eval()
        .cpu()
    )
    cases = [
        "What is the capital of France?",
        "Paris is the capital of France.",
        "Bananas grow in tropical climates.",
        "How do I reset my password?",
        "To reset your password, open account settings.",
        "The cat is sleeping on a sofa.",
        "HELLO, World!",
        "Café naïve re\u0301sume\u0301",
        "中文测试 日本語 한국어",
        "Привет, мир!",
        "ΟΔΥΣΣΕΥΣ ΣΟΣ İ I ı",
        "a\u0000b\ufffdc\u200dd\ue000e\u0378f",
        "a\t\nb\rc\u00a0d\u0085e\u001cf",
        "hello[MASK]world [CLS] [SEP] [PAD] [UNK]",
        "[mask] [Mask]",
        "a" * 101,
        "🤖👨‍👩‍👧‍👦",
        "can't foo_bar $5.99 + a=b",
        "\u0301",
        "\u200b",
        "𠀀 丽",
        "[MASK][MASK]",
    ]
    token_rows = []
    for limit in (2, 3, 16, 512):
        for text in cases + [" token" * 511]:
            row = tokenizer(text, truncation=True, max_length=limit)
            token_rows.append({"text": text, "max_length": limit, "ids": row["input_ids"]})
    batches = [cases[:6], cases[6:10], [cases[0]], list(reversed(cases[:6]))]
    vectors = []
    with torch.inference_mode():
        for texts in batches:
            inputs = tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            outputs = model(**inputs).last_hidden_state[:, 0]
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
            vectors.append(
                {
                    "texts": texts,
                    "ids": inputs["input_ids"].tolist(),
                    "mask": inputs["attention_mask"].tolist(),
                    "vectors": outputs.tolist(),
                }
            )
    data = {
        "model_revision": root.name,
        "artifacts": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "model.safetensors")
        },
        "environment": {
            "python": platform.python_version(),
            "device": "cpu",
            "attention": "eager",
            **{
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers")
            },
        },
        "token_cases": token_rows,
        "batches": vectors,
    }
    with Path(args.out).open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
