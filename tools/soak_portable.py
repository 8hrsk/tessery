"""Bounded, model-free CPU soak for Kaggle; never qualifies Metal inference."""

import argparse
import gc
import hashlib
import json
import platform
import resource
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from metal_inference.errors import CanceledError
from metal_inference.tokenizer import QwenTokenizer
from metal_inference.wordpiece import WordPieceTokenizer
from tessery import DocumentIndex, EmbeddingModel, ModelDescriptor

ROOT = Path(__file__).resolve().parents[1]


class SyntheticBackend:
    """Deterministic vectors to exercise queue/index plumbing, not model quality."""

    max_padded_tokens = 128

    def forward(self, ids, lengths, *, dimensions):
        out = np.zeros((len(ids), dimensions), np.float32)
        for row, length in enumerate(lengths):
            for token in ids[row, :length]:
                out[row, int(token) % dimensions] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def close(self):
        pass


def source_hash():
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    paths += sorted((ROOT / "tools/fixtures").glob("*.json"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rss():
    stat = Path("/proc/self/statm")
    if stat.exists():
        import os

        return int(stat.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    # macOS fallback is peak RSS, explicitly identified in the report.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def cycle(model, bpe, wordpiece, directory):
    texts = ["hello worlds Café", "中文 hello", "hello[MASK]worlds"]
    expected = model.encode(texts)
    np.testing.assert_array_equal(model.encode(texts), expected)
    for tokenizer, text in [(bpe, "ab" * 10000), (wordpiece, "hello " * 10000)]:
        checks = 0

        def canceled():
            nonlocal checks
            checks += 1
            return checks == 10

        try:
            tokenizer.batch([text], max_length=512, canceled=canceled)
        except CanceledError:
            pass
        else:
            raise AssertionError("CPU tokenizer did not observe cancellation")
        assert checks == 10
        tokenizer.encode("hello")
    index = DocumentIndex.build(model, {"one": texts[0], "two": texts[1]})
    path = directory / "index.sqlite"
    index.save(path)
    try:
        loaded = DocumentIndex.load(path)
        assert loaded.chunks == index.chunks
        assert loaded.search(model, texts[0], k=1)[0].chunk.source == "one"
    finally:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-python-growth-mib", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 86400 or args.max_python_growth_mib < 1:
        parser.error("seconds must be 1..86400 and growth budget must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting","metal_qualified":false}\n')
    started = time.monotonic()
    payload = {
        "status": "running",
        "scope": "portable_cpu_synthetic_backend",
        "metal_qualified": False,
        "model_inference_qualified": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": source_hash(),
        "requested_seconds": args.seconds,
        "iterations": 0,
        "checkpoints": [],
        "rss_kind": "current" if Path("/proc/self/statm").exists() else "peak",
        "python_growth_budget_bytes": args.max_python_growth_mib * 1024**2,
    }

    def checkpoint():
        payload["elapsed_seconds"] = time.monotonic() - started
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    try:
        bpe = QwenTokenizer(json.loads((ROOT / "tools/fixtures/bpe.json").read_text()))
        wordpiece = WordPieceTokenizer(
            json.loads((ROOT / "tools/fixtures/wordpiece.json").read_text())
        )
        with (
            tempfile.TemporaryDirectory() as temp,
            EmbeddingModel(
                SyntheticBackend(),
                wordpiece,
                32,
                64,
                8,
                descriptor=ModelDescriptor(),
            ) as model,
        ):
            directory = Path(temp)
            for _ in range(5):
                cycle(model, bpe, wordpiece, directory)
            gc.collect()
            tracemalloc.start()
            baseline = tracemalloc.get_traced_memory()[0]
            run_started = last = time.monotonic()
            while time.monotonic() - run_started < args.seconds:
                cycle(model, bpe, wordpiece, directory)
                payload["iterations"] += 1
                now = time.monotonic()
                if now - last >= 30:
                    gc.collect()
                    growth = tracemalloc.get_traced_memory()[0] - baseline
                    payload["checkpoints"].append(
                        {
                            "seconds": now - run_started,
                            "rss_bytes": rss(),
                            "python_growth_bytes": growth,
                        }
                    )
                    assert growth <= payload["python_growth_budget_bytes"]
                    checkpoint()
                    last = now
            gc.collect()
            payload["python_growth_bytes"] = tracemalloc.get_traced_memory()[0] - baseline
            assert payload["python_growth_bytes"] <= payload["python_growth_budget_bytes"]
            payload["rss_bytes"] = rss()
            payload["workload_seconds"] = time.monotonic() - run_started
        assert payload["source_sha256"] == source_hash(), "Sources changed during soak"
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        checkpoint()
        print(json.dumps({k: v for k, v in payload.items() if k != "checkpoints"}), flush=True)


if __name__ == "__main__":
    main()
