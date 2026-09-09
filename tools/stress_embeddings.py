"""Seeded real-model stress: ragged batches, Unicode, cancellation, overload, reload."""

import argparse
import asyncio
import hashlib
import threading
from pathlib import Path

import numpy as np
from diagnose_metal import memory, save_json, source_hashes

from tessery import EmbeddingModel, ModelProfile
from tessery.errors import ClosedError, OverloadError


async def queue_stress(model, text):
    entered, release = threading.Event(), threading.Event()
    original = model._tokenizer.batch
    first = True

    def gated(texts, *, max_length, canceled=None):
        nonlocal first
        if first:
            first = False
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Queue test failed to release tokenizer")
        return original(texts, max_length=max_length, canceled=canceled)

    model._tokenizer.batch = gated
    jobs = []
    try:
        jobs.append(asyncio.create_task(model.encode_async([text])))
        assert await asyncio.to_thread(entered.wait, 10)
        jobs.extend(asyncio.create_task(model.encode_async([text])) for _ in range(5))
        await asyncio.sleep(0)
        # Admission cap is four: running + three queued, then two rejected.
        jobs[0].cancel()
        jobs[2].cancel()
        release.set()
        results = await asyncio.gather(*jobs, return_exceptions=True)
        assert sum(isinstance(x, asyncio.CancelledError) for x in results) == 2
        assert sum(isinstance(x, OverloadError) for x in results) == 2
        assert sum(isinstance(x, np.ndarray) for x in results) == 2
        await model.encode_async([text])  # Drain worker and prove admission recovery.
        return {"canceled": 2, "overloaded": 2, "completed": 2}
    finally:
        release.set()
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        model._tokenizer.batch = original


async def forward_cancellation(model, text):
    entered = threading.Event()
    original = model._backend.forward

    def observe(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)

    model._backend.forward = observe
    job = asyncio.create_task(model.encode_async([text]))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if job.done():
            await job
            return False  # Never count a completed request as canceled.
        job.cancel()
        try:
            await job
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation was not delivered")
        await model.encode_async(["recovery after cancellation"])
        return True
    finally:
        await asyncio.gather(job, return_exceptions=True)
        model._backend.forward = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10000:
        parser.error("rounds must be 1..10000")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "seed": args.seed,
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rounds": [],
        "reloads": 0,
        "closed_lifecycles": 0,
        "forward_cancellations": 0,
        "scope": "seeded_correctness_stress_not_latency_benchmark",
    }
    profile = (
        ModelProfile.from_file(args.profile_file)
        if args.profile_file
        else "qwen3-embedding-0.6b-dwq"
    )
    rng = np.random.default_rng(args.seed)
    corpus = [
        "Кошки и собаки",
        "中文 café e\u0301 😀",
        "a",
        " \t punctuation!? 123",
        "[MASK] token <|endoftext|>",
    ]
    corpus += [" token" * n for n in (6, 7, 8, 30, 31, 32, 127, 510, 511, 512)]
    model = None
    try:
        for round_id in range(args.rounds):
            if model is None:
                model = EmbeddingModel.load(args.model_dir, profile=profile, max_pending=4)
                payload["reloads"] += int(round_id > 0)
                loaded = model.memory_stats().active_bytes
            # Mix a long passage with short rows; compare batching against independent calls.
            size = (1, 3, 8, 32)[round_id % 4]
            choices = rng.choice(len(corpus), size=size, replace=True)
            texts = [corpus[i] for i in choices]
            expected = np.concatenate([model.encode([t]) for t in texts])
            actual = model.encode(texts)
            np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-4)
            order = rng.permutation(size)
            np.testing.assert_allclose(
                model.encode([texts[i] for i in order]), expected[order], atol=5e-6, rtol=1e-4
            )
            queue = asyncio.run(queue_stress(model, corpus[0]))
            canceled = asyncio.run(forward_cancellation(model, corpus[-1]))
            payload["forward_cancellations"] += int(canceled)
            model.trim_memory()
            assert model.memory_stats().active_bytes == loaded
            assert model.memory_stats().cache_bytes == 0
            payload["rounds"].append(
                {
                    "round": round_id,
                    "batch": size,
                    "corpus_indices": choices.tolist(),
                    "queue": queue,
                    "forward_canceled": canceled,
                    "after_trim": memory(model),
                    "max_abs_batch_error": float(np.max(np.abs(actual - expected))),
                }
            )
            if round_id % 3 == 2 or round_id == args.rounds - 1:
                model.close()
                assert model.memory_stats().active_bytes == 0
                try:
                    model.encode(["closed"])
                except ClosedError:
                    pass
                else:
                    raise AssertionError("Closed model accepted request")
                model = None
                payload["closed_lifecycles"] += 1
            save_json(args.output, payload)
            print({"round": round_id, "batch": size, "forward_canceled": canceled}, flush=True)
        assert payload["forward_cancellations"] > 0
        assert payload["source_hashes"] == source_hashes()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        if model is not None:
            model.close()
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
