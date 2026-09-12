"""Offline, single-model latency/memory/soak diagnostics. No external framework.

Run each model in a fresh process and keep raw JSON, including source hashes.
This is developer tooling: it intentionally accesses tokenizer/backend internals.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import resource
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from metal_inference import EmbeddingModel, ModelProfile, get_profile
from metal_inference.batching import execution_batches
from metal_inference.errors import ClosedError

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, payload):
    """Replace a checkpoint atomically; serialization/failure preserves the old file."""
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def source_hashes():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/metal_inference").rglob("*"))
        if path.suffix in {".py", ".mm", ".metal", ".dylib", ".so"}
    }


def memory(model):
    # ps reports current resident KiB; ru_maxrss is a lifetime high-water mark,
    # in bytes on macOS. Neither is the sum of the engine's owned Metal buffers.
    rss = int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())])) * 1024
    return {
        **asdict(model.memory_stats()),
        "process_rss_bytes": rss,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def delta(before, after):
    out = {}
    for key in (
        "allocations",
        "allocated_bytes",
        "completed_commands",
        "gpu_timed_commands",
        "gpu_seconds",
        "encode_seconds",
        "submit_wait_seconds",
    ):
        out[key] = after[key] - before[key]
    for key in ("plan_hits", "plan_builds"):
        out[key] = after.get(key, 0) - before.get(key, 0)
    out["dispatches"] = {
        k: v - before["dispatches"].get(k, 0)
        for k, v in after["dispatches"].items()
        if v != before["dispatches"].get(k, 0)
    }
    return out


def summarize(samples, batch_size):
    return {
        "latency_seconds": samples,
        "p50_seconds": float(np.percentile(samples, 50)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "texts_per_second": batch_size * len(samples) / sum(samples),
    }


def benchmark(model, batch_size, tokens, iterations, warmup):
    special = 2 if model.descriptor.architecture == "bert_f32" else 1
    texts = [" token" * (tokens - special)] * batch_size
    ids, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
    # Fail rather than label a different tokenizer's actual shape as requested.
    if ids.shape != (batch_size, tokens) or not np.all(lengths == tokens):
        raise ValueError("Synthetic text does not produce requested token count")
    for _ in range(warmup):
        model.encode(texts)
    before = model._backend.runtime.diagnostics()
    samples = []
    reference = None
    for _ in range(iterations):
        started = time.perf_counter()
        output = model.encode(texts)
        samples.append(time.perf_counter() - started)
        if reference is None:
            reference = output
        else:
            np.testing.assert_array_equal(output, reference)
    return {
        "batch_size": batch_size,
        "tokens_per_text": tokens,
        "padded_tokens": int(ids.size),
        "iterations": iterations,
        "warmup": warmup,
        **summarize(samples, batch_size),
        "runtime": delta(before, model._backend.runtime.diagnostics()),
        "memory": memory(model),
    }


async def exercise_concurrency(model, rounds):
    texts = ["a small local retrieval test", "documents about Metal inference"]
    expected = model.encode(texts)
    canceled = 0
    for _ in range(rounds):
        tasks = [asyncio.create_task(model.encode_async(texts)) for _ in range(6)]
        # Admit work, then cancel both the first and a queued request. Completion
        # timing is intentionally uncontrolled here; deterministic unit tests
        # separately gate the in-flight and queued cancellation paths.
        await asyncio.sleep(0)
        tasks[0].cancel()
        tasks[-1].cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, asyncio.CancelledError):
                canceled += 1
            elif isinstance(result, BaseException):
                raise result
            else:
                np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(await model.encode_async(texts), expected)
    return {"rounds": rounds, "canceled_tasks": canceled, "recovery_verified": True}


def soak(model, seconds, on_checkpoint=None):
    if not seconds:
        return {"requested_seconds": 0, "skipped": True}
    batches = [
        ["a short document"],
        [" token" * 126] * 4,
        [" token" * 510, "short", "a different text"],
        ["retrieval system"] * 32,
    ]
    special = 2 if model.descriptor.architecture == "bert_f32" else 1
    # Include each large-M projection guard, short paths and neighboring fallbacks.
    # Repeat each case immediately to exercise warm replay as well as eviction.
    batches += [
        [" token" * (length - special) for length in lengths]
        for lengths in (
            [3],
            [3, 7, 10],
            [33] * 4,
            [20] * 8,
            [24],
            [25],
            [159],
            [161],
            [16],
            [7, 7],
            [128],
            [256],
        )
    ]
    plans = []
    for texts in batches:
        _, lengths = model._tokenizer.batch(texts, max_length=model.max_length)
        plans.append(
            {
                "lengths": lengths.tolist(),
                "completed_calls": 0,
                "execution": [
                    (r.tolist(), w)
                    for r, w in execution_batches(
                        lengths,
                        model._backend.max_padded_tokens,
                        model.max_length,
                        model.descriptor.architecture,
                    )
                ],
            }
        )
    references = [model.encode(batch) for batch in batches]
    baseline = memory(model)
    before = model._backend.runtime.diagnostics()
    points = [{"elapsed_seconds": 0.0, **baseline}]
    started = time.perf_counter()
    count, next_sample = 0, 10.0

    def checkpoint():
        if on_checkpoint is not None:
            on_checkpoint(
                {
                    "status": "running",
                    "requested_seconds": seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                    "completed_calls": count,
                    "checkpoints": points,
                    "cases": plans,
                    "runtime": delta(before, model._backend.runtime.diagnostics()),
                }
            )

    checkpoint()
    while time.perf_counter() - started < seconds:
        i = (count // 2) % len(batches)
        np.testing.assert_array_equal(model.encode(batches[i]), references[i])
        stats = model.memory_stats()
        if (
            stats.active_bytes - stats.cache_bytes
            != baseline["active_bytes"] - baseline["cache_bytes"]
        ):
            raise AssertionError("Live Metal buffers grew between completed requests")
        if (
            stats.cache_bytes + stats.plan_cache_bytes
            > model._backend.runtime.workspace_limit_bytes
        ):
            raise AssertionError("Workspace and plan cache exceeded its shared budget")
        if model._backend.runtime.diagnostics()["plan_cache_entries"] > 4:
            raise AssertionError("Native plan count exceeded its bound")
        plans[i]["completed_calls"] += 1
        count += 1
        elapsed = time.perf_counter() - started
        if elapsed >= next_sample:
            points.append({"elapsed_seconds": elapsed, **memory(model)})
            checkpoint()
            next_sample = elapsed + 10
            print(f"soak {elapsed:.0f}s, {count} calls", flush=True)
    points.append({"elapsed_seconds": time.perf_counter() - started, **memory(model)})
    rss = [point["process_rss_bytes"] for point in points]
    return {
        "requested_seconds": seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_calls": count,
        "bitwise_repeatability": True,
        "live_buffers_stable": True,
        "workspace_cache_bounded": True,
        "execution_plan_cache_bounded": True,
        "rss_end_minus_start_bytes": rss[-1] - rss[0],
        "rss_sample_range_bytes": max(rss) - min(rss),
        "checkpoints": points,
        "cases": plans,
        "runtime": delta(before, model._backend.runtime.diagnostics()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", default="qwen3-embedding-0.6b-dwq")
    group.add_argument("--profile-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--soak-seconds", type=int, default=300)
    parser.add_argument("--concurrency-rounds", type=int, default=20)
    args = parser.parse_args()
    if not (
        1 <= args.iterations <= 1000
        and 1 <= args.warmup <= 20
        and 0 <= args.soak_seconds <= 86400
        and 0 <= args.concurrency_rounds <= 1000
        and all(1 <= n <= 32 for n in args.batch_sizes)
        and all(3 <= n <= 512 for n in args.lengths)
    ):
        parser.error("diagnostic options outside supported bounds")
    profile = (
        ModelProfile.from_file(args.profile_file)
        if args.profile_file
        else get_profile(args.profile)
    )
    if any(length > profile.max_length for length in args.lengths):
        parser.error("requested sequence length exceeds the selected profile")
    sources = source_hashes()
    output = {
        "schema_version": 1,
        "label": args.label,
        "scope": "single_host_diagnostic_not_release_gate",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "chip": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "source_hashes": sources,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": {"power_thermal_background_load": "uncontrolled", "network": "not required"},
        "model": profile.model_id,
        "compatibility_id": profile.compatibility_id,
        "dimensions": profile.default_dimensions,
        "max_length": profile.max_length,
        "matrix": [],
        "status": "running",
    }
    # Exclusive creation prevents accidentally replacing earlier evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)

    def save():
        save_json(args.output, output)

    model = None
    try:
        started = time.perf_counter()
        model = EmbeddingModel.load(args.model_dir, profile=profile)
        output["load_seconds"] = time.perf_counter() - started
        output["loaded_runtime"] = model._backend.runtime.diagnostics()
        started = time.perf_counter()
        model.encode(["warmup"])
        output["first_encode_seconds"] = time.perf_counter() - started
        for batch in args.batch_sizes:
            for length in args.lengths:
                row = benchmark(model, batch, length, args.iterations, args.warmup)
                output["matrix"].append(row)
                save()
                print(f"batch={batch} tokens={length} p50={row['p50_seconds']:.4f}s", flush=True)
        output["concurrency"] = asyncio.run(exercise_concurrency(model, args.concurrency_rounds))

        def checkpoint(record):
            output["soak"] = record
            save()

        output["soak"] = soak(model, args.soak_seconds, checkpoint)
        model.close()
        output["after_close"] = memory(model)
        assert output["after_close"]["active_bytes"] == 0
        try:
            model.encode(["must fail"])
        except ClosedError:
            output["closed_rejected"] = True
        else:
            raise AssertionError("Closed model accepted inference")
        # A fresh runtime must work after teardown, with the same profile.
        with EmbeddingModel.load(args.model_dir, profile=profile) as reloaded:
            reloaded.encode(["reload after close"])
        assert reloaded.memory_stats().active_bytes == 0
        output["reload_verified"] = True
        output["workspace_sources_unchanged"] = sources == source_hashes()
        if not output["workspace_sources_unchanged"]:
            raise RuntimeError("Workspace sources changed during measurement")
        output["status"] = "passed"
    except BaseException as error:
        output["status"] = "failed"
        output["error_type"] = type(error).__name__
        raise
    finally:
        if model is not None:
            model.close()
        save()


if __name__ == "__main__":
    main()
