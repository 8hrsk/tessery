"""Separate shader-library and kernel-route effects using three fixed Qwen models."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import re
import resource
import subprocess
import time
from contextlib import ExitStack
from functools import partial
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from diagnose_metal import delta, save_json, source_hashes, summarize
from shader_library_control import force_old_linear, revision_shader, runtime_with_shader

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel


def memory_snapshot():
    swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([KMG])", swap)
    assert match is not None
    used = float(match.group(1)) * {"K": 2**10, "M": 2**20, "G": 2**30}[match.group(2)]
    return {
        "rss_bytes": int(subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())]))
        * 1024,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "system_pressure_level": int(
            subprocess.check_output(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True
            )
        ),
        "system_swap_used_bytes": int(used),
    }


def balanced_orders(labels, samples, seed):
    assert samples >= 24 and samples % 24 == 0
    orders = list(permutations(labels))
    rng = np.random.default_rng(seed)
    return [orders[i] for _ in range(samples // 24) for i in rng.permutation(len(orders))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--old-revision", default="e49b39c")
    parser.add_argument("--new-revision", default="6bc4035")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument(
        "--allocation-order",
        choices=["baseline-first", "candidate-first"],
        default="baseline-first",
    )
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 24 <= args.samples <= 96 or args.samples % 24:
        parser.error("samples must be a multiple of24 from24..96")
    old_revision, old_shader = revision_shader(args.old_revision)
    new_revision, new_shader = revision_shader(args.new_revision)
    if old_shader == new_shader:
        parser.error("old and new shader bytes must differ")
    helpers = [
        Path(__file__),
        Path(__file__).with_name("shader_library_control.py"),
        Path(__file__).with_name("diagnose_metal.py"),
    ]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    allocation = ["baseline", "legacy", "candidate"]
    if args.allocation_order == "candidate-first":
        allocation.reverse()
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "helper_hashes": hashes,
        "harness_sha256": hashes[Path(__file__).name],
        "old_revision": old_revision,
        "new_revision": new_revision,
        "requested_shader_hashes": {
            "old": hashlib.sha256(old_shader).hexdigest(),
            "new": hashlib.sha256(new_shader).hexdigest(),
        },
        "allocation_order": allocation,
        "reverse_cases": args.reverse_cases,
        "samples_per_label": args.samples,
        "conditions": (
            "Three resident models; four labels; all24 permutations balanced; "
            "fixed plans; memory checks between six-permutation groups"
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "regex": importlib.metadata.version("regex"),
        "environment": {
            k: os.environ.get(k)
            for k in (
                "MTL_SHADER_VALIDATION",
                "MTL_DEBUG_LAYER",
                "MTL_DEVICE_WRAPPER_TYPE",
                "VECLIB_MAXIMUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "models": {},
        "results": [],
    }
    instances = {}
    payload["memory_samples"] = []
    initial_swap = None

    def check_memory(phase):
        nonlocal initial_swap
        sample = memory_snapshot()
        sample["phase"] = phase
        payload["memory_samples"].append(sample)
        if initial_swap is None:
            initial_swap = sample["system_swap_used_bytes"]
        if sample["system_pressure_level"] != 1:
            raise MemoryError("system_memory_pressure_not_normal")
        if sample["system_swap_used_bytes"] > initial_swap + 16 * 1024 * 1024:
            raise MemoryError("system_swap_growth_exceeded_16MiB")

    try:
        check_memory("before_load")
        with ExitStack() as stack:
            for name in allocation:
                source = old_shader if name == "baseline" else new_shader
                created = time.perf_counter()
                with patch(
                    "metal_inference.qwen3.MetalRuntime", partial(runtime_with_shader, source)
                ):
                    model = stack.enter_context(EmbeddingModel.load(args.model_dir))
                runtime = model._backend.runtime
                assert runtime._plans_enabled
                if name != "candidate":
                    force_old_linear(runtime)
                instances[name] = model
                check_memory("loaded_" + name)
                payload["models"][name] = {
                    "load_seconds": time.perf_counter() - created,
                    "actual_shader_sha256": runtime.diagnostics()["shader_sha256"],
                    "forced_route": "old16" if name != "candidate" else "new32",
                    "model_id": model.descriptor.model_id,
                    "compatibility_id": model.descriptor.compatibility_id,
                    "profile_sha256": model._backend.profile.identity_sha256,
                }
            for field in ("model_id", "compatibility_id", "profile_sha256"):
                assert len({m[field] for m in payload["models"].values()}) == 1
            labels = ["baseline_a", "baseline_b", "legacy", "candidate"]
            models = {
                n: instances["baseline"] if n.startswith("baseline_") else instances[n]
                for n in labels
            }
            cases = [128, 512]
            if args.reverse_cases:
                cases.reverse()
            for seq in cases:
                texts = [" token" * (seq - 1)]
                ids, lengths = instances["baseline"]._tokenizer.batch(texts, max_length=512)
                assert lengths.tolist() == [seq]
                for model in instances.values():
                    other_ids, other_lengths = model._tokenizer.batch(texts, max_length=512)
                    np.testing.assert_array_equal(other_ids, ids)
                    np.testing.assert_array_equal(other_lengths, lengths)
                plans = list(execution_batches(lengths, 4096, 512, "qwen3_uint4"))
                elapsed = {n: 0.0 for n in labels}
                warmups = {n: 0 for n in labels}
                references = {}
                rng = np.random.default_rng(5011 + seq)
                while any(elapsed[n] < 1.0 or warmups[n] < 3 for n in labels):
                    for name in rng.permutation(labels):
                        started = time.perf_counter()
                        references[name] = models[name].encode(texts)
                        elapsed[name] += time.perf_counter() - started
                        warmups[name] += 1
                for value in references.values():
                    np.testing.assert_array_equal(value, references["baseline_a"])
                timings = {n: [] for n in labels}
                gpu = {n: [] for n in labels}
                encode = {n: [] for n in labels}
                counters = {n: [] for n in labels}
                orders = balanced_orders(labels, args.samples, 9113 + seq)
                check_memory(f"warm_{seq}")
                for order_index, order in enumerate(orders):
                    for name in order:
                        runtime = models[name]._backend.runtime
                        before = runtime.diagnostics()
                        started = time.perf_counter()
                        output = models[name].encode(texts)
                        wall = time.perf_counter() - started
                        after = runtime.diagnostics()
                        measured = delta(before, after)
                        assert after["plan_hits"] > before["plan_hits"]
                        assert after["plan_builds"] == before["plan_builds"]
                        assert measured["dispatches"].get("linear4_32x32_k64", 0) == (
                            140 if name == "candidate" else 0
                        )
                        assert measured["dispatches"].get("linear4_16x32_k64", 0) == (
                            0 if name == "candidate" else 140
                        )
                        np.testing.assert_array_equal(output, references["baseline_a"])
                        timings[name].append(wall)
                        gpu[name].append(measured["gpu_seconds"])
                        encode[name].append(measured["encode_seconds"])
                        counters[name].append(measured)
                    if (order_index + 1) % 6 == 0:
                        check_memory(f"measured_{seq}_{order_index + 1}")
                baseline = float(np.median(timings["baseline_a"] + timings["baseline_b"]))
                legacy = float(np.median(timings["legacy"]))
                candidate = float(np.median(timings["candidate"]))
                row = {
                    "lengths": [seq],
                    "plans": [(r.tolist(), w) for r, w in plans],
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "vectors": references["baseline_a"].tolist(),
                    "exact_equality": True,
                    "warmups": warmups,
                    "measurement_orders": orders,
                    "api_timings": {n: summarize(t, 1) for n, t in timings.items()},
                    "gpu_timings": {n: summarize(t, 1) for n, t in gpu.items()},
                    "encode_timings": {n: summarize(t, 1) for n, t in encode.items()},
                    "normal_runtime_samples": counters,
                    "ratios": {
                        "old_library_over_new_library_old_route": baseline / legacy,
                        "new_library_old_over_new_route": legacy / candidate,
                        "old_library_old_over_new_library_new": baseline / candidate,
                        "identical_a_over_b": float(
                            np.median(timings["baseline_a"]) / np.median(timings["baseline_b"])
                        ),
                    },
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {
                        "seq": seq,
                        "ratios": row["ratios"],
                        "api_p50_ms": {
                            n: round(t["p50_seconds"] * 1000, 3)
                            for n, t in row["api_timings"].items()
                        },
                    },
                    flush=True,
                )
            payload["runtime_final"] = {
                n: m._backend.runtime.diagnostics() for n, m in instances.items()
            }
        assert all(
            m._backend.runtime.active_bytes
            == m._backend.runtime.cache_bytes
            == m._backend.runtime._plan_bytes
            == 0
            for m in instances.values()
        )
        assert payload["source_hashes"] == source_hashes()
        assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in helpers}
        check_memory("after_close")
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        if isinstance(error, MemoryError):
            payload["memory_stop_reason"] = str(error)
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
