"""Balanced full-API qualification of fused MLP at 160 or 24 execution rows."""

import argparse
import hashlib
import time
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np
from diagnose_metal import ROOT, delta, save_json, source_hashes, summarize

from metal_inference.batching import execution_batches
from tessery import EmbeddingModel, MetalRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--rows", type=int, choices=[24, 160], required=True)
    parser.add_argument("--layout", choices=["split", "flat"], default="split")
    parser.add_argument("--candidate-shader", type=Path)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--reverse-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 6 <= args.samples <= 96 or args.samples % 6:
        parser.error("samples must be a multiple of six from 6..96")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": args.rows,
        "layout": args.layout,
        "samples_per_label": args.samples,
        "reverse_cases": args.reverse_cases,
        "results": [],
        "conditions": (
            "Public encode; old unfused MLP only at target height versus forced fused route; "
            "other heights keep current dispatch; unchanged up scratch; >=3 warmups and "
            ">=1s per label; balanced six permutations in shuffled blocks; uncontrolled thermals."
        ),
    }
    shader_path = ROOT / "src/metal_inference/native/kernels.metal"
    source = shader_path.read_bytes()
    if args.candidate_shader:
        candidate = args.candidate_shader.read_bytes()
        payload["candidate_sha256"] = hashlib.sha256(candidate).hexdigest()
        source += b"\n" + candidate
    read = Path.read_bytes

    def runtime(**kw):
        with patch.object(Path, "read_bytes", lambda p: source if p == shader_path else read(p)):
            return MetalRuntime(**kw)

    try:
        with patch("metal_inference.qwen3.MetalRuntime", runtime):
            model = EmbeddingModel.load(args.model_dir)
        with model:
            rt = model._backend.runtime
            rt._plans_enabled = False  # These experiments mutate kernel routing between calls.
            current = rt._gated4
            state = {"label": "previous_a"}

            def route(buffers, *, rows, cols, k):
                if rows != args.rows or (cols, k) != (3072, 1024):
                    return current(buffers, rows=rows, cols=cols, k=k)
                if state["label"] != "selected":
                    rt._linear4([*buffers[:4], buffers[7]], rows=rows, cols=cols, k=k)
                    rt._linear4([buffers[0], *buffers[4:7], buffers[8]], rows=rows, cols=cols, k=k)
                    rt._dispatch("silu_gate", buffers[7:], threads=rows * cols, n=rows * cols)
                    return None
                if rows == 160 or args.layout == "split":
                    rt._dispatch(
                        "gated4_16x32_k64",
                        buffers[:8],
                        threads=(rows // 16) * (cols // 32) * 256,
                        group_size=256,
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                if rows == 24:
                    offset = 16 if args.layout == "split" else 0
                    rt._dispatch(
                        "gated4_8x32",
                        buffers[:8],
                        threads=((rows - offset) // 8) * (cols // 32) * 128,
                        group_size=128,
                        n=offset,
                        rows=rows,
                        cols=cols,
                        k=k,
                    )
                return None

            rt._gated4 = route
            payload["runtime_shader_sha256"] = rt.diagnostics()["shader_sha256"]
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            cases = (
                [[160], [33] * 4, [32] * 5, [20] * 8, [161]]
                if args.rows == 160
                else [[24], [12] * 2, [8] * 3, [3, 7, 10], [25]]
            )
            if args.reverse_cases:
                cases.reverse()
            for case in cases:
                texts = [" token" * (n - 1) for n in case]
                ids, lengths = model._tokenizer.batch(texts, max_length=512)
                assert lengths.tolist() == case
                plans = [
                    (r.tolist(), w) for r, w in execution_batches(lengths, 4096, 512, "qwen3_uint4")
                ]
                targets = sum(len(r) * w == args.rows for r, w in plans)
                names = ["previous_a", "previous_b", "selected"]
                refs, warm = {}, {n: 0.0 for n in names}
                counts = {n: 0 for n in names}
                rng = np.random.default_rng(2100 + sum(case))
                while any(warm[n] < 1.0 or counts[n] < 3 for n in names):
                    for name in rng.permutation(names):
                        state["label"] = name
                        started = time.perf_counter()
                        refs[name] = model.encode(texts)
                        warm[name] += time.perf_counter() - started
                        counts[name] += 1
                timings, counters = {n: [] for n in names}, {n: [] for n in names}
                orders = list(permutations(names))
                for order in [
                    orders[i] for _ in range(args.samples // 6) for i in rng.permutation(6)
                ]:
                    for name in order:
                        state["label"] = name
                        before = rt.diagnostics()
                        started = time.perf_counter()
                        output = model.encode(texts)
                        elapsed = time.perf_counter() - started
                        d = delta(before, rt.diagnostics())
                        np.testing.assert_array_equal(output, refs["previous_a"])
                        if targets:
                            expected = 28 * targets if name == "selected" else 0
                            kernel = "gated4_16x32_k64" if args.rows == 160 else "gated4_8x32"
                            assert d["dispatches"].get(kernel, 0) == expected
                        timings[name].append(elapsed)
                        counters[name].append(d)
                a, b, new = [float(np.median(timings[n])) for n in names]
                old = float(np.median(timings["previous_a"] + timings["previous_b"]))
                row = {
                    "lengths": case,
                    "plans": plans,
                    "target_buckets": targets,
                    "input_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                    "vectors": refs["previous_a"].tolist(),
                    "exact_baseline_equality": True,
                    "timings": {n: summarize(v, len(case)) for n, v in timings.items()},
                    "normal_runtime_samples": counters,
                    "speedup": old / new,
                    "identical_control_a_over_b": a / b,
                    "control_within_10_percent": 0.9 <= a / b <= 1.1,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print(
                    {k: row[k] for k in ("lengths", "speedup", "control_within_10_percent")},
                    flush=True,
                )
            rt._gated4 = current
            model.trim_memory()
            payload["runtime_before_close"] = rt.diagnostics()
        payload["runtime_after_close"] = rt.diagnostics()
        assert rt.active_bytes == rt.cache_bytes == 0
        assert payload["source_hashes"] == source_hashes()
        payload["status"] = "passed"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error_type"] = type(error).__name__
        raise
    finally:
        save_json(args.output, payload)


if __name__ == "__main__":
    main()
