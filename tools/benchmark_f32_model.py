"""Paired full-model API timings for selected versus previous F32 projections."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize

from tessery import EmbeddingModel, ModelProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file", required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.samples <= 100:
        parser.error("samples must be 3..100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "baseline_dispatch_commit": "b2fcc2fa8664518a56fee60341cd5690a22a9d8d",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": (
            "paired randomized order; 3 warmups; unprofiled full API; uncontrolled thermals"
        ),
        "results": [],
    }
    try:
        with EmbeddingModel.load(
            args.model_dir, profile=ModelProfile.from_file(args.profile_file)
        ) as model:
            assert model.descriptor.architecture == "bert_f32"
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            rt = model._backend.runtime
            selected = rt._matmul_f32

            def previous(buffers, *, rows, cols, k):
                tiled = rows >= 8 and rows % 8 == 0 and cols % 32 == 0 and k % 8 == 0 and k <= 512
                rt._dispatch(
                    "matmul_f32_tiled" if tiled else "matmul_f32",
                    buffers,
                    threads=((rows + 7) // 8) * (cols // 32) * 128
                    if tiled
                    else ((rows + 3) // 4) * cols * 32,
                    group_size=128 if tiled else 32,
                    rows=rows,
                    cols=cols,
                    k=k,
                )

            rng = np.random.default_rng(273)
            special = 2
            for lengths in (
                [3],
                [7],
                [9],
                [16],
                [32],
                [64],
                [128],
                [512],
                [128, 127, 33],
                [33] * 8,
            ):
                texts = [" token" * (n - special) for n in lengths]
                _, actual_lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert actual_lengths.tolist() == lengths
                samples = {"previous": [], "selected": []}
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(samples)):
                        rt._matmul_f32 = previous if name == "previous" else selected
                        started = time.perf_counter()
                        outputs[name] = model.encode(texts)
                        if iteration >= 3:
                            samples[name].append(time.perf_counter() - started)
                    np.testing.assert_allclose(
                        outputs["selected"], outputs["previous"], atol=5e-6, rtol=1e-4
                    )
                    difference = max(
                        difference, float(np.max(np.abs(outputs["selected"] - outputs["previous"])))
                    )
                row = {
                    "lengths": lengths,
                    "max_abs_error": difference,
                    "timings": {k: summarize(v, len(texts)) for k, v in samples.items()},
                    "speedup": float(
                        np.median(samples["previous"]) / np.median(samples["selected"])
                    ),
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print({k: v for k, v in row.items() if k != "timings"}, flush=True)
            rt._matmul_f32 = selected
            payload["runtime"] = rt.diagnostics()
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
