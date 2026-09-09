"""Paired full-model API timings for selected versus previous quantized projections."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes, summarize

from tessery import EmbeddingModel


def previous(rt, buffers, *, rows, cols, k):
    # Measured Qwen projection shapes; incomplete 16-row tiles and other
    # projection shapes retain the full-tile / small-tail dispatch below.
    if (
        rows >= 16
        and rows % 16 == 0
        and (cols, k) in ((1024, 1024), (2048, 1024), (3072, 1024), (1024, 2048), (1024, 3072))
    ):
        rt._dispatch(
            "linear4_16x32_k64",
            buffers,
            threads=(rows // 16) * (cols // 32) * 256,
            group_size=256,
            rows=rows,
            cols=cols,
            k=k,
        )
        return
    if rows >= 5 and cols % 32 == 0 and k % 64 == 0:
        complete = rows // 8
        if complete:
            rt._dispatch(
                "linear4_tiled",
                buffers,
                threads=complete * (cols // 32) * 128,
                group_size=128,
                rows=rows,
                cols=cols,
                k=k,
            )
        if rows % 8:
            # n is the starting row for the single partial row tile.
            small = rows % 8 <= 4
            rt._dispatch(
                "linear4" if small else "linear4_tail",
                buffers,
                threads=cols * 32 if small else (cols // 32) * 128,
                group_size=32 if small else 128,
                n=complete * 8,
                rows=rows,
                cols=cols,
                k=k,
            )
    else:
        rt._dispatch(
            "linear4",
            buffers,
            threads=((rows + 3) // 4) * cols * 32,
            group_size=32,
            rows=rows,
            cols=cols,
            k=k,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
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
        "baseline_dispatch_commit": "7673c19e9f483b5bad6426f3e8258e958529b2af",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditions": (
            "paired randomized order; 3 warmups; unprofiled full API; uncontrolled thermals"
        ),
        "results": [],
    }
    try:
        with EmbeddingModel.load(args.model_dir) as model:
            payload["model"] = model.descriptor.model_id
            payload["compatibility_id"] = model.descriptor.compatibility_id
            rt = model._backend.runtime
            selected = rt._linear4

            rng = np.random.default_rng(273)
            special = 1
            for lengths in ([7], [16], [17], [33], [65], [128], [129], [257], [33] * 8):
                texts = [" token" * (n - special) for n in lengths]
                _, actual_lengths = model._tokenizer.batch(texts, max_length=model.max_length)
                assert actual_lengths.tolist() == lengths
                samples = {"previous": [], "selected": []}
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(samples)):
                        rt._linear4 = (
                            (lambda buffers, **kw: previous(rt, buffers, **kw))
                            if name == "previous"
                            else selected
                        )
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
            rt._linear4 = selected
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
