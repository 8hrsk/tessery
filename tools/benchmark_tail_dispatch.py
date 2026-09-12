"""Paired normal-path full-model comparison against the 0.6 tail dispatcher."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from diagnose_metal import save_json, source_hashes

from tessery import EmbeddingModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.samples <= 100:
        parser.error("samples must be 5..100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write('{"status":"starting"}\n')
    payload = {
        "status": "running",
        "source_hashes": source_hashes(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": [],
        "conditions": (
            "randomized paired order; three warmups; one process/model; "
            "normal encoder boundaries; uncontrolled thermals"
        ),
    }
    try:
        rng = np.random.default_rng(77)
        with EmbeddingModel.load(args.model_dir) as model:
            rt = model._backend.runtime
            rt._plans_enabled = False  # These experiments mutate kernel routing between calls.
            current = rt._linear4

            def previous(buffers, *, rows, cols, k):
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
                        rt._dispatch(
                            "linear4_tail",
                            buffers,
                            threads=(cols // 32) * 128,
                            group_size=128,
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

            for n in (2, 7, 8, 9, 10, 11, 12, 17, 31, 32, 33):
                texts = [" token" * (n - 1)]
                samples = {"previous": [], "selected": []}
                difference = 0.0
                for iteration in range(args.samples + 3):
                    outputs = {}
                    for name in rng.permutation(list(samples)):
                        rt._linear4 = previous if name == "previous" else current
                        start = time.perf_counter()
                        outputs[name] = model.encode(texts)
                        if iteration >= 3:
                            samples[name].append(time.perf_counter() - start)
                    np.testing.assert_allclose(
                        outputs["selected"], outputs["previous"], atol=5e-6, rtol=1e-4
                    )
                    difference = max(
                        difference, float(np.max(np.abs(outputs["selected"] - outputs["previous"])))
                    )
                row = {
                    "tokens": n,
                    "samples": samples,
                    "speedup": float(
                        np.median(samples["previous"]) / np.median(samples["selected"])
                    ),
                    "max_abs_error": difference,
                }
                payload["results"].append(row)
                save_json(args.output, payload)
                print({k: v for k, v in row.items() if k != "samples"}, flush=True)
            rt._linear4 = current
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
