"""Validate repeat identity and screen timing drift in isolated full-model runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_mlp_isolated import stability_summary
from diagnose_metal import save_json


def compare(first, second):
    assert first["status"] == second["status"] == "passed"
    assert {first["engine_order"], second["engine_order"]} == {"tessery-first", "mlx-first"}
    assert first["paired_controls"] and second["paired_controls"]
    for key in (
        "harness_sha256",
        "reference_sha256",
        "profile_sha256",
        "mlx_mask",
        "requested_lengths",
        "include_batches",
        "samples_per_label",
    ):
        assert first[key] == second[key]
    workers = [
        first["workers"]["tessery"],
        first["workers"]["mlx"],
        second["workers"]["mlx"],
        second["workers"]["tessery"],
    ]
    assert all(w["source_hashes"] == workers[0]["source_hashes"] for w in workers)
    indexed = [{str(r["case"]): r for r in w["results"]} for w in workers]
    assert all(len(rows) == len(w["results"]) for rows, w in zip(indexed, workers, strict=True))
    keys = set(indexed[0])
    assert all(set(rows) == keys for rows in indexed)
    for key in keys:
        rows = [w[key] for w in indexed]
        for field in ("lengths", "input_ids_sha256", "plans"):
            assert all(row[field] == rows[0][field] for row in rows)
        for row in rows:
            np.testing.assert_allclose(row["vectors"], rows[0]["vectors"], atol=5e-6, rtol=1e-4)
    for engine in ("tessery", "mlx"):
        for field in ("python", "numpy", "regex_version", "compatibility_id"):
            assert first["workers"][engine][field] == second["workers"][engine][field]
    assert first["workers"]["mlx"]["mlx_version"] == second["workers"]["mlx"]["mlx_version"]
    normalized = [
        {
            "results": {
                key: {
                    "timings": {"p50_seconds": row["p50_seconds"]},
                    "control_within_10_percent": row["control_within_10_percent"],
                }
                for key, row in rows.items()
            }
        }
        for rows in indexed
    ]
    result = stability_summary(normalized)
    for row in result:
        row["lengths"] = indexed[0][row["case"]]["lengths"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = [json.loads(path.read_text()) for path in args.runs]
    rows = compare(*data)
    payload = {
        "status": "passed",
        "meaning": "Identity/numerical checks passed; timing_screen_passed is separate per case",
        "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.runs},
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "screen_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_mlp_isolated.py").read_bytes()
        ).hexdigest(),
        "source_hashes": data[0]["workers"]["tessery"]["source_hashes"],
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write('{"status":"starting"}\n')
    save_json(args.output, payload)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
