"""One sampler, four fresh processes: baseline/candidate/candidate/baseline."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from diagnose_metal import save_json

# The same sampler is used for both source trees. Override only its source-root
# provenance before importing it; verify the imported runtime comes from that tree.
WORKER = """
import os, runpy, sys
from pathlib import Path
source = Path(os.environ['TESSERY_SOURCE_ROOT']).resolve()
harness = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(harness.parent))
import metal_inference, tessery
assert Path(tessery.__file__).resolve().parent == source/'src/tessery'
assert Path(metal_inference.__file__).resolve().parent == source/'src/metal_inference'
import diagnose_metal
diagnose_metal.ROOT = source
sys.argv = sys.argv[1:]
runpy.run_path(str(harness), run_name='__main__')
"""


def file_hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def source_files(root):
    return sorted(
        p
        for p in (root / "src").rglob("*")
        if p.suffix in {".py", ".mm", ".metal", ".dylib", ".so"}
    )


def compare(workers):
    assert len(workers) == 4
    indexed = [{tuple(row["lengths"]): row for row in w["results"]} for w in workers]
    assert all(len(rows) == len(w["results"]) for rows, w in zip(indexed, workers, strict=True))
    assert all(set(rows) == set(indexed[0]) for rows in indexed)
    for a, b in [(0, 3), (1, 2)]:
        assert workers[a]["source_hashes"] == workers[b]["source_hashes"]
    for field in [
        "python",
        "numpy",
        "regex_version",
        "compatibility_id",
        "profile_sha256",
        "timing_scope",
    ]:
        assert all(w[field] == workers[0][field] for w in workers)
    results = []
    for case in indexed[0]:
        rows = [w[case] for w in indexed]
        for field in ["input_ids_sha256", "plans"]:
            assert all(r[field] == rows[0][field] for r in rows)
        for row in rows[1:]:
            np.testing.assert_array_equal(row["vectors"], rows[0]["vectors"])
        medians = [r["p50_seconds"] for r in rows]
        drift = [
            max(medians[a], medians[b]) / min(medians[a], medians[b]) for a, b in [(0, 3), (1, 2)]
        ]
        results.append(
            {
                "lengths": list(case),
                "exact_vectors": True,
                "baseline_over_candidate": [medians[0] / medians[1], medians[3] / medians[2]],
                "p50_seconds": medians,
                "p95_seconds": [r["p95_seconds"] for r in rows],
                "baseline_candidate_drift": drift,
                "control_a_over_b": [r["identical_control_a_over_b"] for r in rows],
                "timing_screen_passed": all(r["control_within_10_percent"] for r in rows)
                and max(drift) <= 1.15,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--profile-file")
    parser.add_argument("--lengths", nargs="+", type=int, default=[3, 7, 24, 160, 512])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "summary.json"
    harness = Path(__file__).with_name("benchmark_mlx_isolated.py").resolve()
    roots = [args.baseline_root.resolve(), args.candidate_root.resolve()]
    helpers = [harness, harness.with_name("diagnose_metal.py"), Path(__file__).resolve()]
    helper_hashes = file_hashes(helpers)
    tree_hashes = [file_hashes(source_files(root)) for root in roots]

    def verify_snapshot():
        assert file_hashes(helpers) == helper_hashes, "Benchmark helper changed"
        assert [file_hashes(source_files(root)) for root in roots] == tree_hashes, (
            "Source tree changed"
        )

    report = {
        "status": "running",
        "sampler_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_order": ["baseline", "candidate", "candidate", "baseline"],
        "roots": [str(p) for p in roots],
        "helper_hashes": helper_hashes,
        "source_tree_hashes": tree_hashes,
        "workers": [],
    }
    save_json(output, report)
    try:
        for i, root in enumerate([roots[0], roots[1], roots[1], roots[0]]):
            verify_snapshot()
            path = (args.output_dir / f"worker-{i}.json").resolve()
            command = [
                sys.executable,
                "-c",
                WORKER,
                str(harness),
                "--engine",
                "tessery",
                "--model-dir",
                args.model_dir,
                "--lengths",
                *[str(n) for n in args.lengths],
                "--paired-controls",
                "--include-batches",
                "--samples",
                str(args.samples),
                "--output",
                str(path),
            ]
            if args.profile_file:
                command.extend(["--profile-file", str(Path(args.profile_file).resolve())])
            if i >= 2:
                command.append("--reverse-cases")
            env = {
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "TESSERY_SOURCE_ROOT": str(root),
                "VECLIB_MAXIMUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            with (args.output_dir / f"worker-{i}.log").open("x") as stream:
                subprocess.run(
                    command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
            verify_snapshot()
            report["workers"].append(json.loads(path.read_text()))
            save_json(output, report)
            print(f"worker {i} completed", flush=True)
        verify_snapshot()
        report["results"] = compare(report["workers"])
        report["status"] = "passed"
        report["meaning"] = (
            "Identity and exact vectors passed; timing_screen_passed is per case, "
            "not a confidence interval."
        )
        for row in report["results"]:
            print(row, flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        raise
    finally:
        save_json(output, report)


if __name__ == "__main__":
    main()
