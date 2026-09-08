# Multi-hour portable qualification on Kaggle

Upload [kaggle-portable-soak.ipynb](../notebooks/kaggle-portable-soak.ipynb) to Kaggle.
Choose CPU, enable Internet for setup, and run the cells in order. No model weights
are downloaded. The setup creates an isolated Python 3.12.13 environment with the
locked development dependencies; it does not replace the notebook's Python packages
apart from installing the pinned uv bootstrap tool.

The setup checks out GitHub `main` once and prints/saves its resolved commit. For
reproduction set `REF` to an existing reviewed commit or tag. The soak records a
SHA-256 of its source inputs and fails if these change during the run. Do not edit
or refresh the checkout while a run is active. On rerun choose a new output name:
existing reports are never overwritten by a new run.

The default workload duration is four hours, configurable up to 24 hours; the
service's own session limits still apply. It exercises actual BPE and WordPiece
cancellation, repeated API calls through the executor, deterministic synthetic
vectors, exact retrieval, SQLite snapshot save/load and cleanup. Each iteration
uses bounded objects; temporary SQLite files are removed and reports are atomically
checkpointed every 30 seconds. Interrupted or failed processes retain a partial
report and cannot produce a `passed` status unless the requested workload finished.
A forcibly killed process may leave status `running`; that is an incomplete result.

Download these Kaggle output files after the run:

- `/kaggle/working/tessery-soak.json`
- `/kaggle/working/tessery-revision.txt`

Acceptance: successful portable pytest suite, `status=passed`, positive iterations,
`workload_seconds >= requested_seconds`, unchanged source fingerprint and retained
Python allocation growth below the configured 32 MiB budget. Linux current RSS is
recorded separately. This is not a proof against every native allocation leak.

The synthetic backend has no neural model and performs no Metal operations.
Kaggle's [official runtime images](https://github.com/Kaggle/docker-python) are
CPU/Linux or NVIDIA GPU containers. They cannot execute Apple's Metal framework.
Neither `pip install tessery` nor enabling a Kaggle GPU changes that platform limit.
The portable build flag is only for tests and must never be used for publishing
an inference wheel.

For a bounded local rehearsal from this checkout:

```sh
uv run --frozen python tools/soak_portable.py \
  --seconds 60 --output artifacts/portable-rehearsal.json
```

Native qualification remains separately available through `tools/diagnose_metal.py`
on Apple Silicon. Its reports include real model correctness, GPU buffer/cache
accounting, RSS, concurrent cancellation and recovery. Multi-hour native execution
was not started as part of preparing this notebook.
