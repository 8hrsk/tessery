# First PyPI release: tessery 0.5.1a1

[Public release](https://pypi.org/project/tessery/0.5.1a1/).

Source commit: `fb5568ed52af01430d1b5910ab92e8e55de59e56`. The tested artifacts were uploaded unchanged.

| Artifact | SHA-256 |
| --- | --- |
| `tessery-0.5.1a1-py3-none-macosx_14_0_arm64.whl` | `e9706be717cc73759097e7ec4599fa98a196ee2c541609282fee8b486a527464` |
| `tessery-0.5.1a1.tar.gz` | `fa1909f3aa190fc17b5021313ff8f50e5c42146a71f61efdf26c10f531aa52a7` |

PyPI-reported SHA-256 values matched the local wheel/source artifacts and their
SPDX sidecars. The wheel is macOS 14+ arm64 with a ctypes native Metal bridge;
Python 3.12+ is required. Model weights and Python itself are not bundled.

Verification after publication:

* Uninstalled the locally installed Tessery wheel from the isolated smoke environment.
* Installed `tessery==0.5.1a1` using `python -m pip install` with the public PyPI index
  and binary distributions only; pip downloaded the native wheel from PyPI.
* Reused the already installed pinned NumPy/regex dependencies; `pip check` passed.
* Exercised public imports, embeddings, workspace reuse/trim/close, SQLite retrieval
  and HTTP equivalence on both existing Qwen and BGE packs.
* External connections and DNS were blocked during model smoke checks; only loopback
  HTTP was allowed. No model downloads occurred.

The engine remains an alpha. This publication does not change earlier numerical,
platform, server or lifecycle limits. Future artifact changes require a new version.
Publication was manual with the user-provided environment token; no credentials
were committed and no automatic publishing workflow was enabled.
