"""Development-only immutable shader and routing controls for model diagnostics."""

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch

from diagnose_metal import ROOT

from tessery import MetalRuntime

SHADER = ROOT / "src/metal_inference/native/kernels.metal"
RELATIVE_SHADER = "src/metal_inference/native/kernels.metal"


def revision_shader(revision):
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", revision + "^{commit}"],
        cwd=ROOT,
        text=True,
    ).strip()
    source = subprocess.check_output(["git", "show", f"{resolved}:{RELATIVE_SHADER}"], cwd=ROOT)
    return resolved, source


def runtime_with_shader(source, **kwargs):
    read_bytes = Path.read_bytes

    def read(path):
        return source if path == SHADER else read_bytes(path)

    # Only runtime construction runs under the exact-path substitution. No file is edited.
    with patch.object(Path, "read_bytes", read):
        runtime = MetalRuntime(**kwargs)
    assert runtime.diagnostics()["shader_sha256"] == hashlib.sha256(source).hexdigest()
    return runtime


def force_old_linear(runtime):
    original = runtime._dispatch

    def dispatch(kernel, buffers, **params):
        if kernel == "linear4_32x32_k64":
            assert params["rows"] in (128, 160, 256, 512)
            assert (params["cols"], params["k"]) in (
                (1024, 1024),
                (2048, 1024),
                (3072, 1024),
                (1024, 2048),
                (1024, 3072),
            )
            assert params["group_size"] == 256
            kernel = "linear4_16x32_k64"
            params["threads"] *= 2
        return original(kernel, buffers, **params)

    assert runtime.diagnostics()["plan_builds"] == 0
    runtime._dispatch = dispatch
