import importlib.util
import io
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "tools/verify_release.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("fault", [None, "portable", "missing_native", "extra_file"])
def test_release_gate_rejects_incomplete_or_portable_payload(tmp_path, fault):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    tag = "any" if fault == "portable" else "macosx_14_0_arm64"
    with zipfile.ZipFile(tmp_path / f"tessery-{version}-py3-none-{tag}.whl", "w") as archive:
        for name in ["tessery/__init__.py", "metal_inference/native/kernels.metal"]:
            archive.writestr(name, "")
        if fault != "missing_native":
            archive.writestr("metal_inference/_native.dylib", b"layout fixture only")
        archive.writestr(
            f"tessery-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: tessery\nVersion: {version}\n",
        )
    with tarfile.open(tmp_path / f"tessery-{version}.tar.gz", "w:gz") as archive:
        archive.addfile(
            tarfile.TarInfo(f"tessery-{version}/src/metal_inference/native/runtime.mm"),
            io.BytesIO(),
        )
    (tmp_path / ".gitignore").write_text("*")
    if fault == "extra_file":
        (tmp_path / "stale.whl").write_bytes(b"stale")
    if fault is None:
        MODULE.verify(tmp_path)
    else:
        with pytest.raises(AssertionError):
            MODULE.verify(tmp_path)
