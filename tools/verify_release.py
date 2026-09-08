"""Reject portable wheels, stale versions and accidental extra upload payloads."""

import argparse
import platform
import tarfile
import tomllib
import zipfile
from pathlib import Path


def verify(directory):
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    wheel = f"tessery-{version}-py3-none-macosx_14_0_arm64.whl"
    source = f"tessery-{version}.tar.gz"
    names = {p.name for p in directory.iterdir()}
    if ".gitignore" in names:
        assert (directory / ".gitignore").read_text().strip() == "*"
        names.remove(".gitignore")  # uv's generated build-directory marker, never uploaded.
    assert names == {wheel, source}, "Unexpected upload files"
    with zipfile.ZipFile(directory / wheel) as archive:
        names = archive.namelist()
        assert "metal_inference/_native.dylib" in names
        assert "metal_inference/native/kernels.metal" in names
        assert "tessery/__init__.py" in names
        metadata = archive.read(f"tessery-{version}.dist-info/METADATA").decode()
        assert f"\nVersion: {version}\n" in metadata
        assert "\nName: tessery\n" in metadata
        assert not any(name.endswith(".safetensors") for name in names)
    with tarfile.open(directory / source) as archive:
        assert f"tessery-{version}/src/metal_inference/native/runtime.mm" in archive.getnames()
    print({"version": version, "files": [wheel, source], "build_host": platform.platform()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
