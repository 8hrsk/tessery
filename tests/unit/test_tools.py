import hashlib
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "namespace,expected",
    [
        ("yuri_mlx_embeddings", 0),
        ("mlx_embeddings", 1),
        ("benchmarks", 1),
    ],
)
def test_sbom_namespace_and_artifact_hash(tmp_path, namespace, expected):
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{namespace}/__init__.py", "")
        archive.writestr("test.dist-info/licenses/LICENSE", "Apache-2.0")
    output = tmp_path / "sbom.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/sbom.py"),
            str(wheel),
            "--epoch",
            "1788566400",
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == expected
    if expected == 0:
        doc = json.loads(output.read_text())
        assert doc["spdxVersion"] == "SPDX-2.3"
        assert (
            doc["packages"][0]["checksums"][0]["checksumValue"]
            == hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
    else:
        assert not output.exists()


def test_dependency_gate_rejects_new_unreviewed_package(tmp_path):
    checker = runpy.run_path(str(ROOT / "tools/check_dependencies.py"))["check"]
    (tmp_path / "policy").mkdir()
    for name in ["uv.lock", "pyproject.toml", "policy/dependency-licenses.json"]:
        (tmp_path / name).write_bytes((ROOT / name).read_bytes())
    with (tmp_path / "uv.lock").open("a") as stream:
        stream.write('\n[[package]]\nname = "mlx-embeddings"\nversion = "0.1.0"\n')
    with pytest.raises(ValueError, match="unreviewed_dependency"):
        checker(tmp_path)
