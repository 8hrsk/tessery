"""Fail-closed allowlist gate for every locked candidate/dev dependency.

License declarations are reviewed inputs, not inferred from arbitrary wheel text.
Adding/changing a package requires updating the reviewed record and notices.
"""

import argparse
import ast
import importlib.metadata
import json
import platform
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    lock = tomllib.loads((root / "uv.lock").read_text())
    project = tomllib.loads((root / "pyproject.toml").read_text())
    policy = json.loads((root / "policy/dependency-licenses.json").read_text())
    allowed = set(policy["allowed_license_expressions"])
    seen = set()
    for package in lock["package"]:
        name, version = package["name"], package["version"]
        if name in seen or name not in policy["packages"]:
            raise ValueError("unreviewed_dependency")
        seen.add(name)
        record = policy["packages"][name]
        if record["version"] != version or record["license"] not in allowed:
            raise ValueError("unreviewed_license_or_version")
        if name == project["project"]["name"]:
            continue
        if package["source"] != {"registry": "https://pypi.org/simple"}:
            raise ValueError("unreviewed_dependency_source")
        artifacts = package.get("wheels", []) + [package.get("sdist", {})]
        for artifact in artifacts:
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", artifact.get("hash", "")):
                raise ValueError("dependency_hash_missing")
    if project["project"].get("dependencies") != ["numpy==2.5.2", "regex==2025.9.18"]:
        raise ValueError("unreviewed_runtime_dependencies")
    if project["build-system"]["requires"] != ["setuptools==80.9.0"]:
        raise ValueError("unreviewed_build_backend")
    for path in (root / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            if any(
                m.split(".")[0]
                in {
                    "mlx",
                    "mlx_embeddings",
                    "pickle",
                    "torch",
                    "transformers",
                    "huggingface_hub",
                    "tokenizers",
                }
                for m in modules
            ):
                raise ValueError("prohibited_production_import")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    check()
    if args.installed:
        for name in ("mlx", "mlx-embeddings", "torch", "transformers", "huggingface-hub"):
            try:
                importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            raise SystemExit("prohibited_installed_framework")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise SystemExit("installed_runtime_scan_requires_target_macos_arm64")
        numpy = importlib.metadata.distribution("numpy")
        for item in numpy.files or []:
            if str(item).endswith((".so", ".dylib")):
                path = numpy.locate_file(item)
                linked = subprocess.check_output(["/usr/bin/otool", "-L", str(path)], text=True)
                if any(name in linked.lower() for name in ("libgfortran", "libgcc", "libquadmath")):
                    raise SystemExit("unapproved_bundled_runtime")
    print("dependency_policy_verified")
