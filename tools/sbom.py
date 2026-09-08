"""Deterministic SPDX 2.3 SBOM for a native package and declared dependencies.

This does not describe a bundled CPython/model/operating-system runtime.
"""

import argparse
import hashlib
import json
import tarfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.artifact.suffix == ".whl":
        with zipfile.ZipFile(args.artifact) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(args.artifact) as archive:
            names = archive.getnames()
    if any({"benchmarks", "mlx_embeddings"} & set(PurePosixPath(n).parts) for n in names):
        raise SystemExit("baseline_must_not_be_bundled")
    if not any(n.endswith("LICENSE") for n in names):
        raise SystemExit("artifact_license_missing")
    digest = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": args.artifact.name,
        "documentNamespace": f"https://spdx.org/spdxdocs/tessery-{digest}",
        "creationInfo": {
            "creators": ["Tool: tessery-sbom-0.5.1a1"],
            "created": datetime.fromtimestamp(args.epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "packages": [
            {
                "SPDXID": "SPDXRef-Foundation",
                "name": "tessery",
                "versionInfo": "0.5.1a1",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "Apache-2.0",
                "licenseDeclared": "Apache-2.0",
                "copyrightText": "Copyright 2026 Metal Inference contributors",
                "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
                "comment": "Own native Metal bridge/kernels; excludes CPython and model weights.",
            }
        ],
        "relationships": [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Foundation",
            }
        ],
    }
    policy = json.loads(
        (Path(__file__).resolve().parents[1] / "policy/dependency-licenses.json").read_text()
    )
    for name in ("numpy", "regex"):
        record = policy["packages"][name]
        reference = "SPDXRef-" + name
        document["packages"].append(
            {
                "SPDXID": reference,
                "name": name,
                "versionInfo": record["version"],
                "downloadLocation": f"https://pypi.org/project/{name}/{record['version']}/",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": record["license"],
                "copyrightText": "NOASSERTION",
                "comment": "Declared external dependency, not bundled into the engine wheel.",
            }
        )
        document["relationships"].append(
            {
                "spdxElementId": "SPDXRef-Foundation",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": reference,
            }
        )
    with args.out.open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
