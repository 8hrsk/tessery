"""Phase-1 CLI; unavailable inference commands fail explicitly and safely."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import EmbeddingError, ManifestError, PrerequisiteError
from .manifests import MANIFEST_LIMIT, parse_manifest, verify_model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yuri-mlx-embeddings")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-model", help="Verify a pinned local model pack")
    validate.add_argument("--model-dir", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--manifest-sha256", required=True)
    commands.add_parser("serve", help="Unavailable until the baseline implementation gate")
    commands.add_parser("benchmark", help="Unavailable until candidate inference exists")
    args = parser.parse_args(argv)
    try:
        if args.command != "validate-model":
            raise PrerequisiteError()
        try:
            with Path(args.manifest).open("rb") as stream:
                manifest = parse_manifest(
                    stream.read(MANIFEST_LIMIT + 1), expected_sha256=args.manifest_sha256
                )
        except OSError:
            raise ManifestError() from None
        verify_model(args.model_dir, manifest)
    except EmbeddingError as error:
        print(error.code, file=sys.stderr)
        return 3 if isinstance(error, PrerequisiteError) else 2
    print("model_verified")
    return 0
