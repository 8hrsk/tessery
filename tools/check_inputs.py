"""Check frozen local input hashes, without regenerating any fixture."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-yuri", action="store_true")
    args = parser.parse_args()
    for relative, expected in json.loads((ROOT / "policy/inputs.sha256.json").read_text()).items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit("frozen_input_digest_mismatch")
    print("local_frozen_inputs_verified")
    if args.legacy_yuri:
        raise SystemExit("legacy_vector_reuse_not_approved: independent engine uses a new space")


if __name__ == "__main__":
    main()
