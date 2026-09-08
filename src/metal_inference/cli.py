"""Standalone command line interface. No daemon or Go supervisor is required."""

import argparse
import json
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .api import EmbeddingModel
from .errors import EmbeddingError, InvalidInputError
from .json_codec import dumps, loads
from .profiles import ModelProfile, get_profile, list_profiles
from .weights import read_artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metal-inference")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles", help="List built-in model profiles")
    for name in ("inspect", "embed", "benchmark"):
        command = sub.add_parser(name)
        command.add_argument("--model-dir", required=True)
        profile_args = command.add_mutually_exclusive_group()
        profile_args.add_argument(
            "--profile", default="qwen3-embedding-0.6b-dwq", choices=list_profiles()
        )
        profile_args.add_argument("--profile-file", help="Caller-trusted data-only model manifest")
        if name != "inspect":
            command.add_argument("--dimensions", type=int)
            command.add_argument("--max-length", type=int)
        if name == "embed":
            command.add_argument("--input", default="-", help="JSON array file; - reads stdin")
            command.add_argument("--output", default="-", help="JSON result file; - writes stdout")
        if name == "benchmark":
            command.add_argument("--batch-size", type=int, choices=(1, 4, 16, 32), default=1)
            command.add_argument("--tokens", type=int, choices=(32, 128, 512), default=32)
            command.add_argument("--iterations", type=int, default=10)
            command.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "profiles":
            print(
                dumps(
                    {
                        "profiles": [
                            {"name": name, **get_profile(name).to_dict()}
                            for name in list_profiles()
                        ]
                    },
                    limit=65536,
                ).decode()
            )
            return 0
        profile = (
            ModelProfile.from_file(args.profile_file)
            if args.profile_file
            else get_profile(args.profile)
        )
        if args.command == "inspect":
            for artifact in profile.artifacts:
                read_artifact(args.model_dir, artifact.name, profile=profile)
            payload: object = {
                "model": profile.model_id,
                "revision": profile.revision,
                "verified": True,
                "backend": "native_metal",
                "profile": profile.to_dict(),
                "compatibility_id": profile.compatibility_id,
            }
        else:
            args.dimensions = (
                profile.default_dimensions if args.dimensions is None else args.dimensions
            )
            args.max_length = profile.max_length if args.max_length is None else args.max_length
            if args.command == "benchmark" and not (
                1 <= args.iterations <= 1000
                and 1 <= args.warmup <= 20
                and args.max_length >= args.tokens
            ):
                raise InvalidInputError()
            if args.command == "embed":
                if args.input == "-":
                    raw = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
                else:
                    with Path(args.input).open("rb") as stream:
                        raw = stream.read(2 * 1024 * 1024 + 1)
                texts = loads(raw, limit=2 * 1024 * 1024)
                if not isinstance(texts, list):
                    raise InvalidInputError()
            started = time.perf_counter()
            with EmbeddingModel.load(
                args.model_dir,
                dimensions=args.dimensions,
                max_length=args.max_length,
                profile=profile,
            ) as model:
                load_seconds = time.perf_counter() - started
                if args.command == "embed":
                    vectors = model.encode(texts)
                    payload = {
                        "model": model.descriptor.model_id,
                        "compatibility_id": model.descriptor.compatibility_id,
                        "dimensions": args.dimensions,
                        "embeddings": vectors.tolist(),
                    }
                else:
                    batch = [" token" * (args.tokens - profile.min_length)] * args.batch_size
                    started = time.perf_counter()
                    model.encode(batch)
                    first = time.perf_counter() - started
                    for _ in range(args.warmup):
                        model.encode(batch)
                    samples = []
                    for _ in range(args.iterations):
                        started = time.perf_counter()
                        model.encode(batch)
                        samples.append(time.perf_counter() - started)
                    percentiles = np.percentile(samples, [50, 95, 99]).tolist()
                    payload = {
                        "scope": "single_host_diagnostic_not_release_gate",
                        "backend": "native_metal",
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "model": model.descriptor.model_id,
                        "compatibility_id": model.descriptor.compatibility_id,
                        "model_load_seconds": load_seconds,
                        "first_encode_seconds": first,
                        "warmup_count": args.warmup,
                        "batch_size": args.batch_size,
                        "tokens_per_text": args.tokens,
                        "latency_seconds": samples,
                        "p50_seconds": percentiles[0],
                        "p95_seconds": percentiles[1],
                        "p99_seconds": percentiles[2],
                        "texts_per_second": args.batch_size * len(samples) / sum(samples),
                        "memory": asdict(model.memory_stats()),
                    }
        encoded = dumps(payload, limit=16 * 1024 * 1024)
        if args.command == "embed" and args.output != "-":
            with Path(args.output).open("xb") as output:
                output.write(encoded + b"\n")
        else:
            print(encoded.decode("utf-8"))
        return 0
    except EmbeddingError as error:
        print(json.dumps({"error": {"code": error.code}}), file=sys.stderr)
        return 2
    except OSError:
        print('{"error":{"code":"io_error"}}', file=sys.stderr)
        return 2
