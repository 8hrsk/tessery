import hashlib
import json

import pytest

from yuri_mlx_embeddings.cli import main


@pytest.mark.parametrize("command", ["serve", "benchmark"])
def test_explicit_prerequisite(command, capsys):
    assert main([command]) == 3
    output = capsys.readouterr()
    assert output.err == "baseline_required\n"
    assert output.out == ""


def test_manifest_failure(tmp_path, capsys):
    assert (
        main(
            [
                "validate-model",
                "--model-dir",
                str(tmp_path),
                "--manifest",
                str(tmp_path / "secret"),
                "--manifest-sha256",
                "a" * 64,
            ]
        )
        == 2
    )
    assert capsys.readouterr().err == "invalid_manifest\n"


def test_validate_success(tmp_path, monkeypatch, capsys):
    from yuri_mlx_embeddings import cli

    data = json.dumps({"test": 1}).encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(data)
    sentinel = object()
    monkeypatch.setattr(cli, "parse_manifest", lambda *a, **k: sentinel)
    calls = []
    monkeypatch.setattr(cli, "verify_model", lambda *args: calls.append(args))
    assert (
        main(
            [
                "validate-model",
                "--model-dir",
                str(tmp_path),
                "--manifest",
                str(path),
                "--manifest-sha256",
                hashlib.sha256(data).hexdigest(),
            ]
        )
        == 0
    )
    assert calls == [(str(tmp_path), sentinel)]
    assert capsys.readouterr().out == "model_verified\n"
