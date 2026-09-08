import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from metal_inference import cli
from metal_inference.api import MemoryStats
from metal_inference.errors import MetalUnavailableError


class FakeModel:
    descriptor = SimpleNamespace(compatibility_id="test-engine")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def encode(self, texts):
        return np.full((len(texts), 384), 1 / np.sqrt(384), np.float32)

    def memory_stats(self):
        return MemoryStats(100, 200)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(cli.EmbeddingModel, "load", lambda *a, **k: FakeModel())


def test_stdin_and_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'["hello"]')))
    assert cli.main(["embed", "--model-dir", "/model"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["embeddings"][0]) == 384
    source = tmp_path / "input.json"
    source.write_text('["hello"]')
    output = tmp_path / "out.json"
    args = ["embed", "--model-dir", "/model", "--input", str(source), "--output", str(output)]
    assert cli.main(args) == 0
    assert json.loads(output.read_bytes()) == payload
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "io_error"


@pytest.mark.parametrize("data", [b"[] trailing", b"null", b'{"input":[]}', b" " * 2097153])
def test_safe_bad_wire(data, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    assert cli.main(["embed", "--model-dir", "/model"]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": {"code": "invalid_input"}}


def test_inspect(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "read_artifact", lambda *args: calls.append(args))
    assert cli.main(["inspect", "--model-dir", "/model"]) == 0
    assert len(calls) == 3
    assert json.loads(capsys.readouterr().out)["verified"]


def test_benchmark_and_limits(capsys):
    args = ["benchmark", "--model-dir", "/model", "--iterations", "2", "--warmup", "1"]
    assert cli.main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["latency_seconds"]) == 2
    assert data["p99_seconds"] >= data["p50_seconds"]
    assert cli.main(args + ["--iterations", "0"]) == 2
    assert "invalid_input" in capsys.readouterr().err


def test_model_error(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise MetalUnavailableError()

    monkeypatch.setattr(cli.EmbeddingModel, "load", fail)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'["text"]')))
    assert cli.main(["embed", "--model-dir", "/model"]) == 2
    assert json.loads(capsys.readouterr().err) == {"error": {"code": "metal_unavailable"}}
