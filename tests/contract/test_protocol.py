import json
import math
from pathlib import Path

import pytest

from yuri_mlx_embeddings.descriptors import MODEL_ID
from yuri_mlx_embeddings.errors import InferenceError, InvalidInputError, OverloadError
from yuri_mlx_embeddings.server import protocol_v1 as protocol


def request(**changes):
    data = {"model": MODEL_ID, "dimensions": 384, "input": ["текст"]}
    data.update(changes)
    return json.dumps(data).encode()


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "other"},
        {"model": None},
        {"dimensions": 383},
        {"dimensions": True},
        {"dimensions": 384.0},
        {"input": []},
        {"input": "text"},
        {"input": [" "]},
        {"input": [""]},
        {"input": [None]},
        {"input": [5]},
        {"input": [True]},
        {"input": ["text"] * 33},
        {"input": ["\ud800"]},
        {"other": 1},
        {"input": ["x" * (protocol.INPUT_LIMIT + 1)]},
        {"input": ["я" * (protocol.INPUT_LIMIT // 2 + 1)]},
        {"input": ["x" * (protocol.INPUT_LIMIT // 2), "x" * (protocol.INPUT_LIMIT // 2 + 1)]},
    ],
)
def test_invalid(changes):
    with pytest.raises(InvalidInputError):
        protocol.decode_request(request(**changes))


@pytest.mark.parametrize("body", [b"{}", b"[]", b"null", b"1", b"{} trailing"])
def test_invalid_shape(body):
    with pytest.raises(InvalidInputError):
        protocol.decode_request(body)


def test_python_empty_and_input_boundaries():
    assert protocol.validate_texts([], allow_empty=True) == ()
    assert protocol.validate_texts(("hello",)) == ("hello",)
    for value in ["text", {}, None]:
        with pytest.raises(InvalidInputError):
            protocol.validate_texts(value)
    assert protocol.decode_request(request(input=["x" * protocol.INPUT_LIMIT]))
    with pytest.raises(InvalidInputError):
        protocol.decode_request(b" " * (protocol.BODY_LIMIT + 1))


@pytest.mark.parametrize("batch", [1, 16, 32])
def test_shape_order_norm(batch):
    decoded = protocol.decode_request(request(input=[f"text {i}" for i in range(batch)]))
    vectors = [[0.0] * i + [1.0] + [0.0] * (383 - i) for i in range(batch)]
    result = json.loads(protocol.encode_response(decoded, vectors))
    assert result["model"] == MODEL_ID
    assert result["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
    assert [row["index"] for row in result["data"]] == list(range(batch))
    assert [row["embedding"] for row in result["data"]] == vectors


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        3.5e38,
        True,
        "1",
        b"1",
        None,
        object(),
        10**1000,
    ],
)
def test_invalid_vectors(value):
    decoded = protocol.decode_request(request())
    with pytest.raises(InferenceError, match="^inference_failed$"):
        protocol.encode_response(decoded, [[value] + [0.0] * 383])


def test_invalid_response_shape_norm_and_limit(monkeypatch):
    decoded = protocol.decode_request(request())
    for vectors in [[], [[0.0] * 383], [[0.0] * 384], [[2.0] + [0.0] * 383]]:
        with pytest.raises(InferenceError):
            protocol.encode_response(decoded, vectors)
    monkeypatch.setattr(protocol, "RESPONSE_LIMIT", 1)
    with pytest.raises(InferenceError):
        protocol.encode_response(decoded, [[1.0] + [0.0] * 383])


def test_safe_error():
    status, body = protocol.encode_error(OverloadError())
    assert status == 429
    assert json.loads(body) == {"error": {"code": "overloaded"}}


def test_full_draft_fixture():
    # Hand-authored unit vector; never a golden model vector or Go canonical fixture.
    path = Path(__file__).parent / "fixtures" / "draft-success.json"
    fixture = path.read_bytes()
    parsed = json.loads(fixture)
    assert len(parsed["data"][0]["embedding"]) == 384
    decoded = protocol.decode_request(request())
    assert json.loads(protocol.encode_response(decoded, [[1.0] + [0.0] * 383])) == parsed
