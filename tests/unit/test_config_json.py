import json
import random

import pytest

from yuri_mlx_embeddings import LoadOptions
from yuri_mlx_embeddings import strict_json as codec
from yuri_mlx_embeddings.descriptors import ENGINE_ID, PROFILES, YURI_V1
from yuri_mlx_embeddings.errors import (
    ConfigurationError,
    InvalidInputError,
    UnsupportedProfileError,
)


def test_immutable_profile():
    assert LoadOptions().profile is YURI_V1
    assert YURI_V1.compatibility_id == ENGINE_ID
    with pytest.raises(TypeError):
        PROFILES["custom"] = YURI_V1
    with pytest.raises(AttributeError):
        YURI_V1.max_length = 513


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dimensions": 0},
        {"dimensions": 1024},
        {"dimensions": True},
        {"dimensions": 384.0},
        {"max_length": 511},
        {"max_length": 513},
        {"max_length": True},
        {"max_length": 512.0},
        {"revision": "main"},
    ],
)
def test_options_reject_profile_changes(kwargs):
    with pytest.raises(ConfigurationError, match="^configuration_error$"):
        LoadOptions(**kwargs)


@pytest.mark.parametrize("model", ["other", [], None])
def test_unknown_model(model):
    with pytest.raises(UnsupportedProfileError):
        LoadOptions(model=model)


@pytest.mark.parametrize(
    "body",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"x":1,"x":2}}',
        b"{}{}",
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1e999",
        b"-1e999",
        b"\xff",
        b"\xef\xbb\xbf{}",
        b"[" * 2000,
        b"not-json",
        b"1" * 10000,
    ],
)
def test_strict_json(body):
    with pytest.raises(InvalidInputError, match="^invalid_input$"):
        codec.loads(body, limit=20000)


def test_limits_and_serialization_errors():
    assert codec.loads(b"{}", limit=2) == {}
    with pytest.raises(InvalidInputError):
        codec.loads(b"{}", limit=1)
    for obj in [float("nan"), float("inf"), object(), "\ud800"]:
        with pytest.raises(InvalidInputError):
            codec.dumps(obj, limit=100)
    recursive = []
    recursive.append(recursive)
    with pytest.raises(InvalidInputError):
        codec.dumps(recursive, limit=100)
    with pytest.raises(InvalidInputError):
        codec.dumps({"x": 1}, limit=1)


def test_seeded_unicode_roundtrip_property():
    rng = random.Random(17)
    alphabet = ["a", "я", "\u0301", "🌍", "\n", "\x00", "漢", "\U0010ffff"]
    for _ in range(500):
        payload = {"text": "".join(rng.choices(alphabet, k=rng.randrange(100)))}
        encoded = codec.dumps(payload, limit=2000)
        assert codec.loads(encoded, limit=2000) == payload
        assert json.loads(encoded) == payload
