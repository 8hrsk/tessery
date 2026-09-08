"""Strict protocol codec; canonical Go fixtures are a separate release gate."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .. import strict_json
from ..descriptors import MODEL_ID
from ..errors import EmbeddingError, InferenceError, InvalidInputError

BODY_LIMIT = 2 * 1024 * 1024
INPUT_LIMIT = 1024 * 1024
RESPONSE_LIMIT = 16 * 1024 * 1024
MAX_BATCH = 32
DIMENSIONS = 384


@dataclass(frozen=True)
class EmbeddingRequest:
    texts: tuple[str, ...]
    model: str = MODEL_ID
    dimensions: int = DIMENSIONS


def validate_texts(texts: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(texts, list | tuple) or not (0 <= len(texts) <= MAX_BATCH):
        raise InvalidInputError()
    if not texts and not allow_empty:
        raise InvalidInputError()
    size = 0
    for value in texts:
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError()
        # Short-circuit oversized strings before allocating an encoded copy.
        if len(value) > INPUT_LIMIT:
            raise InvalidInputError()
        try:
            size += len(value.encode("utf-8"))
        except UnicodeError:
            raise InvalidInputError() from None
        if size > INPUT_LIMIT:
            raise InvalidInputError()
    return tuple(texts)


def decode_request(body: bytes) -> EmbeddingRequest:
    request = strict_json.loads(body, limit=BODY_LIMIT)
    if not isinstance(request, dict) or set(request) != {"input", "model", "dimensions"}:
        raise InvalidInputError()
    if (
        request["model"] != MODEL_ID
        or type(request["dimensions"]) is not int
        or request["dimensions"] != DIMENSIONS
        or not isinstance(request["input"], list)
    ):
        raise InvalidInputError()
    return EmbeddingRequest(validate_texts(request["input"]))


def encode_response(request: EmbeddingRequest, vectors: Sequence[Sequence[float]]) -> bytes:
    if len(vectors) != len(request.texts):
        raise InferenceError()
    rows: list[dict[str, object]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != DIMENSIONS:
            raise InferenceError()
        converted: list[float] = []
        for value in vector:
            if isinstance(value, bool | str | bytes):
                raise InferenceError()
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                raise InferenceError() from None
            if not math.isfinite(number) or abs(number) > 3.4028234663852886e38:
                raise InferenceError()
            converted.append(number)
        if abs(math.sqrt(math.fsum(x * x for x in converted)) - 1.0) > 1e-4:
            raise InferenceError()
        rows.append({"object": "embedding", "index": index, "embedding": converted})
    try:
        return strict_json.dumps(
            {
                "object": "list",
                "data": rows,
                "model": request.model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            },
            limit=RESPONSE_LIMIT,
        )
    except InvalidInputError:
        raise InferenceError() from None


def encode_error(error: EmbeddingError) -> tuple[int, bytes]:
    return error.http_status, strict_json.dumps({"error": {"code": error.code}}, limit=1024)
