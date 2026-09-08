"""RFC JSON boundary shared by protocol and trusted manifest parsing."""

import json
import math
from typing import Any

from .errors import InvalidInputError


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _constant(_: str) -> Any:
    raise ValueError


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError
    return parsed


def loads(data: bytes, *, limit: int) -> Any:
    if len(data) > limit:
        raise InvalidInputError()
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (ValueError, RecursionError):
        raise InvalidInputError() from None


def dumps(value: object, *, limit: int) -> bytes:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        raise InvalidInputError() from None
    if len(encoded) > limit:
        raise InvalidInputError()
    return encoded
