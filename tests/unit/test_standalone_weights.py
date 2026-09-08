import hashlib
import json
import struct

import numpy as np
import pytest

from metal_inference import weights
from metal_inference.errors import ManifestError


def container(header, payload=b"\x00" * 8):
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + payload


def test_snapshot_views():
    data = container(
        {
            "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "b": {"dtype": "U32", "shape": [1], "data_offsets": [4, 8]},
        }
    )
    tensors = weights.SafeTensors(data)
    assert np.array_equal(tensors.view("a", shape=(2,), dtype="BF16"), np.zeros(4, np.uint8))
    for args in [("a", (3,), "BF16"), ("none", (2,), "BF16"), ("a", (2,), "U32")]:
        with pytest.raises(ManifestError):
            tensors.view(args[0], shape=args[1], dtype=args[2])


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        {},
        {"dtype": "pickle", "shape": [2], "data_offsets": [0, 8]},
        {"dtype": "U32", "shape": [True], "data_offsets": [0, 4]},
        {"dtype": "U32", "shape": [], "data_offsets": [0, 8]},
        {"dtype": "U32", "shape": [2, 2, 2], "data_offsets": [0, 8]},
        {"dtype": "U32", "shape": [2], "data_offsets": [0]},
        {"dtype": "U32", "shape": [2], "data_offsets": [0, True]},
        {"dtype": "U32", "shape": [2], "data_offsets": [-1, 7]},
        {"dtype": "U32", "shape": [2], "data_offsets": [0, 7]},
        {"dtype": "U32", "shape": [2], "data_offsets": [0, 12]},
    ],
)
def test_malformed_tensor(record):
    with pytest.raises(ManifestError):
        weights.SafeTensors(container({"x": record}))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"1234567",
        struct.pack("<Q", 0),
        struct.pack("<Q", 2**32),
        struct.pack("<Q", 20) + b"{}",
        struct.pack("<Q", 1) + b"!" + b"\0" * 8,
        container([]),
        container({}),
        container({"a": {"dtype": "U32", "shape": [1], "data_offsets": [4, 8]}}),
        container(
            {
                "a": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]},
            }
        ),
    ],
)
def test_invalid_container(data):
    with pytest.raises(ManifestError):
        weights.SafeTensors(data)


def test_safe_read_snapshot(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    data = b'{"hello":1}'
    monkeypatch.setattr(
        weights, "ARTIFACTS", {"config.json": (len(data), hashlib.sha256(data).hexdigest())}
    )
    file = root / "config.json"
    file.write_bytes(data)
    assert weights.read_json(str(root), "config.json") == {"hello": 1}
    (root / "untrusted.py").write_text("raise RuntimeError")
    assert weights.read_artifact(str(root), "config.json") == data
    for name in ["../config.json", "unknown"]:
        with pytest.raises(ManifestError):
            weights.read_artifact(str(root), name)
    file.write_bytes(b"x" * len(data))
    with pytest.raises(ManifestError):
        weights.read_artifact(str(root), "config.json")
    file.unlink()
    file.symlink_to(root / "untrusted.py")
    with pytest.raises(ManifestError):
        weights.read_artifact(str(root), "config.json")


def test_directory_failures(tmp_path):
    for path in ["relative", str(tmp_path / ".."), str(tmp_path / "missing")]:
        with pytest.raises(ManifestError):
            weights.read_artifact(path, "config.json")
