import copy
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from yuri_mlx_embeddings.descriptors import MODEL_ID, REVISION
from yuri_mlx_embeddings.errors import ManifestError
from yuri_mlx_embeddings.manifests import REQUIRED_FILES, parse_manifest, verify_model


def parse(obj):
    data = json.dumps(obj).encode()
    return parse_manifest(data, expected_sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def pack(tmp_path):
    # macOS /var is a symlink, so tests use the canonical /private/var spelling.
    root = tmp_path.resolve() / "модель с пробелами"
    root.mkdir()
    rows = []
    for name in sorted(REQUIRED_FILES):
        data = name.encode()
        path = root / name
        path.write_bytes(data)
        path.chmod(0o600)
        rows.append(
            {
                "path": name,
                "size": len(data),
                "mode": 0o600,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    obj = {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "license": "Apache-2.0",
        "files": rows,
    }
    return root, obj


def test_valid(pack):
    root, obj = pack
    verify_model(str(root), parse(obj))


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"model": "wrong"},
        {"model_revision": "main"},
        {"tokenizer_revision": "main"},
        {"license": "GPL-3.0"},
        {"files": None},
        {"files": []},
        {"other": 1},
    ],
)
def test_manifest_header(pack, update):
    _, obj = pack
    obj.update(update)
    with pytest.raises(ManifestError):
        parse(obj)


@pytest.mark.parametrize(
    "update",
    [
        {"path": "../config.json"},
        {"path": "/config.json"},
        {"path": "x/y"},
        {"path": []},
        {"path": ""},
        {"sha256": "A" * 64},
        {"sha256": None},
        {"size": 0},
        {"size": -1},
        {"size": True},
        {"size": 4 * 1024**3 + 1},
        {"mode": True},
        {"mode": 0o777},
        {"mode": 0o4600},
        {"extra": "x"},
    ],
)
def test_artifact_record(pack, update):
    _, obj = pack
    obj["files"][0].update(update)
    with pytest.raises(ManifestError):
        parse(obj)


def test_invalid_manifest_shapes(pack):
    _, obj = pack
    for value in [None, [], {}, {"schema_version": 1}]:
        with pytest.raises(ManifestError):
            parse(value)
    for row in [None, [], {}, copy.deepcopy(obj["files"][1])]:
        invalid = copy.deepcopy(obj)
        invalid["files"][0] = row
        with pytest.raises(ManifestError):
            parse(invalid)
    for data in [b'{"x":1,"x":2}', b"NaN", b" " * 65537]:
        with pytest.raises(ManifestError):
            parse_manifest(data, expected_sha256=hashlib.sha256(data).hexdigest())
    for digest in [None, "bad", "f" * 64]:
        with pytest.raises(ManifestError):
            parse_manifest(b"{}", expected_sha256=digest)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "corrupt",
        "size",
        "mode",
        "directory",
        "symlink",
        "hardlink",
        "fifo",
        "socket",
    ],
)
def test_pack_rejections(pack, kind):
    root, obj = pack
    path = root / "config.json"
    manifest = parse(obj)
    sock = None
    if kind == "extra":
        (root / "extra.json").write_text("x")
    elif kind == "corrupt":
        path.write_bytes(b"x" * path.stat().st_size)
    elif kind == "size":
        path.write_bytes(b"x")
    elif kind == "mode":
        path.chmod(0o644)
    else:
        path.unlink()
        if kind == "directory":
            path.mkdir()
        elif kind == "symlink":
            path.symlink_to(root / "tokenizer.json")
        elif kind == "hardlink":
            os.link(root / "tokenizer.json", path)
        elif kind == "fifo":
            os.mkfifo(path)
        elif kind == "socket":
            import socket

            sock = socket.socket(socket.AF_UNIX)
            # Bind in a short private path, then rename the same inode into the pack.
            base = "/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
            with tempfile.TemporaryDirectory(dir=base) as short_dir:
                short = Path(short_dir) / "s"
                sock.bind(str(short))
                short.rename(path)
    try:
        with pytest.raises(ManifestError, match="^invalid_manifest$"):
            verify_model(str(root), manifest)
    finally:
        if sock:
            sock.close()


def test_paths(pack):
    root, obj = pack
    linked = root.parent / "alias"
    linked.symlink_to(root, target_is_directory=True)
    for path in [
        "relative",
        str(root / ".." / root.name),
        str(linked),
        str(root / "missing"),
        str(root) + "/./",
    ]:
        with pytest.raises(ManifestError):
            verify_model(path, parse(obj))


def test_streaming_large_file(pack):
    root, obj = pack
    path = root / obj["files"][0]["path"]
    data = b"x" * (2 * 1024 * 1024 + 13)
    path.write_bytes(data)
    obj["files"][0].update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    verify_model(str(root), parse(obj))


def test_safe_os_error(pack, monkeypatch):
    root, obj = pack

    def fail(*args, **kwargs):
        raise PermissionError("secret path")

    monkeypatch.setattr(os, "open", fail)
    with pytest.raises(ManifestError) as error:
        verify_model(str(root), parse(obj))
    assert "secret" not in str(error.value)
    assert error.value.__suppress_context__


def test_constructed_manifest_cannot_traverse(pack):
    root, obj = pack
    manifest = parse(obj)
    bad = replace(manifest.artifacts[0], path="../secret")
    for changed in [
        replace(manifest, artifacts=(bad, *manifest.artifacts[1:])),
        replace(manifest, model_revision="main"),
        replace(manifest, tokenizer_revision="main"),
    ]:
        with pytest.raises(ManifestError):
            verify_model(str(root), changed)


def test_file_growth_during_hashing(pack, monkeypatch):
    root, obj = pack
    original = os.fstat
    first = True

    def grow(fd):
        nonlocal first
        result = original(fd)
        if first:
            first = False
            with (root / obj["files"][0]["path"]).open("ab") as stream:
                stream.write(b"x")
        return result

    monkeypatch.setattr(os, "fstat", grow)
    with pytest.raises(ManifestError):
        verify_model(str(root), parse(obj))


def test_extra_file_after_hashing(pack, monkeypatch):
    root, obj = pack
    original = os.listdir
    calls = 0

    def changed(fd):
        nonlocal calls
        calls += 1
        return original(fd) + (["unexpected"] if calls == 2 else [])

    monkeypatch.setattr(os, "listdir", changed)
    with pytest.raises(ManifestError):
        verify_model(str(root), parse(obj))
