import copy
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace

import pytest

from metal_inference import Artifact, EmbeddingModel, ModelProfile, get_profile, list_profiles
from metal_inference.backend import config_float, config_int
from metal_inference.errors import ConfigurationError, ManifestError, UnsupportedProfileError
from metal_inference.weights import read_artifact, read_json


def test_registry_roundtrip_and_identity(tmp_path):
    assert len(list_profiles()) == 2
    for name in list_profiles():
        profile = get_profile(name)
        file = tmp_path / "profile.json"
        file.write_text(json.dumps(profile.to_dict()))
        assert ModelProfile.from_file(file) == profile
        assert ModelProfile.from_dict(profile.to_dict()) == profile
        assert get_profile(profile) is profile
        assert get_profile(name).compatibility_id == profile.compatibility_id
        renamed = replace(
            profile,
            artifacts=tuple(
                replace(a, filename="blob-" + a.name) for a in reversed(profile.artifacts)
            ),
        )
        assert renamed.identity_sha256 == profile.identity_sha256
        assert renamed.compatibility_id == profile.compatibility_id
        changed = replace(profile, revision="different-weights-revision")
        assert changed.compatibility_id != profile.compatibility_id
        with pytest.raises(FrozenInstanceError):
            profile.revision = "mutated"
        with pytest.raises(ManifestError):
            profile.artifact("unlisted.bin")
    assert get_profile(list_profiles()[0]).compatibility_id == "metal-inference-qwen3-f32-v1"
    for name in ("unknown", None, 1):
        with pytest.raises(UnsupportedProfileError):
            get_profile(name)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"unexpected": "code.py"},
        {"model_id": ""},
        {"revision": 5},
        {"native_dimensions": True},
        {"native_dimensions": 4097},
        {"min_dimensions": 1},
        {"default_dimensions": 10000},
        {"max_length": 513},
        {"max_length": 1},
        {"artifacts": []},
        {"artifacts": {}},
        {"artifacts": [None] * 3},
        {"min_dimensions": 32},
    ],
)
def test_invalid_profile_schema(change):
    data = get_profile("bge-small-en-v1.5").to_dict()
    data.update(change)
    with pytest.raises(ManifestError):
        ModelProfile.from_dict(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "import:bad.py"),
        ("tokenizer", "sentencepiece"),
        ("pooling", "mean"),
    ],
)
def test_unsupported_adapter_combinations(field, value):
    data = get_profile("bge-small-en-v1.5").to_dict()
    data[field] = value
    with pytest.raises(UnsupportedProfileError):
        ModelProfile.from_dict(data)


@pytest.mark.parametrize(
    "change",
    [
        {"name": []},
        {"name": "other.bin"},
        {"filename": "../secret"},
        {"filename": "/secret"},
        {"filename": "sub/file"},
        {"filename": "."},
        {"filename": ""},
        {"size": True},
        {"size": 0},
        {"size": 2**31 + 1},
        {"size": 17 * 1024 * 1024},
        {"sha256": "f" * 63},
        {"sha256": "A" * 64},
        {"sha256": None},
    ],
)
def test_invalid_artifacts(change):
    fields = {"name": "config.json", "filename": "config.json", "size": 2, "sha256": "a" * 64}
    fields.update(change)
    with pytest.raises(ManifestError):
        Artifact(**fields)


def test_duplicate_manifest_and_bounded_file(tmp_path):
    profile = get_profile("bge-small-en-v1.5")
    for artifacts in (
        (profile.artifacts[0],) * 3,
        tuple(replace(a, filename="duplicate") for a in profile.artifacts),
    ):
        with pytest.raises(ManifestError):
            replace(profile, artifacts=artifacts)
    for value in (None, [], {}, {"schema_version": 1}):
        with pytest.raises(ManifestError):
            ModelProfile.from_dict(value)
    file = tmp_path / "manifest.json"
    for data in (b'{"schema_version":1,"schema_version":1}', b" " * 65537, b"invalid"):
        file.write_bytes(data)
        with pytest.raises(ManifestError):
            ModelProfile.from_file(file)
    with pytest.raises(ManifestError):
        ModelProfile.from_file(tmp_path / "missing")


def test_profile_file_refuses_links_and_special_files(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    target = tmp_path / "regular"
    target.write_text(json.dumps(get_profile("bge-small-en-v1.5").to_dict()))
    link = tmp_path / "link"
    link.symlink_to(target)
    for path in (fifo, link, tmp_path):
        with pytest.raises(ManifestError):
            ModelProfile.from_file(path)


def test_explicit_blob_mapping_integrity_and_no_links(tmp_path):
    root = tmp_path.resolve()
    profile = get_profile("bge-small-en-v1.5")
    data = b'{"value":3}'
    artifact = Artifact("config.json", "opaque-blob", len(data), hashlib.sha256(data).hexdigest())
    profile = replace(profile, artifacts=(artifact, *profile.artifacts[1:]))
    blob = root / artifact.filename
    blob.write_bytes(data)
    assert read_json(str(root), "config.json", profile=profile) == {"value": 3}
    assert read_artifact(str(root), "config.json", profile=profile) == data
    blob.write_bytes(b"x" * len(data))
    with pytest.raises(ManifestError):
        read_artifact(str(root), "config.json", profile=profile)
    blob.unlink()
    target = root / "target"
    target.write_bytes(data)
    blob.symlink_to(target)
    with pytest.raises(ManifestError):
        read_artifact(str(root), "config.json", profile=profile)


@pytest.mark.parametrize("options", [{"dimensions": 32}, {"max_length": 1}, {"dimensions": 385}])
def test_bert_limits_before_file_io(options):
    with pytest.raises(ConfigurationError):
        EmbeddingModel.load("/not-read", profile="bge-small-en-v1.5", **options)


def test_bounded_config_parsing():
    assert config_int({"n": 32}, "n", 1, 64) == 32
    assert config_float({"eps": 1e-12}, "eps", 1e-12, 1) == 1e-12
    for value in (None, True, "32", 0, 65, 1.5):
        with pytest.raises(UnsupportedProfileError):
            config_int({"n": value}, "n", 1, 64)
    for value in (None, True, "1", -1, float("nan"), float("inf"), 10**400):
        with pytest.raises(UnsupportedProfileError):
            config_float({"eps": value}, "eps", 1e-12, 1)


def test_fingerprint_includes_weight_digest():
    data = get_profile("bge-small-en-v1.5").to_dict()
    changed = copy.deepcopy(data)
    changed["artifacts"][2]["sha256"] = "f" * 64
    assert (
        ModelProfile.from_dict(data).compatibility_id
        != ModelProfile.from_dict(changed).compatibility_id
    )
