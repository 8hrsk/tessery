from types import SimpleNamespace

import pytest

from metal_inference import metal
from metal_inference.errors import NativeBuildError


@pytest.mark.parametrize("changed_during_load", [False, True])
def test_stale_or_changing_native_library_has_stable_error(
    tmp_path, monkeypatch, changed_during_load
):
    library = tmp_path / "_native.so"
    library.write_bytes(b"placeholder")
    monkeypatch.setattr(metal.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(metal.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        metal.Path,
        "glob",
        lambda self, pattern: iter([library]) if pattern == "_native*.so" else iter([]),
    )

    def fake_load(path):
        if changed_during_load:
            library.write_bytes(b"changed")
        return SimpleNamespace()  # No current ABI symbols.

    monkeypatch.setattr(metal.ct, "CDLL", fake_load)
    with pytest.raises(NativeBuildError) as error:
        metal._library()
    assert str(error.value) == "native_engine_not_built"
