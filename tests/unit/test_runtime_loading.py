from types import SimpleNamespace

import pytest

from metal_inference import metal
from metal_inference.errors import NativeBuildError


@pytest.mark.parametrize(
    "seq,dim,tiled",
    [
        (32, 32, False),
        (63, 128, False),
        (64, 128, True),
        (65, 128, False),
        (127, 32, False),
        (128, 32, True),
        (129, 32, False),
        (511, 128, False),
        (512, 128, True),
        (128, 64, False),
        (128, 256, False),
    ],
)
def test_attention_dispatch_verified_shape_guard(seq, dim, tiled):
    calls = []
    runtime = SimpleNamespace(_dispatch=lambda *args, **kwargs: calls.append((args, kwargs)))
    metal.MetalRuntime._attention(
        runtime, [], tokens=2 * seq, seq=seq, heads=4, kv_heads=2, dim=dim
    )
    args, kwargs = calls[0]
    assert args[0] == ("attention_tiled" if tiled else "attention")
    assert kwargs["group_size"] == (128 if tiled else 32)
    assert kwargs["threads"] == (2 * seq // 8 * 4 * 128 if tiled else 2 * seq * 4 * 32)


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
