from types import SimpleNamespace

import pytest

from metal_inference import metal
from metal_inference.errors import NativeBuildError


@pytest.mark.parametrize(
    "rows,cols,k,selected",
    [
        (m, n, k, True)
        for m in (16, 32, 128, 4096)
        for n, k in ((1024, 1024), (2048, 1024), (3072, 1024), (1024, 2048), (1024, 3072))
    ]
    + [(m, 1024, 1024, False) for m in (1, 4, 7, 8, 15, 17, 24, 31, 33, 264)]
    + [
        (16, n, k, False)
        for n, k in ((64, 1024), (1025, 1024), (1024, 64), (1024, 1023), (4096, 1024))
    ],
)
def test_quantized_large_tile_dispatch_guard(rows, cols, k, selected):
    calls = []
    runtime = SimpleNamespace(_dispatch=lambda *args, **kwargs: calls.append((args, kwargs)))
    metal.MetalRuntime._linear4(runtime, [], rows=rows, cols=cols, k=k)
    assert (calls[0][0][0] == "linear4_16x32_k64") == selected
    if selected:
        assert len(calls) == 1
        assert calls[0][1] == dict(
            threads=(rows // 16) * (cols // 32) * 256, group_size=256, rows=rows, cols=cols, k=k
        )
    else:
        assert all(args[0] in {"linear4_tiled", "linear4_tail", "linear4"} for args, _ in calls)


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
        (65, 32, False),
        (71, 128, False),
        (136, 32, False),
        (513, 32, False),
        (1024, 128, True),
    ],
)
def test_attention_dispatch_verified_shape_guard(seq, dim, tiled):
    calls = []
    runtime = SimpleNamespace(_dispatch=lambda *args, **kwargs: calls.append((args, kwargs)))
    metal.MetalRuntime._attention(
        runtime, [], tokens=2 * seq, seq=seq, heads=4, kv_heads=2, dim=dim
    )
    args, kwargs = calls[0]
    tail = 64 <= seq <= 512 and seq % 32 and dim in (32, 128)
    assert args[0] == (
        f"attention_tail_{dim}" if tail else "attention_tiled" if tiled else "attention"
    )
    assert kwargs["group_size"] == (128 if tiled or tail else 32)
    expected_threads = (
        2 * ((seq + 7) // 8) * 4 * 128
        if tail
        else (2 * seq // 8 * 4 * 128 if tiled else 2 * seq * 4 * 32)
    )
    assert kwargs["threads"] == expected_threads


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


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 12, 15, 16, 129, 4096])
@pytest.mark.parametrize(
    "cols,k",
    [(384, 384), (1536, 384), (384, 1536), (385, 384), (384, 1535), (384, 512), (384, 2048)],
)
def test_f32_verified_shapes_and_tail_dispatch(rows, cols, k):
    calls = []
    runtime = SimpleNamespace(_dispatch=lambda *a, **kw: calls.append((a, kw)))
    metal.MetalRuntime._matmul_f32(runtime, [], rows=rows, cols=cols, k=k)
    selected = rows >= 8 and (cols, k) in ((384, 384), (1536, 384), (384, 1536))
    if selected:
        assert calls[0][0][0] == "matmul_f32_chunk32"
        assert calls[0][1] == dict(
            threads=(rows // 8) * (cols // 32) * 128, group_size=128, rows=rows, cols=cols, k=k
        )
        assert len(calls) == 1 + bool(rows % 8)
        if rows % 8:
            assert calls[1][0][0] == "matmul_f32"
            assert calls[1][1] == dict(
                threads=((rows % 8 + 3) // 4) * cols * 32,
                group_size=32,
                n=rows // 8 * 8,
                rows=rows,
                cols=cols,
                k=k,
            )
    else:
        tiled = rows >= 8 and rows % 8 == 0 and cols % 32 == 0 and k % 8 == 0 and k <= 512
        assert len(calls) == 1
        assert calls[0][0][0] == ("matmul_f32_tiled" if tiled else "matmul_f32")
        assert calls[0][1].get("n", 0) == 0
