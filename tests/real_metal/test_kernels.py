import math
import os

import numpy as np
import pytest

from metal_inference.errors import InferenceError
from metal_inference.metal import MetalRuntime

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in native GPU tests"),
]


@pytest.fixture
def runtime():
    with MetalRuntime() as engine:
        yield engine


def bf16(array):
    return (np.asarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def run(engine, kernel, inputs, shape, **params):
    buffers = []
    try:
        for array in inputs:
            array = np.ascontiguousarray(array)
            buffers.append(engine.buffer(array.nbytes, array))
        output = engine.buffer(int(np.prod(shape)) * 4)
        buffers.append(output)
        with engine.command():
            engine._dispatch(kernel, buffers, **params)
        return engine.read(output, shape)
    finally:
        for buffer in buffers:
            buffer.close()


def test_add_and_resources(runtime):
    rng = np.random.default_rng(42)
    a = rng.normal(size=(7, 37)).astype(np.float32)
    b = rng.normal(size=a.shape).astype(np.float32)
    np.testing.assert_array_equal(runtime.add(a, b), a + b)
    assert runtime.active_bytes == 0
    assert runtime.peak_bytes == a.nbytes * 3
    with pytest.raises(InferenceError):
        runtime.add(a, b[:1])
    with pytest.raises(InferenceError):
        runtime.buffer(0)


def test_quantized_embedding_and_linear(runtime):
    rng = np.random.default_rng(1)
    code = rng.integers(0, 16, size=(8, 128), dtype=np.uint32)
    packed = np.bitwise_or.reduce(
        code.reshape(8, 16, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
    )
    scales = bf16(np.full((8, 2), 0.125))
    biases = bf16(np.full((8, 2), -0.5))
    dequant = code.astype(np.float32) * 0.125 - 0.5
    ids = np.array([7, 1, 3], np.uint32)
    embedded = run(
        runtime, "embedding4", [ids, packed, scales, biases], (3, 128), threads=384, n=384, cols=128
    )
    np.testing.assert_array_equal(embedded, dequant[ids])
    x = rng.normal(size=(3, 128)).astype(np.float32)
    result = run(
        runtime,
        "linear4",
        [x, packed, scales, biases],
        (3, 8),
        threads=8 * 32,
        group_size=32,
        rows=3,
        cols=8,
        k=128,
    )
    np.testing.assert_allclose(result, x @ dequant.T, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(1, 17, 3), (7, 65, 9), (4, 128, 16)])
def test_general_matmul(runtime, shape):
    m, k, n = shape
    rng = np.random.default_rng(7)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    np.testing.assert_allclose(runtime.matmul(a, b), a @ b, atol=1e-5, rtol=1e-5)
    assert runtime.active_bytes == 0
    with pytest.raises(InferenceError):
        runtime.matmul(a, b[:1])


def test_rms_and_pool(runtime):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(2, 3, 64)).astype(np.float32)
    result = run(
        runtime,
        "rms_norm",
        [x, bf16(np.ones(64))],
        x.shape,
        threads=6 * 32,
        group_size=32,
        cols=64,
        eps=1e-6,
    )
    expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(result, expected, atol=1e-6)
    pooled = run(
        runtime,
        "pool_project",
        [x, np.array([1, 3], np.uint32)],
        (2, 32),
        threads=2 * 32,
        group_size=32,
        seq=3,
        cols=64,
        dim=32,
    )
    expected = x[np.arange(2), [0, 2], :32]
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_allclose(pooled, expected, atol=1e-6)


def test_attention_against_independent_numpy(runtime):
    rng = np.random.default_rng(17)
    b, s, h, kv, d = 2, 5, 4, 2, 64
    q = rng.normal(size=(b, s, h, d)).astype(np.float32)
    k = rng.normal(size=(b, s, kv, d)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    lengths = np.array([5, 2], np.uint32)
    result = run(
        runtime,
        "attention",
        [q, k, v, lengths],
        q.shape,
        threads=b * s * h * 32,
        group_size=32,
        seq=s,
        heads=h,
        kv_heads=kv,
        dim=d,
        scale=d**-0.5,
    )
    expected = np.empty_like(q)
    for batch in range(b):
        for pos in range(s):
            end = min(pos + 1, int(lengths[batch]))
            for head in range(h):
                kh = head // (h // kv)
                scores = k[batch, :end, kh] @ q[batch, pos, head] / np.sqrt(d)
                probs = np.exp(scores - scores.max())
                probs /= probs.sum()
                expected[batch, pos, head] = probs @ v[batch, :end, kh]
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dim", [32, 128])
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("seq,length_pair", [(64, (1, 33)), (128, (31, 32)), (512, (511, 512))])
def test_tiled_attention_ragged_f64(runtime, dim, bidirectional, seq, length_pair):
    rng = np.random.default_rng(393)
    heads, kv = 4, 2
    q = rng.normal(size=(2, seq, heads, dim)).astype(np.float32)
    k, v = [rng.normal(size=(2, seq, kv, dim)).astype(np.float32) for _ in range(2)]
    lengths = np.array(length_pair, np.uint32)
    result = run(
        runtime,
        "attention_tiled",
        [q, k, v, lengths],
        q.shape,
        threads=2 * (seq // 8) * heads * 128,
        group_size=128,
        seq=seq,
        heads=heads,
        kv_heads=kv,
        dim=dim,
        scale=dim**-0.5,
        bidirectional=bidirectional,
    )
    expected = np.empty_like(q, dtype=np.float64)
    for b in range(2):
        for h in range(heads):
            kh = h // (heads // kv)
            length = int(lengths[b])
            scores = q[b, :, h].astype(np.float64) @ k[b, :length, kh].astype(np.float64).T
            scores *= dim**-0.5
            if not bidirectional:
                scores[np.arange(length)[None, :] > np.arange(seq)[:, None]] = -np.inf
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            expected[b, :, h] = probs @ v[b, :length, kh].astype(np.float64)
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pattern", ["alternating", "later_maximum", "uniform"])
def test_tiled_attention_extreme_logits_and_masked_blocks(runtime, bidirectional, pattern):
    rng = np.random.default_rng(718)
    seq, heads, dim = 128, 2, 128
    q = np.full((1, seq, heads, dim), 10, np.float32)
    k = q.copy()
    k[:, 1::2] *= -1  # Logits near +/-1131: a naive exp would overflow.
    if pattern == "later_maximum":
        k[:] = -10
        k[:, 32:64] = 10
    elif pattern == "uniform":
        k[:] = 0
    v = rng.normal(size=q.shape).astype(np.float32)
    length = 65 if pattern == "later_maximum" else 33
    lengths = np.array([length], np.uint32)
    v[:, length:] = 1e10  # Entire later blocks are masked, including padded queries.
    result = run(
        runtime,
        "attention_tiled",
        [q, k, v, lengths],
        q.shape,
        threads=(seq // 8) * heads * 128,
        group_size=128,
        seq=seq,
        heads=heads,
        kv_heads=heads,
        dim=dim,
        scale=dim**-0.5,
        bidirectional=bidirectional,
    )
    expected = np.empty_like(q, dtype=np.float64)
    for pos in range(seq):
        end = length if bidirectional else min(pos + 1, length)
        if pattern == "later_maximum":
            chosen = v[0, 32 : min(end, 64)] if end > 32 else v[0, :end]
        else:
            chosen = v[0, :end:2] if pattern == "alternating" else v[0, :end]
        expected[0, pos] = chosen.astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-5)


def test_rope_and_silu(runtime):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 3, 4, 64)).astype(np.float32)
    buffer = runtime.buffer(x.nbytes, x)
    try:
        with runtime.command():
            runtime._dispatch(
                "rope",
                [buffer],
                threads=x.size // 2,
                n=x.size // 2,
                heads=4,
                seq=3,
                dim=64,
                theta=10000,
            )
        result = runtime.read(buffer, x.shape)
        angles = (
            np.arange(3)[None, :, None, None]
            * 10000 ** (-2 * np.arange(32) / 64)[None, None, None, :]
        )
        expected = np.concatenate(
            (
                x[..., :32] * np.cos(angles) - x[..., 32:] * np.sin(angles),
                x[..., :32] * np.sin(angles) + x[..., 32:] * np.cos(angles),
            ),
            axis=-1,
        )
        np.testing.assert_allclose(result, expected, atol=2e-6)
    finally:
        buffer.close()
    up = rng.normal(size=x.shape).astype(np.float32)
    a = runtime.buffer(x.nbytes, x)
    b = runtime.buffer(up.nbytes, up)
    try:
        with runtime.command():
            runtime._dispatch("silu_gate", [a, b], threads=x.size, n=x.size)
        np.testing.assert_allclose(runtime.read(a, x.shape), x / (1 + np.exp(-x)) * up, atol=1e-6)
    finally:
        a.close()
        b.close()


def test_layer_norm_and_absolute_embedding(runtime):
    rng = np.random.default_rng(37)
    x = rng.normal(size=(3, 67)).astype(np.float32)
    scale = rng.normal(size=67).astype(np.float32)
    bias = rng.normal(size=67).astype(np.float32)
    out = run(
        runtime,
        "layer_norm",
        [x, scale, bias],
        x.shape,
        threads=3 * 32,
        group_size=32,
        cols=67,
        eps=1e-12,
    )
    expected = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + 1e-12
    ) * scale + bias
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=1e-5)
    words = rng.normal(size=(17, 67)).astype(np.float32)
    positions = rng.normal(size=(4, 67)).astype(np.float32)
    types = rng.normal(size=(2, 67)).astype(np.float32)
    ids = np.array([[1, 3, 0, 0], [5, 6, 7, 8]], np.uint32)
    out = run(
        runtime,
        "embedding_position",
        [ids, words, positions, types],
        (2, 4, 67),
        threads=8 * 67,
        n=8 * 67,
        cols=67,
        seq=4,
    )
    np.testing.assert_allclose(out, words[ids] + types[0] + positions, atol=1e-6)


def test_gelu_erf_form_and_bias(runtime):
    x = np.linspace(-12, 12, 10001, dtype=np.float32)
    buffer = runtime.buffer(x.nbytes, x)
    try:
        with runtime.command():
            runtime._dispatch("gelu_f32", [buffer], threads=x.size, n=x.size)
        expected = np.array(
            [0.5 * float(v) * (1 + math.erf(float(v) / math.sqrt(2))) for v in x], np.float32
        )
        np.testing.assert_allclose(runtime.read(buffer, x.shape), expected, atol=2e-6, rtol=1e-6)
    finally:
        buffer.close()
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    bias = np.array([-2, -1, 1, 2], np.float32)
    a, b = runtime.buffer(x.nbytes, x), runtime.buffer(bias.nbytes, bias)
    try:
        with runtime.command():
            runtime._dispatch("add_bias", [a, b], threads=x.size, n=x.size, cols=4)
        np.testing.assert_array_equal(runtime.read(a, x.shape), x + bias)
    finally:
        a.close()
        b.close()


def test_bidirectional_attention_and_cls_pool(runtime):
    rng = np.random.default_rng(83)
    batch, seq, heads, dim = 2, 5, 3, 8
    q, k, v = [rng.normal(size=(batch, seq, heads, dim)).astype(np.float32) for _ in range(3)]
    lengths = np.array([5, 2], np.uint32)
    out = run(
        runtime,
        "attention",
        [q, k, v, lengths],
        q.shape,
        threads=batch * seq * heads * 32,
        group_size=32,
        seq=seq,
        heads=heads,
        kv_heads=heads,
        dim=dim,
        scale=dim**-0.5,
        bidirectional=True,
    )
    expected = np.empty_like(q)
    for b in range(batch):
        for h in range(heads):
            scores = q[b, :, h] @ k[b, : lengths[b], h].T / np.sqrt(dim)
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            expected[b, :, h] = probabilities @ v[b, : lengths[b], h]
    np.testing.assert_allclose(out, expected, atol=2e-6, rtol=2e-5)
    x = q.reshape(batch, seq, heads * dim)
    pooled = run(
        runtime,
        "pool_project",
        [x, lengths],
        (batch, heads * dim),
        threads=batch * 32,
        group_size=32,
        seq=seq,
        cols=heads * dim,
        dim=heads * dim,
        first_token=True,
    )
    expected = x[:, 0] / np.linalg.norm(x[:, 0], axis=-1, keepdims=True)
    np.testing.assert_allclose(pooled, expected, atol=1e-6)


@pytest.mark.parametrize(
    "shape", [(8, 8, 32), (16, 384, 384), (32, 512, 384), (32, 1536, 384), (8, 65, 32), (9, 64, 32)]
)
def test_tiled_matmul_and_unaligned_fallback(runtime, shape):
    m, k, n = shape
    rng = np.random.default_rng(33)
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    expected = a.astype(np.float64) @ b.astype(np.float64)
    np.testing.assert_allclose(runtime.matmul(a, b), expected, atol=5e-5, rtol=5e-5)
    tiled = m % 8 == 0 and k % 8 == 0 and k <= 512
    name = "matmul_f32_tiled" if tiled else "matmul_f32"
    if m >= 8 and (n, k) in ((384, 384), (1536, 384), (384, 1536)):
        name = "matmul_f32_chunk32"
    assert runtime.diagnostics()["dispatches"] == {name: 1}
    assert runtime.active_bytes == 0


@pytest.mark.parametrize(
    "m,n,k",
    [(m, 64, k) for m in (1, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 31) for k in (64, 1024, 3072)]
    + [(7, 33, 128)],
)
def test_uint4_tiled_accuracy_and_route(runtime, m, n, k):
    rng = np.random.default_rng(417)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
    packed = np.bitwise_or.reduce(
        codes.reshape(n, k // 8, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
    )
    scales = bf16(rng.uniform(0.01, 0.2, size=(n, k // 64)))
    biases = bf16(rng.uniform(-1, 0.1, size=scales.shape))
    decoded = codes.astype(np.float32) * (scales.astype(np.uint32) << 16).view(np.float32).repeat(
        64, axis=1
    )
    decoded += (biases.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
    x = rng.normal(size=(m, k)).astype(np.float32)
    buffers = [runtime.buffer(a.nbytes, a) for a in (x, packed, scales, biases)]
    buffers.append(runtime.buffer(m * n * 4))
    try:
        with runtime.command():
            runtime._linear4(buffers, rows=m, cols=n, k=k)
        expected = x.astype(np.float64) @ decoded.astype(np.float64).T
        np.testing.assert_allclose(
            runtime.read(buffers[-1], (m, n)), expected, atol=5e-5, rtol=5e-5
        )
        expected_dispatches = {"linear4": 1}
        if m >= 5 and n % 32 == 0:
            expected_dispatches = {}
            if m >= 8:
                expected_dispatches["linear4_tiled"] = 1
            if m % 8:
                expected_dispatches["linear4" if m % 8 <= 4 else "linear4_tail"] = 1
        assert runtime.diagnostics()["dispatches"] == expected_dispatches

    finally:
        for buffer in buffers:
            buffer.close()


@pytest.mark.parametrize(
    "m,n,k",
    [
        (16, 1024, 1024),
        (32, 2048, 1024),
        (16, 3072, 1024),
        (32, 1024, 2048),
        (32, 1024, 3072),
        (4096, 1024, 1024),
    ],
)
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
def test_large_uint4_tile_f64_and_previous_accumulation(runtime, m, n, k, pattern):
    rng = np.random.default_rng(821)
    codes = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
    x = rng.normal(size=(m, k)).astype(np.float32)
    if pattern == "cancellation":
        codes[:, 1::2] = codes[:, ::2]
        x[:, ::2], x[:, 1::2] = 1.0, -1.0
    packed = np.bitwise_or.reduce(
        codes.reshape(n, k // 8, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1
    )
    scales = bf16(rng.uniform(0.01, 0.2, size=(n, k // 64)))
    biases = bf16(rng.uniform(-1, 0.1, size=scales.shape))
    weights = codes.astype(np.float32) * (scales.astype(np.uint32) << 16).view(np.float32).repeat(
        64, axis=1
    )
    weights += (biases.astype(np.uint32) << 16).view(np.float32).repeat(64, axis=1)
    expected = x.astype(np.float64) @ weights.astype(np.float64).T
    inputs = [x, packed, scales, biases]
    previous = run(
        runtime,
        "linear4_tiled",
        inputs,
        (m, n),
        threads=(m // 8) * (n // 32) * 128,
        group_size=128,
        rows=m,
        cols=n,
        k=k,
    )
    actual = run(
        runtime,
        "linear4_16x32_k64",
        inputs,
        (m, n),
        threads=(m // 16) * (n // 32) * 256,
        group_size=256,
        rows=m,
        cols=n,
        k=k,
    )
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
    np.testing.assert_array_equal(actual, previous)


def test_workspace_reuse_bound_abort_and_trim(runtime):
    runtime.workspace_limit_bytes = 1024
    with runtime._workspace() as allocate:
        a, b = allocate(512), allocate(512)
        assert a is not b
        with runtime.command():
            runtime._dispatch("add", [a, b, a], threads=128, n=128)
    assert runtime.active_bytes == runtime.cache_bytes == 1024
    allocations = runtime.diagnostics()["allocations"]
    with pytest.raises(InferenceError), runtime._workspace() as allocate:
        recovered = allocate(512)
        with runtime.command():
            runtime._dispatch("nonexistent_kernel", [recovered], threads=1)
    assert runtime.diagnostics()["allocations"] == allocations
    with pytest.raises(InferenceError):
        allocate(512)  # No use after lease release.
    with runtime._workspace() as allocate:
        large = allocate(2048)
        assert large.pointer
    assert not large.pointer
    assert runtime.cache_bytes <= 1024
    with runtime._workspace() as allocate:
        allocate(768)
    assert runtime.cache_bytes == runtime.active_bytes == 768
    np.testing.assert_array_equal(
        runtime.add(np.ones(4, np.float32), np.ones(4, np.float32)), [2] * 4
    )
    runtime.trim_workspace()
    assert runtime.active_bytes == runtime.cache_bytes == 0


def test_workspace_uploads_are_not_retained_and_live_leases_are_exclusive(runtime):
    x = np.ones(16, np.float32)
    with runtime._workspace() as allocate:
        upload = allocate(x.nbytes, x)
        with pytest.raises(InferenceError), runtime._workspace():
            pass
        with pytest.raises(InferenceError):
            runtime.close()
        scratch = allocate(512)
    assert not upload.pointer and scratch.pointer
    runtime.close()
    assert not scratch.pointer and runtime.active_bytes == runtime.cache_bytes == 0


def test_kernel_profile_timings_abort_and_recovery(runtime):
    a = np.arange(4096, dtype=np.float32)
    with runtime.profile_kernels() as records:
        np.testing.assert_array_equal(runtime.add(a, a), a + a)
        assert len(records) == 1 and records[0]["kernel"] == "add"
        assert np.isfinite(records[0]["gpu_seconds"]) and records[0]["gpu_seconds"] > 0
        with pytest.raises(InferenceError), runtime.profile_kernels():
            pass
        with pytest.raises(InferenceError):
            runtime.close()
        with pytest.raises(InferenceError), runtime.command():
            runtime._dispatch("absent", [], threads=1)
        assert len(records) == 1
        np.testing.assert_array_equal(runtime.add(a, a), a + a)
        assert len(records) == 2
    np.testing.assert_array_equal(runtime.add(a, a), a + a)
    assert len(records) == 2
    assert runtime.active_bytes == 0


def test_kernel_profile_capacity_aborts_without_publishing_partial_results(runtime):
    a = np.ones(1, dtype=np.float32)
    buffers = [runtime.buffer(a.nbytes, a) for _ in range(3)]
    try:
        with runtime.profile_kernels() as records:
            with pytest.raises(InferenceError), runtime.command():
                for _ in range(2049):
                    runtime._dispatch("add", buffers, threads=1, n=1)
            assert records == []
            np.testing.assert_array_equal(runtime.add(a, a), a + a)
            assert len(records) == 1
    finally:
        for buffer in buffers:
            buffer.close()
    assert runtime.active_bytes == 0


@pytest.mark.parametrize("m", [8, 9, 10, 11, 12, 13, 14, 15, 128, 4096])
@pytest.mark.parametrize("n,k", [(384, 384), (1536, 384), (384, 1536)])
@pytest.mark.parametrize("pattern", ["random", "cancellation"])
def test_f32_chunked_projections_and_bounded_tail(runtime, m, n, k, pattern):
    rng = np.random.default_rng(2026 + m)
    x = rng.normal(size=(m, k)).astype(np.float32)
    w = rng.normal(size=(n, k)).astype(np.float32)
    if pattern == "cancellation":
        # Opposing products, with a small representable residual in every pair.
        x[:, 1::2] = -x[:, ::2]
        w[:, 1::2] = w[:, ::2] + np.float32(2**-16)
    expected = x.astype(np.float64) @ w.astype(np.float64).T
    output = runtime.matmul(x, np.ascontiguousarray(w.T))
    np.testing.assert_allclose(output, expected, atol=5e-5, rtol=5e-5)
    dispatches = {"matmul_f32_chunk32": 1}
    if m % 8:
        dispatches["matmul_f32"] = 1
    assert runtime.diagnostics()["dispatches"] == dispatches
    assert runtime.active_bytes == 0


def test_f32_scalar_tail_does_not_touch_complete_rows(runtime):
    rng = np.random.default_rng(33)
    x = rng.normal(size=(15, 1536)).astype(np.float32)
    w = rng.normal(size=(384, 1536)).astype(np.float32)
    sentinel = np.full((15, 384), np.nan, np.float32)
    buffers = [runtime.buffer(a.nbytes, a) for a in (x, w, sentinel)]
    try:
        with runtime.command():
            runtime._dispatch(
                "matmul_f32",
                buffers,
                threads=2 * 384 * 32,
                group_size=32,
                n=8,
                rows=15,
                cols=384,
                k=1536,
            )
        output = runtime.read(buffers[-1], (15, 384))
        assert np.isnan(output[:8]).all()
        expected = x[8:].astype(np.float64) @ w.astype(np.float64).T
        np.testing.assert_allclose(output[8:], expected, atol=5e-5, rtol=5e-5)
    finally:
        for buffer in buffers:
            buffer.close()
