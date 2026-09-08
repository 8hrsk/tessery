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
    assert runtime.diagnostics()["dispatches"] == {name: 1}
    assert runtime.active_bytes == 0


@pytest.mark.parametrize(
    "m,n,k", [(1, 32, 64), (7, 33, 128), (8, 32, 64), (16, 64, 1024), (8, 32, 3072)]
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
        name = "linear4_tiled" if m % 8 == 0 and n % 32 == 0 else "linear4"
        assert runtime.diagnostics()["dispatches"] == {name: 1}
    finally:
        for buffer in buffers:
            buffer.close()


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
