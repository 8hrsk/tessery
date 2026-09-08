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
