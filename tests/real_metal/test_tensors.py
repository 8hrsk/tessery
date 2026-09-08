import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from metal_inference import MetalRuntime, Tensor
from metal_inference.errors import ClosedError, InferenceError

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(os.getenv("METAL_INFERENCE_TEST") != "1", reason="opt-in native GPU tests"),
]


@pytest.fixture
def runtime():
    with MetalRuntime() as engine:
        yield engine


@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 65, 9), (4, 128, 16)])
def test_composed_operations_keep_intermediates_on_metal(runtime, monkeypatch, shape):
    m, k, n = shape
    rng = np.random.default_rng(24)
    x = rng.normal(size=(m, k)).astype(np.float32)
    w = rng.normal(size=(k, n)).astype(np.float32)
    bias = rng.normal(size=(m, n)).astype(np.float32)
    read = runtime.read

    def no_host_read(*args):
        pytest.fail("intermediate copied to host")

    with runtime.tensor(x) as a, runtime.tensor(w) as b, runtime.tensor(bias) as c:
        monkeypatch.setattr(runtime, "read", no_host_read)
        result = ((a @ b) + c).silu()
        assert result.shape == (m, n)
        assert result.dtype == np.float32
        assert result.nbytes == m * n * 4
        gc.collect()
        assert runtime.active_bytes == x.nbytes + w.nbytes + bias.nbytes + result.nbytes
        monkeypatch.setattr(runtime, "read", read)
        expected = x @ w + bias
        expected /= 1 + np.exp(-expected)
        np.testing.assert_allclose(result.numpy(), expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(a.numpy(), x)
        result.close()
    assert runtime.active_bytes == 0


@pytest.mark.parametrize("shape", [(), (19,), (2, 3, 5)])
def test_scalar_and_nd_tensor_add_silu(runtime, shape):
    data = np.full(shape, 1.5, dtype=np.float32)
    with runtime.tensor(data) as a:
        with (a + a).silu() as result:
            expected = data * 2 / (1 + np.exp(-data * 2))
            np.testing.assert_allclose(result.numpy(), expected, atol=1e-6)
            assert result.shape == shape
        with pytest.raises(InferenceError):
            a.transpose()
        with pytest.raises(InferenceError):
            a @ a
    assert runtime.active_bytes == 0


def test_upload_and_download_are_independent_copies(runtime):
    original = np.arange(35, dtype=np.float32).reshape(5, 7)[:, ::2]
    expected = original.copy()
    with runtime.tensor(original) as tensor:
        original.fill(-1)
        with tensor.transpose() as transposed:
            np.testing.assert_array_equal(transposed.numpy(), expected.T)
        host = tensor.numpy()
        host.fill(0)
        np.testing.assert_array_equal(tensor.numpy(), expected)
    assert runtime.active_bytes == 0


def test_validation_and_cross_runtime(runtime):
    data = np.ones((3, 4), dtype=np.float32)
    for invalid in ([], np.ones(4), np.empty(0, dtype=np.float32)):
        with pytest.raises(InferenceError):
            runtime.tensor(invalid)
    with runtime.tensor(data) as a, runtime.tensor(data[:1]) as b:
        for operation in (lambda: a + b, lambda: a @ b, lambda: a + None, lambda: a @ 1):
            with pytest.raises(InferenceError):
                operation()
        with MetalRuntime() as other, other.tensor(data) as foreign:
            for operation in (lambda: a + foreign, lambda: a @ foreign):
                with pytest.raises(InferenceError):
                    operation()
        for shape in ((-3, -4), (True, 12), [3, 4]):
            with pytest.raises(InferenceError):
                Tensor(a._buffer, shape)
        with pytest.raises(InferenceError):
            Tensor(a._buffer, (5, 4))
    assert runtime.active_bytes == 0


def test_close_invalidates_owned_tensors_and_is_idempotent(runtime):
    a = runtime.tensor(np.ones((2, 2), dtype=np.float32))
    b = runtime.tensor(np.ones((2, 2), dtype=np.float32))
    a.close()
    for operation in (a.numpy, a.silu, a.transpose, a.__enter__, lambda: b + a):
        with pytest.raises(ClosedError):
            operation()
    runtime.close()
    assert runtime.active_bytes == 0
    for operation in (b.numpy, b.silu, lambda: runtime.tensor(np.ones(2, np.float32))):
        with pytest.raises(ClosedError):
            operation()
    a.close()
    b.close()
    runtime.close()


def test_failed_commands_release_outputs(runtime, monkeypatch):
    with runtime.tensor(np.ones((2, 2), np.float32)) as a:
        resident = runtime.active_bytes

        def fail_dispatch(*args, **kwargs):
            raise InferenceError()

        with monkeypatch.context() as patch:
            patch.setattr(runtime, "_dispatch", fail_dispatch)
            for operation in (a.silu, a.transpose, lambda: a + a, lambda: a @ a):
                with pytest.raises(InferenceError):
                    operation()
                assert runtime.active_bytes == resident
        np.testing.assert_array_equal((a + a).numpy(), np.full((2, 2), 2, np.float32))


def test_nested_commands_cannot_destroy_runtime(runtime):
    with runtime.tensor(np.ones((2, 2), np.float32)) as a:
        with runtime.command():
            for operation in (
                runtime.close,
                a.silu,
                lambda: runtime.tensor(np.ones(2, np.float32)),
            ):
                with pytest.raises(InferenceError):
                    operation()
        np.testing.assert_array_equal(a.numpy(), np.ones((2, 2), np.float32))


def test_parallel_tensor_calls_are_serialized_and_release_memory(runtime):
    data = np.arange(15, dtype=np.float32).reshape(3, 5)
    with runtime.tensor(data) as a:

        def compute(_):
            with (a + a).transpose() as result:
                return result.numpy()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(compute, range(12)))
        for result in results:
            np.testing.assert_array_equal(result, (data * 2).T)
        assert runtime.active_bytes == data.nbytes
    assert runtime.active_bytes == 0
