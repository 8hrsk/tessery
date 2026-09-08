import asyncio
import http.client
import json
import socket
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from metal_inference.api import HealthStatus, MemoryStats, ModelDescriptor
from metal_inference.errors import ConfigurationError, InvalidInputError, OverloadError
from metal_inference.server import EmbeddingServer


class Model:
    descriptor = ModelDescriptor()
    mode = "ok"
    entered = threading.Event()
    canceled = threading.Event()

    async def encode_async(self, texts, *, dimensions):
        self.entered.set()
        if self.mode == "overload":
            raise OverloadError()
        if self.mode == "error":
            raise RuntimeError("private prompt should not appear")
        if self.mode == "slow":
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.canceled.set()
                raise
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise InvalidInputError()
        return np.ones((len(texts), dimensions or 32), np.float32)

    def health(self):
        return HealthStatus(True, True, self.descriptor.compatibility_id)

    def memory_stats(self):
        return MemoryStats(100, 200, 20)


@contextmanager
def serving(**kwargs):
    model = Model()
    model.entered = threading.Event()
    model.canceled = threading.Event()
    with EmbeddingServer(model, port=0, **kwargs) as server:
        worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        worker.start()
        try:
            yield model, server
        finally:
            server.shutdown()
            worker.join(timeout=5)
            assert not worker.is_alive()


def request(server, method="POST", path="/v1/embeddings", body=None, headers=None):
    client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        client.request(
            method,
            path,
            body=body if body is not None else '{"input":"hello"}',
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def test_metadata_vectors_and_errors():
    with serving() as (model, server):
        assert server.server_address[0] == "127.0.0.1"
        status, payload = request(server)
        assert status == 200 and len(payload["data"][0]["embedding"]) == 32
        assert request(server, "GET", "/health")[1]["ready"]
        assert (
            request(server, "GET", "/v1/models")[1]["data"][0]["model_id"]
            == model.descriptor.model_id
        )
        assert request(server, "GET", "/v1/memory")[1]["cache_bytes"] == 20
        assert request(server, "GET", "/missing")[0] == 404
        assert request(server, path="/missing")[0] == 404
        assert request(server, "DELETE")[0] == 501
        for body in (
            "{}",
            '{"input":null}',
            '{"input":"x","model":"bad"}',
            '{"input":NaN}',
            '{"input":"x","input":"y"}',
        ):
            assert request(server, body=body)[0] == 400
        assert request(server, headers={"Content-Type": "text/plain"})[0] == 415
        assert request(server, headers={"Content-Length": "2097153"})[0] == 413
        assert request(server, headers={"Transfer-Encoding": "chunked"})[0] == 400
        model.mode = "overload"
        assert request(server)[0] == 429
        model.mode = "error"
        assert request(server) == (500, {"error": {"code": "inference_failed"}})


def test_auth_origin_host_and_timeout():
    token = "test-token-0123456789"
    with serving(token=token, request_timeout=0.1) as (model, server):
        auth = {"Authorization": "Bearer " + token}
        assert request(server)[0] == 401
        assert request(server, headers={"Authorization": "Bearer é"})[0] == 401
        assert request(server, headers=auth)[0] == 200
        assert request(server, headers={**auth, "Origin": "http://example.com"})[0] == 403
        assert request(server, headers={**auth, "Host": "example.com"})[0] == 403
        model.mode = "slow"
        assert request(server, headers=auth)[0] == 504
        assert model.canceled.wait(1)
        model.mode = "ok"
        assert request(server, headers=auth)[0] == 200


def test_connection_limit_and_duplicate_headers():
    with serving(max_connections=1, request_timeout=1) as (model, server):
        # Hold the single worker on a body read, then verify no extra worker is spawned.
        held = socket.create_connection(server.server_address, timeout=2)
        held.sendall(
            (
                f"POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                "Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
            ).encode()
        )
        # Wait until the held socket has consumed the single worker slot.
        acquired = False
        for _ in range(1000):
            if not server._slots.acquire(blocking=False):
                acquired = True
                break
            server._slots.release()
            threading.Event().wait(0.001)
        assert acquired
        try:
            assert request(server)[0] == 429
        finally:
            held.close()
    with serving() as (_, server):
        with socket.create_connection(server.server_address, timeout=2) as wire:
            wire.sendall(
                (
                    f"POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
                    "Content-Length: 0\r\nContent-Length: 0\r\n\r\n"
                ).encode()
            )
            assert b"400" in wire.recv(4096)


@pytest.mark.parametrize(
    "kwargs",
    [{"port": -1}, {"max_connections": 0}, {"token": "short"}, {"request_timeout": float("nan")}],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ConfigurationError):
        EmbeddingServer(SimpleNamespace(), **kwargs)
