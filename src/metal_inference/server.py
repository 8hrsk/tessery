"""Bounded, loopback-only HTTP service for trusted local applications."""

import asyncio
import hmac
import math
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, cast

from .api import EmbeddingModel
from .errors import ConfigurationError, EmbeddingError, InvalidInputError
from .json_codec import dumps, loads

BODY_LIMIT = 2 * 1024 * 1024


class EmbeddingServer(ThreadingMixIn, HTTPServer):
    """At most max_connections worker threads; model admission is separately bounded.

    The caller owns the model. server_close waits for request workers; close the
    model afterwards to drain any GPU command whose response timed out.
    """

    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True
    request_queue_size = 16

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        port: int = 8765,
        max_connections: int = 16,
        request_timeout: float = 30.0,
        token: str | None = None,
    ) -> None:
        if (
            type(port) is not int
            or not 0 <= port <= 65535
            or type(max_connections) is not int
            or not 1 <= max_connections <= 64
            or isinstance(request_timeout, bool)
            or not isinstance(request_timeout, int | float)
            or not math.isfinite(request_timeout)
            or not 0 < request_timeout <= 3600
            or (
                token is not None
                and (
                    not isinstance(token, str)
                    or not 16 <= len(token) <= 4096
                    or not token.isascii()
                    or any(not 33 <= ord(c) <= 126 for c in token)
                )
            )
        ):
            raise ConfigurationError()
        self.model = model
        self.request_timeout = float(request_timeout)
        self.token = token
        self._slots = threading.BoundedSemaphore(max_connections)
        super().__init__(("127.0.0.1", port), _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                body = b'{"error":{"code":"overloaded"}}'
                request.sendall(
                    b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\n"
                    b"Content-Type: application/json\r\nRetry-After: 1\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Never write request bodies, authorization or stack traces to stderr.
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Tessery"
    sys_version = ""

    @property
    def engine_server(self) -> EmbeddingServer:
        return cast(EmbeddingServer, self.server)

    def setup(self) -> None:
        self.request.settimeout(min(10.0, self.engine_server.request_timeout))
        super().setup()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _reply(self, status: int, payload: object) -> None:
        self.close_connection = True
        body = dumps(payload, limit=16 * 1024 * 1024)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        if status == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._reply(code, {"error": {"code": "http_error"}})

    def handle_expect_100(self) -> bool:
        self._reply(417, {"error": {"code": "expectation_failed"}})
        return False

    def _allowed(self) -> bool:
        port = self.engine_server.server_port
        hosts = self.headers.get_all("Host", [])
        if (
            len(hosts) != 1
            or hosts[0] not in {f"127.0.0.1:{port}", f"localhost:{port}"}
            or self.headers.get("Origin") is not None
        ):
            self._reply(403, {"error": {"code": "forbidden"}})
            return False
        token = self.engine_server.token
        if token is not None:
            auth = self.headers.get_all("Authorization", [])
            if len(auth) != 1 or not hmac.compare_digest(
                auth[0].encode("utf-8"), ("Bearer " + token).encode()
            ):
                self._reply(401, {"error": {"code": "unauthorized"}})
                return False
        return True

    def do_GET(self) -> None:
        if not self._allowed():
            return
        model = self.engine_server.model
        if self.path == "/health":
            health = model.health()
            self._reply(200 if health.ready else 503, asdict(health))
        elif self.path == "/v1/models":
            self._reply(200, {"object": "list", "data": [asdict(model.descriptor)]})
        elif self.path == "/v1/memory":
            self._reply(200, asdict(model.memory_stats()))
        else:
            self._reply(404, {"error": {"code": "not_found"}})

    def do_POST(self) -> None:
        if not self._allowed():
            return
        if self.path != "/v1/embeddings":
            self._reply(404, {"error": {"code": "not_found"}})
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if (
                self.headers.get("Transfer-Encoding") is not None
                or len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdecimal()
                or len(lengths[0]) > 10
            ):
                raise InvalidInputError()
            length = int(lengths[0])
            if length > BODY_LIMIT:
                self._reply(413, {"error": {"code": "input_too_large"}})
                return
            content_types = self.headers.get_all("Content-Type", [])
            if (
                len(content_types) != 1
                or content_types[0].split(";", 1)[0].strip().lower() != "application/json"
            ):
                self._reply(415, {"error": {"code": "unsupported_media_type"}})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise InvalidInputError()
            data = loads(raw, limit=BODY_LIMIT)
            model = self.engine_server.model
            if (
                not isinstance(data, dict)
                or not set(data) <= {"input", "model", "dimensions"}
                or "input" not in data
                or ("model" in data and data["model"] != model.descriptor.model_id)
            ):
                raise InvalidInputError()
            texts = [data["input"]] if isinstance(data["input"], str) else data["input"]

            async def encode() -> Any:
                return await asyncio.wait_for(
                    model.encode_async(texts, dimensions=data.get("dimensions")),
                    timeout=self.engine_server.request_timeout,
                )

            vectors = asyncio.run(encode())
            self._reply(
                200,
                {
                    "object": "list",
                    "model": model.descriptor.model_id,
                    "compatibility_id": model.descriptor.compatibility_id,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": vector.tolist()}
                        for i, vector in enumerate(vectors)
                    ],
                },
            )
        except TimeoutError:
            self._reply(504, {"error": {"code": "deadline_exceeded"}})
        except EmbeddingError as error:
            self._reply(error.http_status, {"error": {"code": error.code}})
        except ConnectionError:
            self.close_connection = True
        except Exception:
            self._reply(500, {"error": {"code": "inference_failed"}})
