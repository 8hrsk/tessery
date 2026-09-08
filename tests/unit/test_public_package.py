import json
import subprocess
import sys


def test_public_import_without_gpu_or_network():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

def offline(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("import attempted network access")
sys.addaudithook(offline)
import metal_inference.metal as backend

def forbidden():
    raise AssertionError("import initialized Metal")
backend._library = forbidden
from tessery import EmbeddingModel, list_profiles
from tessery.errors import MetalUnavailableError
from tessery.server import EmbeddingServer
from metal_inference import EmbeddingModel as PreviousModel
from metal_inference.errors import MetalUnavailableError as PreviousError
assert EmbeddingModel is PreviousModel
assert MetalUnavailableError is PreviousError
assert callable(EmbeddingServer)
assert "qwen3-embedding-0.6b-dwq" in list_profiles()
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_public_module_cli_profiles():
    result = subprocess.run(
        [sys.executable, "-m", "tessery", "profiles"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["profiles"]) == 2
