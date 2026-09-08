"""Explicit offline development build; runtime never installs or compiles itself."""

import platform
import subprocess
from pathlib import Path


def build(destination: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Native engine requires macOS arm64")
    root = Path(__file__).resolve().parents[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "/usr/bin/xcrun",
            "clang++",
            "-std=c++17",
            "-O3",
            "-fobjc-arc",
            "-dynamiclib",
            "-Wl,-install_name,@rpath/_native.dylib",
            "-mmacosx-version-min=14.0",
            "-arch",
            "arm64",
            "-framework",
            "Foundation",
            "-framework",
            "Metal",
            str(root / "src/metal_inference/native/runtime.mm"),
            "-o",
            str(destination),
        ],
        check=True,
    )


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "src/metal_inference/_native.dylib")
