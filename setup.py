"""Build the independent C ABI bridge into the platform wheel, entirely offline."""

import os
import platform
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_ext import build_ext


class BuildMetal(build_ext):
    def get_ext_filename(self, name):
        return os.path.join(*name.split(".")) + ".dylib"

    def build_extension(self, extension):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Native builds require macOS arm64 and Xcode Command Line Tools")
        destination = Path(self.get_ext_fullpath(extension.name))
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
                extension.sources[0],
                "-o",
                str(destination),
            ],
            check=True,
        )


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # The bridge uses ctypes and the C ABI, not the CPython extension ABI.
        return ("py3", "none", "macosx_14_0_arm64")


portable_tests = os.environ.get("METAL_INFERENCE_PORTABLE_TESTS") == "1"
setup(
    ext_modules=[]
    if portable_tests
    else [Extension("metal_inference._native", sources=["src/metal_inference/native/runtime.mm"])],
    cmdclass={} if portable_tests else {"build_ext": BuildMetal, "bdist_wheel": PlatformWheel},
    package_data={"tessery": ["py.typed"], "metal_inference": ["native/kernels.metal", "py.typed"]},
)
