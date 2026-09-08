# Publishing Tessery to PyPI

`pip` installs packages; PyPI is the public registry. `uv publish` or `twine upload`
performs the upload, much like `npm publish`. Tessery's distribution name and public
Python import are both `tessery`, starting with 0.5.1a1.
The first release is now [published on PyPI](https://pypi.org/project/tessery/0.5.1a1/);
see [the release verification](PYPI_RELEASE.md).

The project is registered as `tessery`. Import names and distribution names are
independent, but this project uses the same public name for both.
See the [PyPA packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## One-time setup

Create a [PyPI account](https://pypi.org/account/register/), verify its email and
configure [two-factor authentication](https://pypi.org/help/#twofa). For a first
manual upload, create an API token with the required account scope. Once the project
exists, use a project-scoped token. PyPI does not accept account passwords for
uploads; the [uv publishing guide](https://docs.astral.sh/uv/guides/package/) describes
token and Trusted Publishing authentication.

TestPyPI is a separate registry with separate accounts and credentials. It is useful
for rehearsing publication without creating a production release. Configure it
separately from PyPI and do not mix up the tokens.

## Build and verify on Apple Silicon

```sh
uv sync --locked --group dev
uv run ruff check .
uv run mypy
METAL_INFERENCE_TEST=1 uv run pytest --cov
uv run python tools/check_dependencies.py --installed
uv run python tools/check_inputs.py
uv build --no-sources --out-dir dist/pypi-0.5.1a1
```

Native tests use the existing local models. Build with
`METAL_INFERENCE_PORTABLE_TESTS` unset: that flag is solely for portable CI tests.
The real native wheel must be built on macOS arm64 with Xcode Command Line Tools;
its tag is `py3-none-macosx_14_0_arm64`, not `py3-none-any`.

Expected artifacts:

* `tessery-0.5.1a1-py3-none-macosx_14_0_arm64.whl`
* `tessery-0.5.1a1.tar.gz`

Install the wheel into a separate environment and exercise imports and inference.
A wheel user needs Apple Silicon, macOS 14+ and Python 3.12+, but no compiler.
Installing from the source archive requires the compiler. Weights remain external
and are never bundled or downloaded automatically. No Windows/Linux inference
support is implied by uploading a source archive.

## Upload the selected release

Make the PyPI token available to uv through `UV_PUBLISH_TOKEN` using your local
credential handling. Do not commit it or paste it into issue/CI logs. Then run:

```sh
uv publish dist/pypi-0.5.1a1/*.whl dist/pypi-0.5.1a1/*.tar.gz
```

Use only that release's wheel/source files, not all historical contents of `dist`
and not SPDX sidecars. Before production, the same files can be uploaded to
TestPyPI using its separate token and endpoint:

```sh
uv publish --publish-url https://test.pypi.org/legacy/ \
  dist/pypi-0.5.1a1/*.whl dist/pypi-0.5.1a1/*.tar.gz
```

The traditional alternative is `python -m build` plus `python -m twine upload`.
These are publishing tools, not runtime dependencies of Tessery. The
[PyPA tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
also documents TestPyPI and Twine.

After production publication, a user can install the alpha explicitly:

```sh
python -m pip install --pre tessery
python -c 'from tessery import EmbeddingModel, list_profiles; print(list_profiles())'
```

PyPI release files cannot be replaced with changed bytes under the same filename.
For a code change, choose a new version and update `pyproject.toml`, the engine's
`__version__`, the reviewed root package in `policy/dependency-licenses.json`,
and the SPDX tool's version; regenerate `uv.lock` and rebuild. Do not delete and
re-upload the same filename. See [PyPI file reuse rules](https://pypi.org/help/#file-name-reuse).

## GitHub Actions with Trusted Publishing

The repository now contains [publish.yml](../.github/workflows/publish.yml).
It is triggered manually with an existing tag such as `v0.6.0a1`; ordinary pushes
and pull requests cannot publish. The checked-out package version must match the
tag. Native checks, both pinned models, artifact layout validation and isolated
wheel inference must pass before the Ubuntu publishing job receives the artifacts.
That job uses the pinned PyPA action with OIDC and attestations, without a stored
API token. See [PyPI's publisher documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

One-time configuration still required:

1. In the existing PyPI `tessery` project's **Publishing** settings, add a GitHub
   publisher: owner `8hrsk`, repository `tessery`, workflow `publish.yml`,
   environment `pypi`.
2. Configure the GitHub `pypi` environment with the desired reviewer and release-tag
   protections. No environment protection was configured automatically.
3. Provision the dedicated runner with labels `self-hosted`, `macOS`, `ARM64`,
   `metal-inference`, Python 3.12.13, uv 0.10.9, Xcode CLT and both verified local
   Qwen/BGE packs. Model paths can be configured with the existing
   `METAL_INFERENCE_MODEL_DIR`, `METAL_INFERENCE_BGE_MODEL_DIR` and
   `METAL_INFERENCE_BGE_PROFILE_FILE` runner environment variables.

After qualification, create/push a version tag and dispatch **Publish verified
Tessery release** with that tag. Only `.whl` and `.tar.gz` from that run are
uploaded. PyPI rejects reuse of an existing release filename; this workflow does
not silently skip duplicates. It cannot publish until the PyPI identity is registered.
The 0.6 candidate has been built and checked locally, but not published or tagged.

## Migration from the local `metal-inference` distribution

The new wheel includes the existing implementation and compatibility imports;
it has no dependency on a separately installed `metal-inference` distribution.
If the old distribution is installed in an environment, uninstall it before
installing the new Tessery wheel, because their implementation files overlap:

```sh
python -m pip uninstall metal-inference
python -m pip install /absolute/path/tessery-0.5.1a1-py3-none-macosx_14_0_arm64.whl
```

`from metal_inference import EmbeddingModel` and the old `metal-inference` CLI
continue to work after installing Tessery. New code should use `tessery`,
`tessery.errors`, and `tessery.server`. Implementation module paths and existing
embedding compatibility IDs remain unchanged; persisted indexes do not need
rebuilding just because the package has a new public name.
