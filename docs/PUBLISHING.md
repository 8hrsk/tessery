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

## Later: publish from GitHub Actions without a permanent token

[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) lets PyPI trust a
specific workflow and issue short-lived credentials through OIDC. Register the
owner `8hrsk`, repository `tessery`, the actual publishing workflow filename,
and optionally a GitHub environment such as `pypi`. A first upload can use a
[pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

The future workflow should build and validate native artifacts on macOS arm64,
then publish precisely those artifacts in a job with `id-token: write`. Configure
an explicit release trigger and environment protections. The existing `ci.yml`
checks changes and builds a source archive; it does not publish. No publishing
workflow, account, token or pending publisher was created by this change.

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
