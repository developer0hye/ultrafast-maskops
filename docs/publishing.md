# Releases

Wheels are built by `.github/workflows/release.yml` with cibuildwheel for
CPython 3.10–3.13 on Linux x86-64 and aarch64 (manylinux_2_28), macOS arm64
and x86-64 (cross-compiled on the Apple silicon runner), and Windows x86-64,
plus a source distribution. Every wheel except the cross-compiled macOS x86-64
one runs `tests/test_parity.py` and `tests/test_sampled.py` against the freshly
built package before it is kept.

## Publishing a version

1. Set `version` in `pyproject.toml` and commit.
2. Tag it: `git tag v0.1.0 && git push origin v0.1.0`.
3. The workflow builds the wheels and the sdist and uploads them to PyPI through
   [Trusted Publishing](https://docs.pypi.org/trusted-publishers/). Configure the
   publisher once on PyPI: owner `developer0hye`, repository
   `ultrafast-maskops`, workflow `release.yml`, environment `pypi`. No API token
   is stored in the repository.
4. `workflow_dispatch` runs the same build without uploading, for a dry run.

The wheel is one small extension (about 0.3 MB) that depends only on NumPy at
run time; nothing is downloaded during the build.
