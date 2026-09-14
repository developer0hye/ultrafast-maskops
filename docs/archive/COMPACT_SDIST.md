# Compact source distribution and source-only rebuild

The original hosted source distribution at
`06a26d106ba8dd896061f987a6561521de68e6e1` was **116,015,854 bytes**. Its tree
included **110,188,256 uncompressed bytes of benchmark archives**. The original
GitHub ZIP digest, source pyproject and retained JUnit reports were checked on a
hosted runner, and the small inspection artifact was read back independently.
See the [original inspection receipt](validation/mask-sdist-original-inspection-v1.json).

The candidate at `e86419ef6fc6e7d34a7b3c0ef09859f3607d49b7` excludes
`bench/results/*.tar.gz*` and `bench/results/*.zip` from the installer sdist.
The archives remain in Git. Build/runtime sources, licenses, tests, benchmark
programs and lightweight reports remain in the source package. This uses the
[supported scikit-build source exclusion setting](https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html#configuring-source-file-inclusion).

The new sdist is **6,036,842 bytes, 94.80% smaller**. This is an installer size
improvement; it does not change the identities or conclusions of runtime
benchmarks performed with earlier wheels.

## Verified source rebuild

[Run 34745913090](https://github.com/developer0hye/ultrafast-maskops/actions/runs/34745913090)
completed on Ubuntu 24.04 / CPython 3.12.14:

- **87 required project source files** matched checkout byte for byte, including
  CMake, native C++, Python, licenses, tests and audit/benchmark programs.
- The source archive was unpacked outside the checkout, without Git metadata.
  The wheel was built using that extracted project source. CMake still downloads
  OpenCV 4.13.0 and checks its pinned SHA-256; this is not an offline source kit.
- A fresh virtual environment installed NumPy and the rebuilt wheel. The standalone
  rasterization check ran without cv2 or Ultralytics installed. Wheel RECORD and
  installed payload checks passed; all **9 bundled notice files** matched source,
  sdist, wheel and installed copies.
- OpenCV's Python reference package and pytest were then installed. **289 core
  tests passed with no skips, failures or errors**. This job does not run Torch
  training or the framework integration suite.

The [preservation receipt](validation/mask-sdist-rebuild-v1-preservation.json)
records the original GitHub ZIP digest, both identical source archive copies,
wheel/native hashes, source manifest, JUnit and complete job log. Every outer ZIP
member was read back; the 87 source files, wheel RECORD and notice bytes were also
rechecked locally. The complete 13.3MB artifact is included in the evidence bundle.

This qualifies source-only rebuilding on Linux 3.12. The earlier full platform
matrix at `06a26d1` ended with 11 jobs passing and one Windows 3.13 worker-reset
failure. That failure remains open under the separate `investigate/windows-reset`
branch. A full matrix for the compact packaging revision, original matrix artifact
preservation, portable release repair and broader release gates remain open.
