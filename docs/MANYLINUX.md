# Linux distribution compatibility qualification

The first platform matrix built wheels on Ubuntu 24.04. Its successful imports
establish that environment, not an older glibc floor. The new build uses PyPA's
[manylinux environment](https://github.com/pypa/manylinux) and
[auditwheel](https://github.com/pypa/auditwheel) to build, inspect and repair a
wheel for `manylinux_2_28_x86_64` before running the actual installed tests.

The container image is pinned to
`quay.io/pypa/manylinux_2_28_x86_64@sha256:53390351aeb4688114b02c36a23b3e6ce1166ee9b7afc5df1a4f776354fc764c`.
The registry manifest digest and linked image-config digest were checked before
use; both original JSON files are retained under `docs/validation/`. This is the
amd64 image created 2026-09-05, not a floating tag at execution time.

## Pilot protocol

[Current pilot run](https://github.com/developer0hye/ultrafast-maskops/actions/runs/34747087156)
selects Python 3.12 only. It is not the complete planned Python 3.10–3.13
portable-wheel matrix. The independent hosted compatibility matrix retains its
separate source revisions, platforms, results and failures.

`bench/build_manylinux.sh` retains compiler, Python, glibc and build-environment
records, the original wheel/sdist, repaired wheel, and auditwheel before/repair/
after logs. Auditwheel is pinned to 6.8.2. It then creates a fresh virtual
environment, clears the GCC toolset `LD_LIBRARY_PATH`, installs the repaired
wheel and checks its RECORD, installed bytes and bundled notices. The runtime
check requires glibc 2.28, verifies the target wheel tag, saves ELF linkage and
symbol versions, and rejects new runtime payload files that would need a new
notice review. ZIP directory entries are not payload files.

Core and pinned-framework tests must then pass; their JUnit files and final
package environment are retained even when a later check fails. Original and
repaired wheels stay separate. Neither file naming nor auditwheel alone proves
functional parity. These hosted checks are compatibility evidence, not training
or startup performance measurements. CPU-feature requirements of NumPy/private
OpenCV, old-kernel behavior and other distributions are not independently
qualified by running inside this container on a current hosted kernel.

## Preserved setup failures

The [first executed attempt](validation/manylinux-v1-bootstrap-failure.json)
stopped at `git rev-parse` because checkout's temporary Git configuration was not
visible in the subsequent container shell. No compilation or tests ran. Its
complete job log/API, empty initial artifact and exact script/workflow sources
are preserved. The correction trusts only `GITHUB_WORKSPACE` inside the disposable
container, after checking that it is the current directory; it does not change
host Git configuration or trust arbitrary directories.

Final artifact readback, all-Python portable qualification and release gates
remain open. No public publication is performed by this workflow.
