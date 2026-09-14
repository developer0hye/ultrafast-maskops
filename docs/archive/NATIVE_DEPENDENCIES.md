# Native dependency inventory and notice packaging

The 2026-09-13 linker inspection found the following dependencies in the current
Release builds. `native-dependency-inventory.json` records the observed M2 link
line and hashes of candidate notice sources. It is a partial inventory, not a
completed source-license audit or a verified wheel manifest.

| Component | Observed linkage | Notice source to preserve |
|---|---|---|
| OpenCV 4.13.0 core/imgproc | Static, both hosts | Top-level LICENSE, legacy BSD notice, embedded third-party notices |
| zlib | Static, both hosts | `3rdparty/zlib/LICENSE` in the pinned OpenCV archive |
| Carotene / tegra_hal | Static, M2 | BSD copyright/license headers under `hal/carotene` |
| KleidiCV 0.7.0, HAL and thread support | Static, M2 | Downloaded source's `LICENSES/Apache-2.0.txt` and copyright notices |
| Apple Accelerate | System framework, M2 | Record as a system dependency; it is not copied into the wheel |
| pybind11 3.0.1 | Headers compiled into the extension | Installed distribution's `licenses/LICENSE` |
| SoftFloat / FDLIBM | Source embedded in OpenCV core | `modules/core/src/softfloat.cpp` and SoftFloat COPYING file |
| DLPack | Headers available to OpenCV core | `3rdparty/dlpack/LICENSE`; inspect compiled usage |

The Linux linker line contains OpenCV core/imgproc, zlib and the system dl/m/
pthread/rt libraries; it does not contain the ARM HAL libraries above.

## Implemented notice collection

`bench/collect_native_notices.py` reads the observed Ninja compiler dependency
graphs. It collects copyright/license comment blocks from all selected compiled
OpenCV, KleidiCV and pybind11 source/header files, conservatively including
archive objects the final linker may discard. Across the M2 and Linux graphs,
584 distinct files produced 144 unique notice blocks. The 375 Linux source
files were independently hashed on the server and matched the collected bytes.
System/toolchain/Python headers and generated build files are explicitly outside
this collection's scope. DLPack's license is included conservatively even though
its headers did not appear in these graphs.

The `licenses/` directory now contains the collected notices, seven upstream
license texts and a compact manifest. This retains the SoftFloat and FDLIBM
notices as well as the per-file copyright holders. CMake installs the directory
next to the private extension's existing OpenCV license. Collection provenance,
source hashes and compressed compiler graphs are retained under `docs/validation`.

The new M2 wheel and sdist preserve all nine added files byte-for-byte. A new
NumPy-only environment imported the installed wheel and executed a native mask
smoke without Python cv2 or Ultralytics. After adding the test oracle, all 120
core tests passed. Wheel RECORD and installed runtime/notice bytes also passed
the archive audit. The rebuilt compiler graph's 573 selected source files are
covered by the recorded source hashes. See `mask-notice-artifacts-m2.json` and
`mask-notices-clean-v1-*` under `docs/validation`.

This particular wheel is tagged `macosx_26_0_arm64`. It proves installation on
the current M2 host, not older macOS compatibility. The prior GPU benchmark
wheels remain unchanged and retain their original identities.

Before distribution, finish the system/toolchain and embedded-notice review,
repeat the inventory and artifact checks for Windows and the supported macOS
baseline, and complete Rust dependency notice bundling in ultrafast-yolo-dataset.
The project LICENSE and Ultralytics NOTICE are separately preserved in wheel
metadata. The collection and passing byte checks do not establish legal
completeness; the remaining review is still a release gate.
