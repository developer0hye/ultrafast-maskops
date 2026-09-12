# Native dependency inventory in progress

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
pthread/rt libraries; it does not contain the ARM HAL libraries above. The wheel
configuration currently installs only the OpenCV top-level license explicitly.
Copying only that file is not the final notice-bundling implementation.

Before distribution, audit the actual compiled source/header graph, collect all
applicable notices, add their installation rules, then inspect wheel and sdist
contents in clean environments. Repeat the inventory for Windows and for Rust
dependencies in ultrafast-yolo-dataset. Also preserve the project LICENSE and
Ultralytics attribution. This work remains a release gate.
