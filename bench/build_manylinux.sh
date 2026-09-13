#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/python/cp312-cp312/bin:$PATH"
mkdir -p manylinux-evidence
git rev-parse HEAD > manylinux-evidence/source-commit.txt
python -VV > manylinux-evidence/python-build.txt
getconf GNU_LIBC_VERSION > manylinux-evidence/glibc-build.txt
gcc --version > manylinux-evidence/gcc.txt
cmake --version > manylinux-evidence/cmake.txt
patchelf --version > manylinux-evidence/patchelf.txt
cp /etc/os-release manylinux-evidence/os-release
python -m pip install build==1.3.0 scikit-build-core==0.11.6 pybind11==3.0.1 ninja==1.13.2 auditwheel==6.8.2 numpy==2.4.4
python -m pip freeze > manylinux-evidence/build-environment.txt
python -m build --wheel --sdist --no-isolation
cp -a dist manylinux-evidence/original-dist
python -m auditwheel show dist/*.whl > manylinux-evidence/auditwheel-before.txt
python -m auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir wheelhouse dist/*.whl > manylinux-evidence/auditwheel-repair.txt 2>&1
python -m auditwheel show wheelhouse/*.whl > manylinux-evidence/auditwheel-after.txt
# Keep original wheel and repaired wheel separate for provenance.
python - <<'PY'
from pathlib import Path
import shutil
original = list(Path('dist').glob('*.whl'))
repaired = list(Path('wheelhouse').glob('*.whl'))
assert len(original) == len(repaired) == 1
original[0].unlink()
shutil.copy2(repaired[0], Path('dist') / repaired[0].name)
PY
python -m venv "$RUNNER_TEMP/manylinux-runtime"
export PATH="$RUNNER_TEMP/manylinux-runtime/bin:$PATH"
export LD_LIBRARY_PATH=""
python -m pip install numpy==2.4.4
python -m pip install --no-deps dist/*.whl
python bench/ci_wheel_check.py
python bench/check_manylinux_runtime.py
python -m pip install pytest==9.1.1 opencv-python==4.13.0.92 numpy==2.4.4
python -m pytest -q tests/test_parity.py tests/test_resize_roi.py tests/test_gpu_series_audit.py tests/test_lifecycle_coordinator.py tests/test_lifecycle_gpu_audit.py --junitxml=dist/core-tests.xml
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.4 pillow==12.1.1 opencv-python==4.13.0.92 pi-heif==1.4.0 https://github.com/ultralytics/ultralytics/archive/795a556942a12fe0124cf767888194a1d0b83e2e.tar.gz
python -m pytest -q tests/test_integration.py tests/test_training.py tests/test_persistent_format.py tests/test_shared_collate.py --junitxml=dist/integration-tests.xml
python -m pip freeze > dist/final-environment.txt
