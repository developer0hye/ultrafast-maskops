"""Check an installed wheel and all bundled notices, including on Windows."""

import importlib
import importlib.util
import json
import platform
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path
from zipfile import ZipFile

from audit_wheel import audit


def main():
    root = Path(__file__).resolve().parents[1]
    packages = [p for p in ("ultrafast_maskops", "ultrafast_yolo_dataset") if (root / "python" / p).is_dir()]
    assert len(packages) == 1
    package = importlib.import_module(packages[0])
    installed = Path(package.__file__).resolve().parent
    assert installed.is_relative_to(Path(sys.prefix).resolve()), "must import the installed wheel"
    if packages[0] == "ultrafast_maskops":
        import numpy as np

        assert importlib.util.find_spec("cv2") is None and importlib.util.find_spec("ultralytics") is None
        polygon = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)
        packed = package.PackedPolygons.from_segments([polygon])
        mask, order = package.Rasterizer().overlap((16, 16), packed, 1)
        assert mask.dtype == np.uint8 and mask.shape == (16, 16) and mask.sum() == 81
        assert order.tolist() == [0]
    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    result = audit(wheels[0], Path(sysconfig.get_path("platlib")))
    assert result["clean"], result["errors"]
    source_notices = (
        root / "licenses" if packages[0] == "ultrafast_maskops" else root / "python" / packages[0] / "licenses"
    )
    source_files = sorted(p for p in source_notices.rglob("*") if p.is_file())
    assert source_files, "third-party notice bundle is missing"
    with ZipFile(wheels[0]) as archive, tarfile.open(sdists[0]) as sdist:
        prefix = sdist.getnames()[0].split("/")[0]
        for path in source_files:
            name = path.relative_to(source_notices).as_posix()
            assert archive.read(f"{packages[0]}/licenses/{name}") == path.read_bytes(), name
            assert (installed / "licenses" / name).read_bytes() == path.read_bytes(), name
            member = f"{prefix}/{path.relative_to(root).as_posix()}"
            assert sdist.extractfile(member).read() == path.read_bytes(), member
    result.update(
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        notice_files_checked=len(source_files),
        package=str(installed),
        installed_packages=subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    )
    (root / "dist/wheel-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Installed wheel and {len(source_files)} notice files verified on {platform.machine()}")


if __name__ == "__main__":
    main()
