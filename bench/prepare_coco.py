"""Create a reproducible Segmentation fixture from the official COCO val2017 archives.

No downloads, evaluation or training are performed by this command. The output
must be new: conversion appends labels in upstream code, so reusing a directory
could silently duplicate annotations.
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from coco_loader import fingerprint, validate_profile
from ultralytics.data import converter

ASSETS = {
    "val2017.zip": {
        "url": "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    },
    "annotations_trainval2017.zip": {
        "url": "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    },
}
CONVERTER_SHA = "e3ab9ecb67cd52a69b2e452de39b3ffc7c18d7eaba72838e3d682ea764acea54"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_profile()
    assert file_hash(Path(converter.__file__)) == CONVERTER_SHA, "unsupported conversion source"
    for name, expected in ASSETS.items():
        assert file_hash(args.archives / name) == expected["sha256"], f"archive hash mismatch: {name}"
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    with zipfile.ZipFile(args.archives / "annotations_trainval2017.zip") as archive:
        archive.extract("annotations/instances_val2017.json", source)
    annotation = source / "annotations/instances_val2017.json"
    document = json.loads(annotation.read_bytes())
    images = sorted(document["images"], key=lambda row: row["file_name"])
    assert len(images) == 5000 and len({row["file_name"] for row in images}) == 5000
    with zipfile.ZipFile(args.archives / "val2017.zip") as archive:
        for row in images:
            name = row["file_name"]
            assert len(name) == 16 and name[:12].isdigit() and name.endswith(".jpg")
            archive.extract("val2017/" + name, source / "images")
    print("Verified and extracted all 5000 official validation images", flush=True)
    input_manifest = {
        "kind": "ultrafast-yolo-coco-source-v1",
        "archives": ASSETS,
        "annotation_sha256": file_hash(annotation),
        "ultralytics_commit": "795a556942a12fe0124cf767888194a1d0b83e2e",
        "converter_sha256": CONVERTER_SHA,
        "images": [
            {
                "name": row["file_name"],
                "height": row["height"],
                "width": row["width"],
                "sha256": file_hash(source / "images/val2017" / row["file_name"]),
            }
            for row in images
        ],
        "annotations": len(document["annotations"]),
        "crowd_annotations": sum(bool(row.get("iscrowd")) for row in document["annotations"]),
        "scope": "all COCO val2017 images; validation corpus for pipeline measurements, no training accuracy claim",
    }
    source_manifest = root / "source-manifest.json"
    source_manifest.write_text(json.dumps(input_manifest, indent=2) + "\n")
    for task in ("segment",):
        target = root / task
        converter.convert_coco(str(source / "annotations"), str(target), use_segments=task == "segment")
        destination = target / "images/val2017"
        destination.mkdir()
        for row in images:
            # Independent image copies prevent JPEG repair in one fixture from
            # changing the other fixture or the extracted reference source.
            shutil.copyfile(source / "images/val2017" / row["file_name"], destination / row["file_name"])
        labels = list((target / "labels/val2017").glob("*.txt"))
        manifest = {
            "kind": "ultrafast-yolo-coco-v1",
            "task": task,
            "count": 5000,
            "label_files": len(labels),
            "label_rows": sum(len(path.read_text().splitlines()) for path in labels),
            "fingerprint": fingerprint(target),
            "source_manifest_sha256": file_hash(source_manifest),
            "image_directory": "images/val2017",
            "legacy_cache": "labels/val2017.cache",
            "conversion": "unmodified pinned Ultralytics convert_coco, cls91to80=True, lvis=False; crowd and invalid boxes excluded, multipart segments merged by upstream, missing labels preserved",
        }
        (target / "coco.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
