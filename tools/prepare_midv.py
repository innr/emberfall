#!/usr/bin/env python3
"""Convert MIDV-2020 passport annotations to a YOLO MRZ detector dataset.

The converter consumes the original MIDV-2020 layout and merges the two VIA
regions named ``mrz_line0`` and ``mrz_line1`` into one bounding box.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - clear CLI error
    raise SystemExit("Pillow is required: pip install Pillow") from exc


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def merge(self, other: "Box") -> "Box":
        return Box(min(self.xmin, other.xmin), min(self.ymin, other.ymin),
                   max(self.xmax, other.xmax), max(self.ymax, other.ymax))

    def clip(self, width: int, height: int) -> "Box":
        return Box(max(0.0, min(self.xmin, width)),
                   max(0.0, min(self.ymin, height)),
                   max(0.0, min(self.xmax, width)),
                   max(0.0, min(self.ymax, height)))


def region_box(region: dict[str, Any]) -> Box | None:
    shape = region.get("shape_attributes", {})
    name = shape.get("name", "rect")
    if name == "rect":
        x, y = float(shape.get("x", 0)), float(shape.get("y", 0))
        w, h = float(shape.get("width", 0)), float(shape.get("height", 0))
        return Box(x, y, x + w, y + h) if w > 0 and h > 0 else None
    if name in {"polygon", "polyline"}:
        xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
        if xs and ys and len(xs) == len(ys):
            return Box(float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
    return None


def annotation_boxes(annotation_file: Path) -> dict[str, Box]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    records = data.get("_via_img_metadata", data)
    result: dict[str, Box] = {}
    for record in records.values() if isinstance(records, dict) else records:
        fields: dict[str, Box] = {}
        for region in record.get("regions", []):
            attrs = region.get("region_attributes", {})
            field = attrs.get("field_name")
            if field in {"mrz_line0", "mrz_line1"}:
                box = region_box(region)
                if box is not None:
                    fields[field] = box
        if len(fields) == 2:
            result[record.get("filename", "")] = fields["mrz_line0"].merge(fields["mrz_line1"])
    return result


def find_images(root: Path, filename: str) -> list[Path]:
    exact = list((root / "dataset" / "images").rglob(filename))
    if exact:
        return exact
    return list((root / "dataset" / "images").rglob(Path(filename).name))


def yolo_line(box: Box, width: int, height: int) -> str:
    b = box.clip(width, height)
    xc, yc = (b.xmin + b.xmax) / 2 / width, (b.ymin + b.ymax) / 2 / height
    bw, bh = (b.xmax - b.xmin) / width, (b.ymax - b.ymin) / height
    if bw <= 0 or bh <= 0:
        raise ValueError(f"invalid box after clipping: {box}")
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def convert(root: Path, out: Path, val_ratio: float = 0.2, seed: int = 20260918,
            types: Iterable[str] = ("aze_passport", "grc_passport", "lva_passport", "srb_passport"),
            limit: int | None = None) -> tuple[int, int]:
    pairs: list[tuple[Path, Box]] = []
    ann_dir = root / "dataset" / "templates" / "annotations"
    for doc_type in types:
        ann_file = ann_dir / f"{doc_type}.json"
        if not ann_file.exists():
            continue
        for filename, box in annotation_boxes(ann_file).items():
            matches = find_images(root, filename)
            pairs.extend((p, box) for p in matches)
    # Avoid duplicate paths when an annotation mirror contains repeated records.
    unique: dict[Path, Box] = {p.resolve(): b for p, b in pairs}
    pairs = sorted(unique.items(), key=lambda item: str(item[0]))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError("No passport images with both mrz_line0 and mrz_line1 were found")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = round(len(pairs) * val_ratio)
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (source, box) in enumerate(items):
            stem = f"{split}_{index:06d}"
            suffix = source.suffix.lower() or ".jpg"
            destination = out / "images" / split / f"{stem}{suffix}"
            shutil.copy2(source, destination)
            with Image.open(source) as image:
                width, height = image.size
            (out / "labels" / split / f"{stem}.txt").write_text(
                yolo_line(box, width, height), encoding="utf-8")
    (out / "classes.txt").write_text("MRZ\n", encoding="utf-8")
    (out / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: MRZ\n",
        encoding="utf-8")
    return len(splits["train"]), len(splits["val"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="MIDV-2020 root containing dataset/")
    parser.add_argument("--out", type=Path, required=True, help="YOLO output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    train, val = convert(args.root, args.out, args.val_ratio, args.seed, limit=args.limit)
    print(f"Converted {train + val} images: train={train}, val={val}")


if __name__ == "__main__":
    main()
