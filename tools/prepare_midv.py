#!/usr/bin/env python3
"""Convert MIDV-2020 passport annotations to a YOLO MRZ dataset.

MIDV-2020 stores text-field boxes in template coordinates and document corner
annotations (``doc_quad``) for photos/scans. For non-template images this tool
maps the MRZ rectangle through the document homography before emitting an
axis-aligned YOLO box.
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


PASSPORT_TYPES = ("aze_passport", "grc_passport", "lva_passport", "srb_passport")


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def merge(self, other: "Box") -> "Box":
        return Box(min(self.xmin, other.xmin), min(self.ymin, other.ymin),
                   max(self.xmax, other.xmax), max(self.ymax, other.ymax))

    def corners(self) -> list[tuple[float, float]]:
        return [(self.xmin, self.ymin), (self.xmax, self.ymin),
                (self.xmax, self.ymax), (self.xmin, self.ymax)]

    def clip(self, width: int, height: int) -> "Box":
        return Box(max(0.0, min(self.xmin, width)),
                   max(0.0, min(self.ymin, height)),
                   max(0.0, min(self.xmax, width)),
                   max(0.0, min(self.ymax, height)))


def _solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with pivoted Gaussian elimination."""
    n = len(vector)
    a = [row[:] + [value] for row, value in zip(matrix, vector)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("degenerate homography points")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [value / scale for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor:
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [a[row][-1] for row in range(n)]


def homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    """Return the 3x3 projective transform mapping four source points to dst."""
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("a homography needs four point pairs")
    matrix: list[list[float]] = []
    vector: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        vector.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.append(v)
    return _solve_linear(matrix, vector) + [1.0]


def project(points: Iterable[tuple[float, float]], h: list[float]) -> list[tuple[float, float]]:
    result = []
    for x, y in points:
        denominator = h[6] * x + h[7] * y + h[8]
        if abs(denominator) < 1e-12:
            raise ValueError("homography projects a point at infinity")
        result.append(((h[0] * x + h[1] * y + h[2]) / denominator,
                       (h[3] * x + h[4] * y + h[5]) / denominator))
    return result


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


def region_quad(region: dict[str, Any]) -> list[tuple[float, float]] | None:
    shape = region.get("shape_attributes", {})
    if shape.get("name") not in {"polygon", "polyline"}:
        return None
    xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
    if len(xs) != 4 or len(ys) != 4:
        return None
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


def records(annotation_file: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    values = data.get("_via_img_metadata", data)
    iterable = values.values() if isinstance(values, dict) else values
    return {record.get("filename", ""): record for record in iterable}


def annotation_boxes(annotation_file: Path) -> dict[str, Box]:
    result: dict[str, Box] = {}
    for filename, record in records(annotation_file).items():
        fields: dict[str, Box] = {}
        for region in record.get("regions", []):
            field = region.get("region_attributes", {}).get("field_name")
            if field in {"mrz_line0", "mrz_line1"}:
                box = region_box(region)
                if box is not None:
                    fields[field] = box
        if len(fields) == 2:
            result[filename] = fields["mrz_line0"].merge(fields["mrz_line1"])
    return result


def _root_candidates(root: Path) -> list[Path]:
    return [root, root / "dataset"] if (root / "dataset").is_dir() else [root]


def find_annotation(root: Path, mode: str, doc_type: str) -> Path | None:
    for base in _root_candidates(root):
        candidates = [base / mode / "annotations" / f"{doc_type}.json",
                      base / "annotations" / f"{doc_type}.json"]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


def find_image(root: Path, mode: str, doc_type: str, filename: str) -> Path | None:
    for base in _root_candidates(root):
        candidate = base / mode / "images" / doc_type / filename
        if candidate.is_file():
            return candidate
    matches = [p for p in root.rglob(filename) if doc_type in p.parts and "images" in p.parts]
    return matches[0] if matches else None


def transformed_box(template_box: Box, template_size: tuple[int, int], doc_quad: list[tuple[float, float]]) -> Box:
    width, height = template_size
    h = homography([(0, 0), (width, 0), (width, height), (0, height)], doc_quad)
    points = project(template_box.corners(), h)
    return Box(min(x for x, _ in points), min(y for _, y in points),
               max(x for x, _ in points), max(y for _, y in points))


def image_box(root: Path, mode: str, doc_type: str, filename: str) -> tuple[Path, Box] | None:
    template_ann = find_annotation(root, "templates", doc_type)
    image = find_image(root, mode, doc_type, filename)
    if not template_ann or not image:
        return None
    template_box = annotation_boxes(template_ann).get(filename)
    if template_box is None:
        return None
    if mode == "templates":
        return image, template_box
    mode_ann = find_annotation(root, mode, doc_type)
    template_image = find_image(root, "templates", doc_type, filename)
    if not mode_ann or not template_image:
        return None
    mode_record = records(mode_ann).get(filename)
    if not mode_record:
        return None
    doc_quad = None
    for region in mode_record.get("regions", []):
        if region.get("region_attributes", {}).get("field_name") == "doc_quad":
            doc_quad = region_quad(region)
            break
    if doc_quad is None:
        return None
    with Image.open(template_image) as template:
        box = transformed_box(template_box, template.size, doc_quad)
    return image, box


def yolo_line(box: Box, width: int, height: int) -> str:
    b = box.clip(width, height)
    xc, yc = (b.xmin + b.xmax) / 2 / width, (b.ymin + b.ymax) / 2 / height
    bw, bh = (b.xmax - b.xmin) / width, (b.ymax - b.ymin) / height
    if bw <= 0 or bh <= 0:
        raise ValueError(f"invalid box after clipping: {box}")
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def convert(root: Path, out: Path, val_ratio: float = 0.2, seed: int = 20260918,
            types: Iterable[str] = PASSPORT_TYPES, mode: str = "photo",
            limit: int | None = None) -> tuple[int, int]:
    pairs: list[tuple[Path, Box]] = []
    for doc_type in types:
        template_ann = find_annotation(root, "templates", doc_type)
        if not template_ann:
            continue
        for filename in annotation_boxes(template_ann):
            pair = image_box(root, mode, doc_type, filename)
            if pair:
                pairs.append(pair)
    unique: dict[Path, Box] = {p.resolve(): b for p, b in pairs}
    pairs = sorted(unique.items(), key=lambda item: str(item[0]))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise RuntimeError("No passport images with usable template/doc_quad annotations were found")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = round(len(pairs) * val_ratio)
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (source, box) in enumerate(items):
            stem = f"{split}_{index:06d}"
            destination = out / "images" / split / f"{stem}{source.suffix.lower() or '.jpg'}"
            shutil.copy2(source, destination)
            with Image.open(source) as image:
                size = image.size
            (out / "labels" / split / f"{stem}.txt").write_text(yolo_line(box, *size), encoding="utf-8")
    (out / "classes.txt").write_text("MRZ\n", encoding="utf-8")
    (out / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: MRZ\n", encoding="utf-8")
    return len(splits["train"]), len(splits["val"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("photo", "scan_upright", "scan_rotated", "templates"), default="photo")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")
    train, val = convert(args.root, args.out, args.val_ratio, args.seed, mode=args.mode, limit=args.limit)
    print(f"Converted {train + val} images: train={train}, val={val}, mode={args.mode}")


if __name__ == "__main__":
    main()
