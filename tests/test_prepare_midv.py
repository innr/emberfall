import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.prepare_midv import Box, convert, homography, project, yolo_line


def _write_annotation(path: Path, filename: str, regions: list[dict]) -> None:
    payload = {"_via_img_metadata": {"x": {"filename": filename, "regions": regions}}}
    path.write_text(json.dumps(payload), encoding="utf-8")


class PrepareMidvTests(unittest.TestCase):
    def test_merge_and_yolo(self):
        self.assertEqual(
            yolo_line(Box(100, 300, 900, 390), 1000, 500),
            "0 0.500000 0.690000 0.800000 0.180000\n",
        )

    def test_identity_homography(self):
        corners = [(0, 0), (100, 0), (100, 50), (0, 50)]
        self.assertEqual(project([(10, 20)], homography(corners, corners)), [(10.0, 20.0)])

    def test_convert_photo_maps_template_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "midv"
            for path in (
                root / "templates/images/aze_passport",
                root / "templates/annotations",
                root / "photo/images/aze_passport",
                root / "photo/annotations",
            ):
                path.mkdir(parents=True)
            filename = "00.jpg"
            for path in (
                root / "templates/images/aze_passport" / filename,
                root / "photo/images/aze_passport" / filename,
            ):
                Image.new("RGB", (1000, 500), "white").save(path)
            fields = [
                {"shape_attributes": {"name": "rect", "x": 100, "y": 300, "width": 800, "height": 40},
                 "region_attributes": {"field_name": "mrz_line0"}},
                {"shape_attributes": {"name": "rect", "x": 100, "y": 350, "width": 800, "height": 40},
                 "region_attributes": {"field_name": "mrz_line1"}},
            ]
            _write_annotation(root / "templates/annotations/aze_passport.json", filename, fields)
            doc_quad = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": [0, 1000, 1000, 0],
                    "all_points_y": [0, 0, 500, 500],
                },
                "region_attributes": {"field_name": "doc_quad"},
            }
            _write_annotation(root / "photo/annotations/aze_passport.json", filename, [doc_quad])
            train, val = convert(root, Path(directory) / "out", val_ratio=0, mode="photo")
            self.assertEqual((train, val), (1, 0))
            self.assertEqual(
                (Path(directory) / "out/labels/train/train_000000.txt").read_text(),
                "0 0.500000 0.690000 0.800000 0.180000\n",
            )


if __name__ == "__main__":
    unittest.main()
