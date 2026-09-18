from __future__ import annotations

import argparse
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


CHARSET = string.ascii_uppercase + string.digits + "<"
WEIGHTS = (7, 3, 1)


def mrz_value(ch: str) -> int:
    if ch == "<":
        return 0
    if ch.isdigit():
        return int(ch)
    return ord(ch) - ord("A") + 10


def check_digit(text: str) -> str:
    total = sum(mrz_value(ch) * WEIGHTS[i % 3] for i, ch in enumerate(text))
    return str(total % 10)


def rand_letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(n))


def random_name(rng: random.Random) -> Tuple[str, str]:
    surnames = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "MILLER", "DAVIS", "ZHANG", "WANG", "LI"]
    given = ["JOHN", "ANNA", "MARIA", "DAVID", "ROBERT", "JAMES", "LIN", "WEI", "MING", "XIN"]
    return rng.choice(surnames), rng.choice(given)


def random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def fmt_date(d: date) -> str:
    return d.strftime("%y%m%d")


@dataclass
class TD3Record:
    line1: str
    line2: str


def make_td3(rng: random.Random) -> TD3Record:
    country = rng.choice(["UTO", "USA", "CHN", "GBR", "FRA", "DEU", "SGP", "JPN"])
    nationality = country
    surname, given = random_name(rng)

    name_field = f"{surname}<<{given}".replace(" ", "<")
    name_field = (name_field + "<" * 39)[:39]
    line1 = f"P<{country}{name_field}"
    assert len(line1) == 44

    passport_no = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(9))
    passport_cd = check_digit(passport_no)

    dob = random_date(rng, date(1950, 1, 1), date(2005, 12, 31))
    dob_s = fmt_date(dob)
    dob_cd = check_digit(dob_s)

    sex = rng.choice(["M", "F", "<"])

    expiry = random_date(rng, date.today() + timedelta(days=30), date.today() + timedelta(days=3650))
    exp_s = fmt_date(expiry)
    exp_cd = check_digit(exp_s)

    personal = "".join(rng.choice(string.ascii_uppercase + string.digits + "<") for _ in range(14))
    personal_cd = check_digit(personal)

    composite = passport_no + passport_cd + dob_s + dob_cd + exp_s + exp_cd + personal + personal_cd
    final_cd = check_digit(composite)

    line2 = (
        passport_no
        + passport_cd
        + nationality
        + dob_s
        + dob_cd
        + sex
        + exp_s
        + exp_cd
        + personal
        + personal_cd
        + final_cd
    )
    assert len(line2) == 44
    return TD3Record(line1=line1, line2=line2)


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_fake_passport(record: TD3Record, width: int, height: int, rng: random.Random):
    bg = rng.randint(224, 245)
    img = Image.new("L", (width, height), color=bg)
    draw = ImageDraw.Draw(img)

    title_font = get_font(max(20, width // 30))
    body_font = get_font(max(14, width // 55))
    mrz_font = get_font(max(18, width // 46))

    margin = width // 18
    draw.text((margin, margin), "PASSPORT", fill=40, font=title_font)
    draw.rectangle((margin, height // 5, width // 3, height * 3 // 5), outline=90, width=2)
    draw.text((margin + 10, height // 5 + 10), "PHOTO", fill=110, font=body_font)

    info_x = width // 2
    info_y = height // 4
    gap = max(24, height // 18)
    for i, txt in enumerate([
        "Surname",
        "Given names",
        "Nationality",
        "Date of birth",
        "Passport No.",
        "Date of expiry",
    ]):
        draw.text((info_x, info_y + i * gap), txt, fill=80, font=body_font)

    line_gap = max(4, height // 100)
    bbox1 = draw.textbbox((0, 0), record.line1, font=mrz_font)
    bbox2 = draw.textbbox((0, 0), record.line2, font=mrz_font)
    text_w = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
    text_h = bbox1[3] - bbox1[1]

    mrz_x = max(margin, (width - text_w) // 2)
    mrz_y = height - margin - (text_h * 2 + line_gap)
    draw.text((mrz_x, mrz_y), record.line1, fill=20, font=mrz_font)
    draw.text((mrz_x, mrz_y + text_h + line_gap), record.line2, fill=20, font=mrz_font)

    pad_x = max(6, width // 200)
    pad_y = max(4, height // 200)
    box = (
        max(0, mrz_x - pad_x),
        max(0, mrz_y - pad_y),
        min(width, mrz_x + text_w + pad_x),
        min(height, mrz_y + text_h * 2 + line_gap + pad_y),
    )
    return img, box


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    if rng.random() < 0.5:
        angle = rng.uniform(-2.0, 2.0)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=240)
    return img


def box_to_yolo(box, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def generate_dataset(out_dir: Path, count: int, seed: int, width: int, height: int):
    rng = random.Random(seed)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    crops_dir = out_dir / "mrz_crops"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    ocr_lines = []
    for idx in range(count):
        rec = make_td3(rng)
        img, box = draw_fake_passport(rec, width, height, rng)

        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))
        aug = augment(img, rng)

        stem = f"{idx:06d}"
        aug.save(images_dir / f"{stem}.jpg", quality=92)
        crop.save(crops_dir / f"{stem}.png")
        (labels_dir / f"{stem}.txt").write_text(box_to_yolo(box, width, height) + "\n", encoding="utf-8")
        ocr_lines.append(f"{stem}.png\t{rec.line1}\n{rec.line2}\n")

    (out_dir / "ocr_labels.txt").write_text("".join(ocr_lines), encoding="utf-8")
    (out_dir / "classes.txt").write_text("mrz\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic TD3 passport MRZ detection/OCR data")
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    generate_dataset(args.out, args.count, args.seed, args.width, args.height)
    print(f"Generated {args.count} samples in {args.out}")


if __name__ == "__main__":
    main()
