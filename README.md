# emberfall

Synthetic passport MRZ dataset generator for **TD3 passport MRZ detection and OCR**.

The goal is to reduce manual annotation: because the program creates the MRZ region itself, it automatically knows both the detection bounding box and the OCR ground truth.

## Output

Each generated sample contains a fake passport-style page, a YOLO bounding-box label covering the two MRZ lines, an MRZ crop, and the exact OCR text.

```text
dataset/
├── images/
├── labels/
├── mrz_crops/
├── classes.txt
└── ocr_labels.txt
```

The YOLO class is `0 = mrz`, using normalized `class_id x_center y_center width height` coordinates.

## TD3 format

Generated MRZ records use two lines of 44 characters, the A-Z / 0-9 / `<` character set, and ICAO-style 7-3-1 check digits.

## Install and generate

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/mrz_generator.py --out dataset --count 1000
python src/mrz_generator.py --out dataset --count 50000 --seed 2026
```

Image size can be changed with `--width` and `--height`.

## Tests

```bash
python -m unittest discover -s tests
```

## Current augmentation

The first version includes contrast variation, Gaussian blur and small rotations. Production-oriented follow-up work should add perspective distortion, motion blur, glare/reflection, low-resolution resampling, JPEG artifacts, uneven illumination, OCR-B font variants, and real target-device images for domain adaptation.

## Recommended data mix

Use roughly 30k-100k synthetic full-page samples for MRZ detection, 100k+ synthetic MRZ crops for OCR, public document datasets for realistic backgrounds, and 200-500 manually labeled images captured by the target device for final fine-tuning and validation.

Synthetic data should reduce, not completely replace, target-device real data.
