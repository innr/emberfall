# MIDV-2020 smoke-test artifacts

This directory contains four downscaled derivative previews, their normalized
YOLO labels, and the JSON test report produced by the MIDV-2020 photo-mode
smoke test. It is intentionally not a copy of the full dataset.

The red rectangle in each preview is the MRZ box produced by mapping the
template annotation through the photo's `doc_quad` homography.

## Reproduction

```bash
python -m unittest discover -s tests
python tools/prepare_midv.py \
  --root data/midv2020/raw \
  --mode photo \
  --out data/midv2020/yolo \
  --val-ratio 0.25 \
  --limit 4
```

## Attribution and licence

Source: [MIDV-2020](https://zenodo.org/records/18786808), “A Comprehensive
Benchmark Dataset for Identity Document Analysis”, Bulatov et al.

The Zenodo record identifies the source as **CC BY-SA 2.5**. Face imagery was
obtained from [Generated Photos](https://generated.photos/); retain that
attribution when redistributing derivatives. See [LICENSE.txt](LICENSE.txt).

The official [MIDV-2020 page](https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html)
also describes the provider access form and required citation. Confirm that
your intended use complies with both the upstream access terms and the licence
before redistributing these artifacts.
