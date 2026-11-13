# ArcheoZS

Detecting archaeological sites from Sentinel-2 imagery and bare-earth elevation, tested the hard way: whole regions held out from training.

**Paper:** [`docs/ArcheoZS_Research_Paper.pdf`](docs/ArcheoZS_Research_Paper.pdf) — *Evaluating Geographic Generalization in Archaeological Site Detection: A Comparative Study of Foundation Models, Satellite Imagery and Bare-Earth Elevation* (Nisha Roy, draft).

## What this project found

196 recorded sites from the EAMENA database (100 settlements, 96 burial mounds) across 14 regions of the Middle East and North Africa, each compared with background patches from the same image. Whole regions at least 300 km apart are held out from training. Every main result is averaged over five background draws and compared with paired bootstrap tests, with a region-level bootstrap as a robustness check.

| Question | Result (within-site AUC; 0.50 = chance) |
|---|---|
| Colour of the ground alone | about 0.60 (random forest, 1 date); 0.63 with 4 seasons |
| Frozen or LoRA-tuned Prithvi-EO-2.0 foundation model | 0.58–0.62, no better than a random forest |
| Four seasons instead of one date | +0.03 to +0.04, likely but not confirmed after correction |
| **Bare-earth elevation (FABDEM)** | **0.68 alone; 0.69 [0.65–0.73] with one date of imagery** |
| Survey triage at 90% of sites kept | skips about 35% of the sampled background patches (chance: 10%) |

The same elevation model scores an overall AUC of 0.78 when regions are mixed between training and test, against 0.66 when regions are held out. Easy test setups overstate how well these methods generalise.

**Limits, stated plainly.** Backgrounds may contain unrecorded sites; one background design was used (3 patches per site, at least 2.5 km from recorded sites); the skip figure is for sampled patches, not whole areas; only 14 regions exist, so even region-level intervals are approximate; elevation may partly separate raised terrain from flat terrain rather than archaeology. See Section 7 of the paper.

## Repository layout

```
download_seasons.py        seasonal Sentinel-2 composites (Copernicus openEO)
download_dem.py            Copernicus GLO-30 elevation on each tile's 10 m grid
download_fabdem.py         FABDEM bare-earth elevation on the same grid
build_season_patches.py    sites + background patches + reference patches
pipeline/features.py       spectral, seasonal and elevation features
pipeline/detect_crossval.py   region-held-out evaluation, Prithvi (frozen / LoRA)
pipeline/detect_seasons.py    1-date vs 4-season, random forest vs Prithvi
pipeline/detect_dem.py        elevation experiments, yes/no calibration
combine_seeds.py           pool the five background draws, paired tests
cluster_boot.py            region-level and two-stage bootstrap
analyze_triage.py          conformal survey triage
docker/                    container recipes
docs/                      the paper
```

## Reproducing

```bash
pip install -r requirements.txt
# 1. imagery and elevation (needs a free Copernicus Data Space account)
python download_seasons.py && python download_dem.py && python download_fabdem.py
# 2. patches
python build_season_patches.py
# 3. experiments
python pipeline/detect_seasons.py
python pipeline/detect_dem.py --extra-tag _fabdem
python combine_seeds.py && python analyze_triage.py --pred outputs/detect_predictions_dem_fabdem.csv
```

A GPU is needed for the Prithvi runs (developed on an A10G); the random-forest experiments run on a CPU.

## Data and ethics

- **Site coordinates and the EAMENA records are not included, and will not be.** Precise locations of archaeological sites can help looters. You must obtain site records from EAMENA (https://eamena.org) under its terms and apply for any access you need.
- The code expects labels under `data/labels/` (not provided) and writes downloaded imagery to `data/` (git-ignored).
- Sentinel-2 and Copernicus GLO-30 are open data. FABDEM (Hawker et al. 2022) is licensed CC BY-NC-SA 4.0: research use only, with attribution.
- Prithvi-EO-2.0 is used through TerraTorch; check its model card for its licence.

## Licence

Code: MIT (see `LICENSE`). The paper in `docs/` is © the author; the data sources above keep their own licences.

## Citation

See `CITATION.cff`.
