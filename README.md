# ArcheoZS: archaeological site detection under region-held-out testing

Can free satellite data find archaeological sites in regions a model has never seen? This project tests that with 196 recorded sites (100 settlements, 96 burial mounds) from the EAMENA database in 14 regions of the Middle East and North Africa. Whole regions at least 300 km apart are held out from training, and every site is compared with background patches from its own image.

**Paper:** [`GeoArcheo/docs/ArcheoZS_Research_Paper.pdf`](GeoArcheo/docs/ArcheoZS_Research_Paper.pdf) (Nisha Roy, draft)
**Code and full documentation:** [`GeoArcheo/`](GeoArcheo/), start with [`GeoArcheo/README.md`](GeoArcheo/README.md)
**Follow-up results (tables and logs):** [`GeoArcheo/results/follow_up/`](GeoArcheo/results/follow_up/)

## Main findings

| Question | Result (within-site AUC; 0.50 = chance) |
|---|---|
| Colour of the ground alone | about 0.60 (random forest, one date) |
| Prithvi-EO-2.0 foundation model, frozen or LoRA-tuned | 0.58–0.62, no better than a random forest |
| Four seasons instead of one date | +0.03 to +0.04, not confirmed after correction |
| **Bare-earth elevation (FABDEM)** | **0.68 alone; 0.69 [0.65–0.73] with one date of imagery** |

Follow-up tests kept the elevation gain with backgrounds 5–10 km from known sites and with backgrounds matched to each site's terrain roughness and slope (+0.07 to +0.10), although matching lowered the elevation-only score by 0.03–0.05. The same model scores 0.78 overall when regions are mixed between training and test, against 0.66 with regions held out, so easy test setups overstate performance.

## Limits

Backgrounds may contain unrecorded sites, only 14 regions exist, and elevation may partly separate raised terrain from flat terrain rather than archaeology. See Section 7 of the paper.

## Data and ethics

Site coordinates and EAMENA records are not included and will not be, because precise locations can help looters. FABDEM is CC BY-NC-SA 4.0 (research use only). Check the Prithvi-EO-2.0 model card for its licence.

## Licence

Code and paper: © 2026 Nisha Roy, all rights reserved. Published for review and reproducibility only, not for reuse or modification (see [`GeoArcheo/LICENSE`](GeoArcheo/LICENSE)). Contributions and pull requests are not accepted. To cite, see [`GeoArcheo/CITATION.cff`](GeoArcheo/CITATION.cff).
