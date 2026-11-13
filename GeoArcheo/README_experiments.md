# Follow-up experiments (reviewer requests)

Three experiments that test the weakest points of the paper. **These scripts have not been run on the real data yet**; they were checked on synthetic files in the same format (so the code runs end to end), not on your tiles. Until you run them, the paper treats all three as future work.

All of them reuse the paper's own features, folds (5 folds over the 14 regions, 300 km), random forest settings, metric (within-site AUC) and statistics (95% intervals from resampling regions; paired comparisons Holm-corrected, convincing if Holm P < 0.025), so the numbers are directly comparable.

## Where each file goes (GeoArcheo folder, on the Mac)

| File | Goes to | New or replaces |
|---|---|---|
| `pipeline/exp_common.py` | `GeoArcheo/pipeline/` | new |
| `pipeline/exp_backgrounds.py` | `GeoArcheo/pipeline/` | new |
| `pipeline/exp_importance.py` | `GeoArcheo/pipeline/` | new |
| `pipeline/exp_seasons.py` | `GeoArcheo/pipeline/` | new |
| `build_season_patches.py` | `GeoArcheo/` | **replaces** the old one (adds `--min-km`, `--terrain-matched`, `--match-pool`; with no flags it samples exactly as before) |
| `run_background_builds.sh` | `GeoArcheo/` | new |

Needs `scikit-learn` and `scipy` (already used). `pip install shap` is optional (adds SHAP to experiment 2).

## 1. Backgrounds: distance and terrain matching

```
bash run_background_builds.sh          # builds km5, km10, matched_km2.5, matched_km5 (slow: start with ONE seed)
python -m pipeline.exp_backgrounds
```

Answers: does the result depend on the 2.5 km buffer, and does the model find archaeology or just unusual terrain? Outputs `outputs/exp_backgrounds_*.csv`. The table includes a "patch-level terrain only" baseline and a check that the matching worked. How to read it is at the top of `pipeline/exp_backgrounds.py`.

Notes: matching uses the roughness and slope of the whole 2.2 km elevation window, so it removes only that kind of terrain cue; a bump at the centre can still differ. Sites whose elevation is missing are dropped in the matched conditions.

## 2. Elevation: importance, SHAP and ablations on held-out regions

```
python -m pipeline.exp_importance --with-colour
```

Permutation importance and SHAP are measured on the held-out regions only; top-k features are chosen inside each fold from the training regions only. Ablations: leave one feature out, one feature alone, drop a window size, centre-height vs peak-height features, slope/roughness only, colour plus each elevation group. Outputs `outputs/exp_importance_*.csv` (features, groups, correlations, ablations, comparisons). Correlated features share importance, so read the group results and the ablations together.

## 3. Seasons

```
python -m pipeline.exp_seasons
```

Single seasons; four seasons with and without the change-over-the-year features; dropping one season at a time; cloud/snow-affected vs clean sites; settlements vs burial mounds; redundancy with the one-date colour features; each season on top of elevation. Outputs `outputs/exp_seasons_*.csv`.

**Check the cloud proxy before trusting the cloud split.** The image files have no cloud mask, so a pixel counts as "bright" when its blue band is above `--blue-thr` (default 2000, which assumes values scaled x10000) and a patch is "affected" when more than `--frac` (default 0.10) of its pixels are bright. The script prints the distribution per season and the number of affected tiles; the reviewer mentions nine winter tiles, so adjust `--blue-thr`, `--frac` and `--blue-band` until that is plausible. Features are cached, so re-running with new settings is quick.

Also note: the paper's "1 date" is the **winter** image (Jan-Feb), the season the reviewer says is partly cloud- or snow-affected. So "four seasons vs one date" partly asks whether other seasons repair a poor winter image; the cloud split in this experiment tests that directly.

## Time

Rough guide on a laptop, not timed on your data: backgrounds builds are the slow part (each build re-reads every tile). Experiment 2 takes roughly half an hour and experiment 3 about 15-30 minutes with 500 trees; add `--trees 200 --seeds 2` to try them quickly.

## After running

Send me the `outputs/exp_*.csv` files (they hold only summary numbers, no coordinates) and I will add the results to the paper and the deck. Do not edit the paper's claims before seeing them: any of the three could change the conclusion.
