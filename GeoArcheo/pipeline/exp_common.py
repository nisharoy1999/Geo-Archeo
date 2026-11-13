# pipeline/exp_common.py
#
# PURPOSE:
#   Shared helpers for the three follow-up experiments requested in review:
#       pipeline/exp_backgrounds.py   background distance / terrain-matched backgrounds
#       pipeline/exp_importance.py    held-out feature importance, SHAP, ablations (elevation)
#       pipeline/exp_seasons.py       why seasonal imagery does not help
#
#   Everything here reuses the pipeline's own features, folds and metric, so the
#   new numbers are directly comparable with the paper:
#     - same features        (pipeline/features.py)
#     - same folds           GroupKFold(5) over the 14 regions
#     - same random forest   500 trees, min_samples_leaf=2, class_weight=balanced
#     - same metric          within-site AUC (site vs its own backgrounds),
#                            averaged over the background draws
#     - same statistics      95% intervals by resampling REGIONS (cluster bootstrap),
#                            paired comparisons Holm-corrected, convincing if Holm P < 0.025
#
# NOTHING here peeks at test regions when choosing features or thresholds.

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from pipeline.detect_dem import clean, holm, within_site          # same code as the paper
from pipeline.features import dem_features, season_features, standardize
from utils.config import load_config

cfg = load_config()
N_FOLDS = 5
SEASONS = ["Jan-Feb", "Apr-Jun", "Jul-Sep", "Oct-Dec"]
B = 5000                                    # bootstrap resamples
# column order of dem_features(): for each window (80, 160, 320, 640 m): mean-height, peak-height; then 7 more
DEM_NAMES = [f"{kind}_{w}m" for w in (80, 160, 320, 640) for kind in ("centre_height", "peak_height")] + \
            ["slope_centre", "slope_patch", "slope_centre_minus_patch", "roughness_centre",
             "roughness_patch", "centre_minus_patch_mean", "share_of_patch_lower"]
SEEDS = [("123", ""), ("7", "_s7"), ("11", "_s11"), ("19", "_s19"), ("42", "_s42")]


def run_paths(base="data/season_patches", n_seeds=5):
    """Image files for the background draws that exist, e.g. data/season_patches_s7.npz."""
    out = []
    for seed, suffix in SEEDS[:n_seeds]:
        p = f"{base}{suffix}.npz"
        if os.path.exists(p):
            out.append((seed, p))
    return out


# --------------------------------------------------------------------------- loading
def load_features(path, extra_tag="_fabdem", blue_band=0, cache=True):
    """Everything the experiments need from one background draw, computed once and cached.

    Returns a dict of arrays, one row per sample (site or background):
      y, site, grp, ids, tile
      dem            (N, 15)   elevation features, NaN where the elevation is missing
      season         (N, 4, F) per-season spectral features
      change         (N, Fc)   how much the centre changes over the year
      bright_grid    (N, 4, len(BLUE_GRID))  share of the patch whose blue band exceeds each value of BLUE_GRID
                     (a cloud/snow proxy; use bright_at(F, threshold) to read one threshold)
    """
    cpath = path.replace(".npz", f"_expcache{extra_tag}_b{blue_band}.npz")
    if cache and os.path.exists(cpath) and os.path.getmtime(cpath) > os.path.getmtime(path):
        z = np.load(cpath, allow_pickle=False)
        return {k: z[k] for k in z.files}
    d = np.load(path)
    e = np.load(path.replace(".npz", f"_extra{extra_tag}.npz"))
    for k in ("y", "site", "ids"):
        if not np.array_equal(d[k], e[k]):
            raise SystemExit(f"{path}: the _extra{extra_tag} file does not match the samples ({k}).")
    X = d["X"]
    fp = np.array([int(X[i].astype(np.int64).sum()) for i in range(len(X))])
    if not np.array_equal(fp, e["check"]):
        raise SystemExit(f"{path}: the _extra{extra_tag} file describes different patches than the image file.")
    n = len(X)
    season, change, bright = [], [], []
    for i in range(n):
        x4 = standardize(X[i], cfg["image"]["mean"], cfg["image"]["std"])
        _, f4 = season_features(x4)
        # season_features() = 4 per-season blocks, then the change block (2 x bands values)
        n_change = 2 * x4.shape[0]
        per = (len(f4) - n_change) // 4
        season.append(f4[:4 * per].reshape(4, per))
        change.append(f4[4 * per:])
        bright.append(bright_grid(X[i], blue_band))
    dem_raw = e["dem"].astype(np.float32)
    dem = np.stack([dem_features(dem_raw[i]) for i in range(n)])
    out = dict(y=d["y"], site=d["site"], grp=d["grp"], ids=np.array(d["ids"]).astype(str),
               tile=np.array(e["tile"]).astype(str), dem=dem,
               season=np.stack(season).astype(np.float32), change=np.stack(change).astype(np.float32),
               bright_grid=np.stack(bright).astype(np.float32))
    if cache:
        np.savez_compressed(cpath, **out)
    return out


BLUE_GRID = np.arange(500, 6001, 250)      # blue-band values tried as "bright" thresholds (raw units)


def bright_grid(x_raw, blue_band=0):
    """x_raw: (bands, 4, H, W) raw int16. For each season, the share of pixels whose blue band exceeds
    each value of BLUE_GRID. Clouds and snow are both bright in blue; bare ground rarely is."""
    b = x_raw[blue_band].reshape(4, -1)
    return np.stack([(b > t).mean(axis=1) for t in BLUE_GRID], axis=1)


def bright_at(F, thr):
    """(N, 4) share of bright pixels per season at the grid value nearest to `thr`."""
    j = int(np.argmin(np.abs(BLUE_GRID - thr)))
    return F["bright_grid"][:, :, j]


def site_table(F):
    """One row per site (the positive sample), in site-index order."""
    pos = np.where(F["y"] == 1)[0]
    pos = pos[np.argsort(F["site"][pos])]
    return pos


# --------------------------------------------------------------------------- models
def make_folds(y, grp):
    return list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, grp))


def forest(trees, seed):
    return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)


def cv_scores(X, y, folds, trees=500, columns=None, return_models=False):
    """Out-of-fold probability for every sample; each fold's test regions are never seen in training.
    columns: None (all), a list of column indices, or a function(train_idx, X, y) -> columns
    chosen from the TRAINING regions only."""
    score = np.zeros(len(y))
    models = []
    for k, (tr, te) in enumerate(folds):
        cols = columns(tr, X, y) if callable(columns) else columns
        cols = slice(None) if cols is None else list(cols)
        rf = forest(trees, k).fit(X[tr][:, cols], y[tr])
        score[te] = rf.predict_proba(X[te][:, cols])[:, 1]
        models.append((rf, cols))
    return (score, models) if return_models else score


def site_auc(score, F):
    """Within-site AUC per site id (same function as the paper)."""
    ws = within_site(score, F["y"], F["site"])
    return {str(F["ids"][s]): v for s, v in ws.items()}


# --------------------------------------------------------------------------- statistics
class Pool:
    """Per-site within-site AUC averaged over the background draws, plus each site's region.
    Only sites present in every draw are used, so models are compared on the same sites."""

    def __init__(self):
        self.runs = []              # list of {model: {site: auc}}
        self.region = {}
        self.kind = {}

    def add(self, per_model, F):
        self.runs.append(per_model)
        for i in site_table(F):
            self.region.setdefault(str(F["ids"][F["site"][i]]), int(F["grp"][i]))

    def sites(self):
        return sorted(set.intersection(*[set(next(iter(r.values()))) for r in self.runs]))

    def per_site(self, model):
        s = self.sites()
        return np.array([np.mean([r[model][x] for r in self.runs]) for x in s])

    def region_index(self):
        s = self.sites()
        reg = np.array([self.region[x] for x in s])
        uniq = np.unique(reg)
        return np.array([np.where(uniq == r)[0][0] for r in reg]), len(uniq)


def cluster_boot(values, region_idx, n_regions, mask=None, reps=B, seed=0):
    """Mean and 95% interval, resampling REGIONS with replacement (all sites of a region kept).
    mask: optional boolean array selecting a subset of sites (e.g. settlements only)."""
    v = np.asarray(values, dtype=float)
    keep = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
    sums = np.bincount(region_idx[keep], weights=v[keep], minlength=n_regions)
    cnt = np.bincount(region_idx[keep], minlength=n_regions).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_regions, size=(reps, n_regions))
    den = cnt[idx].sum(1)
    ok = den > 0
    means = sums[idx].sum(1)[ok] / den[ok]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v[keep].mean()) if keep.any() else float("nan"), float(lo), float(hi), means


def paired(pool, a, b, mask=None):
    """Mean difference a - b (b may be None = chance 0.5), region-bootstrap interval and one-sided P."""
    va = pool.per_site(a)
    vb = pool.per_site(b) if b else np.full(len(va), 0.5)
    ri, nr = pool.region_index()
    m, lo, hi, means = cluster_boot(va - vb, ri, nr, mask)
    p = max(float(np.mean(means <= 0)), 1.0 / len(means))
    return m, lo, hi, p


def fmt_ci(m, lo, hi):
    return f"{m:.3f} [{lo:.2f}-{hi:.2f}]"


def report_comparisons(pool, comps, title):
    """comps: list of (label, a, b). Prints a Holm-corrected table, returns rows for CSV."""
    res = [paired(pool, a, b) for _, a, b in comps]
    adj = holm(np.array([r[3] for r in res]))
    print(f"\n  {title}  (Holm-corrected; convincing if Holm P < 0.025)")
    rows = []
    for (label, _, _), (m, lo, hi, p), pa in zip(comps, res, adj):
        verdict = "CONVINCING" if pa < 0.025 else "not shown"
        print(f"    {label:<52}{m:+.3f} [{lo:+.3f} to {hi:+.3f}]  P={p:.3f}  Holm P={pa:.3f}  {verdict}")
        rows.append(dict(comparison=label, diff=round(m, 4), lo=round(lo, 4), hi=round(hi, 4),
                         p=round(p, 4), holm_p=round(float(pa), 4), verdict=verdict))
    return rows


def write_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  saved -> {path}")


def site_kinds(F_ids):
    """Site type per site id from the EAMENA file (settlement / burial mound), '?' if unknown."""
    from pipeline.detect_dem import site_types
    t = site_types()
    return np.array([t.get(s, "?") for s in F_ids])
