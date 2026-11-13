# pipeline/detect_dem.py
#
# PURPOSE:
#   Two fixes tested together on the SAME samples, folds and background draws
#   as detect_seasons.py (runs on the Mac, CPU only, no Prithvi needed):
#
#   1. ELEVATION: does adding "is this spot raised above its surroundings"
#      (Copernicus 30 m elevation) help? Rows:
#          Random forest, 1 date / 4 seasons            (as before)
#          Elevation only
#          Random forest, 1 date + elevation
#          Random forest, 4 seasons + elevation
#      Results are also split by site type: elevation should help settlement
#      mounds (tells) much more than graves.
#
#   2. YES/NO CALIBRATION: scores drift from one image to another, so a
#      single threshold works badly in unseen regions. Each score is compared
#      with the scores of random, unlabeled patches from the same image
#      ("tile-normalized"). Balanced accuracy is reported three ways:
#          raw score, threshold 0.5
#          raw score, threshold chosen on the other folds
#          tile-normalized score, threshold chosen on the other folds
#
#   All five background draws are pooled (each site vs 15 backgrounds), and
#   the comparisons are Holm-corrected, exactly like combine_seeds.py.
#
# INPUT:   data/season_patches[_sN].npz and data/season_patches[_sN]_extra.npz
# OUTPUT:  outputs/detect_predictions_dem[_sN].csv  (works with analyze_triage.py)
#          outputs/dem_pooled.csv
#
# USAGE (from the GeoArcheo folder):
#   python -m pipeline.detect_dem
#   python -m pipeline.detect_dem --extra-tag _fabdem      # bare-earth elevation (FABDEM)

import argparse
import csv
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from pipeline.features import dem_features, season_features, standardize
from utils.config import load_config

cfg = load_config()
N_FOLDS = 5
RUNS = [("seed 123", "data/season_patches.npz"),
        ("seed 7", "data/season_patches_s7.npz"),
        ("seed 11", "data/season_patches_s11.npz"),
        ("seed 19", "data/season_patches_s19.npz"),
        ("seed 42", "data/season_patches_s42.npz")]
NAMES = {"rf_1date": "Random forest, 1 date",
         "rf_4season": "Random forest, 4 seasons",
         "dem_only": "Elevation only",
         "rf_1date_dem": "Random forest, 1 date + elevation",
         "rf_4season_dem": "Random forest, 4 seasons + elevation"}
COMPARISONS = [("4 seasons: + elevation minus without", "rf_4season_dem", "rf_4season"),
               ("1 date: + elevation minus without", "rf_1date_dem", "rf_1date"),
               ("Elevation only minus chance (0.5)", "dem_only", None)]


def clean(a):
    """Random forests here cannot take missing values: 0 + a 'missing' flag."""
    miss = ~np.isfinite(a).all(axis=1, keepdims=True)
    return np.hstack([np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0), miss.astype(np.float32)])


EXTRA_TAG = ""


def load_run(path):
    d = np.load(path)
    e = np.load(path.replace(".npz", f"_extra{EXTRA_TAG}.npz"))
    for k in ("y", "site", "ids"):
        if not np.array_equal(d[k], e[k]):
            raise SystemExit(f"{path}: the _extra file does not match the samples ({k}). "
                             "Re-run build_season_patches.py --extra-only with the same seed.")
    X = d["X"]
    fp = np.array([int(X[i].astype(np.int64).sum()) for i in range(len(X))])
    if not np.array_equal(fp, e["check"]):
        raise SystemExit(f"{path}: the _extra file describes different patches than the image file. "
                         "Re-run build_season_patches.py --extra-only with the same seed as that file.")
    rf1, rf4 = zip(*[season_features(standardize(X[i], cfg['image']['mean'], cfg['image']['std']))
                     for i in range(len(X))])
    dem_raw = e["dem"].astype(np.float32)          # read once (each e[...] access re-reads the file)
    dem = np.stack([dem_features(dem_raw[i]) for i in range(len(X))])
    feats = {"rf_1date": np.stack(rf1), "rf_4season": np.stack(rf4), "dem_only": dem}
    ref = {"rf_1date": e["ref_rf1"], "rf_4season": e["ref_rf4"], "dem_only": e["ref_dem"]}
    for base in ("rf_1date", "rf_4season"):
        feats[base + "_dem"] = np.hstack([feats[base], dem])
        ref[base + "_dem"] = np.hstack([ref[base], ref["dem_only"]])
    feats = {k: clean(v) for k, v in feats.items()}
    ref = {k: clean(v) for k, v in ref.items()}
    return dict(feats=feats, ref=ref, y=d["y"], site=d["site"], grp=d["grp"], ids=list(d["ids"]),
                tile=e["tile"], ref_tile=e["ref_tile"],
                dem_ok=np.isfinite(dem_raw.reshape(len(X), -1)).any(axis=1))


def best_threshold(score, y):
    cand = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 97)))
    ba = [((score[y == 1] >= t).mean() + (score[y == 0] < t).mean()) / 2 for t in cand]
    return cand[int(np.argmax(ba))]


def balanced_acc(score, y, t):
    return ((score[y == 1] >= t).mean() + (score[y == 0] < t).mean()) / 2


def run_one(r):
    y, grp = r["y"], r["grp"]
    folds = list(GroupKFold(n_splits=N_FOLDS).split(r["feats"]["rf_1date"], y, grp))
    raw = {m: np.zeros(len(y)) for m in NAMES}
    norm = {m: np.zeros(len(y)) for m in NAMES}
    fold_of = np.zeros(len(y), dtype=int)
    for fold, (tr, te) in enumerate(folds):
        fold_of[te] = fold
        te_tiles = set(r["tile"][te])
        ref_idx = np.where(np.isin(r["ref_tile"], list(te_tiles)))[0]
        for m in NAMES:
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                        random_state=fold, n_jobs=-1).fit(r["feats"][m][tr], y[tr])
            s = rf.predict_proba(r["feats"][m][te])[:, 1]
            raw[m][te] = s
            rs = rf.predict_proba(r["ref"][m][ref_idx])[:, 1] if len(ref_idx) else np.array([])
            for j, i in enumerate(te):
                pool = rs[r["ref_tile"][ref_idx] == r["tile"][i]]
                norm[m][i] = (s[j] - pool.mean()) / (pool.std() + 1e-6) if len(pool) >= 5 else np.nan
    # yes/no accuracy, threshold for each fold chosen on the OTHER folds' predictions
    acc = {}
    for m in NAMES:
        a05 = balanced_acc(raw[m], y, 0.5)
        a_raw, a_norm = [], []
        for f in range(N_FOLDS):
            te, ot = fold_of == f, fold_of != f
            a_raw.append((te.sum(), balanced_acc(raw[m][te], y[te], best_threshold(raw[m][ot], y[ot]))))
            okn = ot & np.isfinite(norm[m])
            tef = te & np.isfinite(norm[m])
            if tef.sum() and okn.sum():
                a_norm.append((tef.sum(), balanced_acc(norm[m][tef], y[tef], best_threshold(norm[m][okn], y[okn]))))
        w = lambda lst: sum(n * a for n, a in lst) / sum(n for n, _ in lst)
        acc[m] = (a05, w(a_raw), w(a_norm) if a_norm else np.nan)
    return raw, norm, fold_of, acc


def within_site(score, y, site):
    out = {}
    for s in np.unique(site):
        m = site == s
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        out[s] = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    return out


def holm(p):
    order = np.argsort(p)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = run
    return adj


def site_types():
    path = cfg['paths']['eamena_geojson']
    try:
        with open(path) as f:
            g = json.load(f)
        return {ft["properties"]["site_id"]: ft["properties"].get("site_type", "?") for ft in g["features"]}
    except Exception:
        return {}


def main():
    global EXTRA_TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-tag", default="", help="e.g. _fabdem to use the bare-earth elevation files")
    args = ap.parse_args()
    EXTRA_TAG = args.extra_tag
    runs = [(name, p) for name, p in RUNS if os.path.exists(p)]
    print(f"Elevation files: *_extra{EXTRA_TAG}.npz")
    print(f"{len(runs)} background draws found: {[n for n, _ in runs]}\n")
    per_run, accs = [], []
    for name, path in runs:
        print(f"{name}: features...", flush=True)
        r = load_run(path)
        print(f"  {int(r['y'].sum())} sites, elevation for {int(r['dem_ok'].sum())}/{len(r['y'])} samples, "
              f"{len(r['ref_tile'])} reference patches; training...", flush=True)
        raw, norm, fold_of, acc = run_one(r)
        accs.append(acc)
        tag = "" if name == "seed 123" else "_s" + name.split()[1]
        path_out = os.path.join("outputs", f"detect_predictions_dem{EXTRA_TAG}{tag}.csv")
        with open(path_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["site_id", "patch", "region_group", "fold"] + list(NAMES))
            for i in range(len(r["y"])):
                w.writerow([r["ids"][r["site"][i]], "site" if r["y"][i] else "background", int(r["grp"][i]),
                            int(fold_of[i])] + [f"{raw[m][i]:.5f}" for m in NAMES])
        ws = {m: within_site(raw[m], r["y"], r["site"]) for m in NAMES}
        per_run.append({m: {r["ids"][s]: v for s, v in ws[m].items()} for m in NAMES})
        print("  " + "  ".join(f"{m} {np.mean(list(ws[m].values())):.3f}" for m in NAMES), flush=True)

    sites = sorted(set.intersection(*[set(p["rf_1date"]) for p in per_run]))
    pooled = {m: np.array([np.mean([p[m][s] for p in per_run]) for s in sites]) for m in NAMES}
    rng = np.random.default_rng(0)
    boot = [rng.choice(len(sites), len(sites), replace=True) for _ in range(5000)]
    types = site_types()
    tarr = np.array([types.get(s, "?") for s in sites])

    print("\n" + "=" * 100)
    print(f"ELEVATION, POOLED OVER {len(per_run)} BACKGROUND DRAWS ({len(sites)} sites, 300 km regions held out)")
    print("=" * 100)
    kinds = [t for t in sorted(set(tarr)) if t != "?"]
    head = f"  {'Method':<38}{'Within-site AUC (all)':>26}" + "".join(f"{k[:14]:>16}" for k in kinds)
    print(head)
    rows = []
    for m in NAMES:
        bs = [pooled[m][i].mean() for i in boot]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        by_kind = "".join(f"{pooled[m][tarr == k].mean():>16.3f}" for k in kinds)
        print(f"  {NAMES[m]:<38}{pooled[m].mean():>12.3f} [{lo:.2f}-{hi:.2f}]{by_kind}")
        rows.append(dict(row=NAMES[m], value=round(pooled[m].mean(), 4), lo=round(lo, 4), hi=round(hi, 4),
                         **{k: round(pooled[m][tarr == k].mean(), 4) for k in kinds}))
    if kinds:
        print("  (" + ", ".join(f"{k}: {int((tarr == k).sum())} sites" for k in kinds) + ")")

    print("\n  Paired comparisons (Holm-corrected):")
    raw_p, lines = [], []
    for name, a, b in COMPARISONS:
        d = pooled[a] - (pooled[b] if b else 0.5)
        bs = np.array([d[i].mean() for i in boot])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = max(float(np.mean(bs <= 0)), 1 / len(bs))
        raw_p.append(p); lines.append((name, d.mean(), lo, hi, p))
    for (name, dm, lo, hi, p), pa in zip(lines, holm(np.array(raw_p))):
        print(f"  {name:<40}{dm:+.3f} [{lo:+.3f} to {hi:+.3f}]  P={p:.3f}  Holm P={pa:.3f}  "
              f"{'CONVINCING' if pa < 0.025 else 'not shown'}")
        rows.append(dict(row=name, value=round(dm, 4), lo=round(lo, 4), hi=round(hi, 4)))

    print("\n" + "=" * 100)
    print("YES/NO DECISION: balanced accuracy (chance 0.50), mean over draws (min-max)")
    print("=" * 100)
    print(f"  {'Method':<38}{'raw, t=0.5':>18}{'raw, tuned t':>18}{'tile-normalized':>20}")
    for m in NAMES:
        cols = []
        for k in range(3):
            v = np.array([a[m][k] for a in accs], dtype=float)
            cols.append(f"{np.nanmean(v):.3f} ({np.nanmin(v):.2f}-{np.nanmax(v):.2f})")
            rows.append(dict(row=f"{NAMES[m]} | balanced accuracy | {['raw 0.5', 'raw tuned', 'tile-normalized'][k]}",
                             value=round(float(np.nanmean(v)), 4), lo=round(float(np.nanmin(v)), 4),
                             hi=round(float(np.nanmax(v)), 4)))
        print(f"  {NAMES[m]:<38}{cols[0]:>18}{cols[1]:>18}{cols[2]:>20}")
    print("  Thresholds are chosen on the other folds only; reference patches are random, unlabeled")
    print("  ground from the same image, as would be available in real use.")

    keys = sorted({k for r in rows for k in r}, key=lambda k: ["row", "value", "lo", "hi"].index(k)
                  if k in ("row", "value", "lo", "hi") else 9)
    with open(os.path.join("outputs", f"dem_pooled{EXTRA_TAG}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> outputs/dem_pooled{EXTRA_TAG}.csv and outputs/detect_predictions_dem{EXTRA_TAG}*.csv")


if __name__ == "__main__":
    main()
