# pipeline/detect_seasons.py
#
# PURPOSE:
#   Does seeing a place in FOUR seasons help find archaeological sites?
#   Same sites, same backgrounds, same region-held-out folds for every row:
#
#     Random forest, 1 date          spectral statistics, Jan-Feb only
#     Random forest, 4 seasons       same statistics for every season + how
#                                    much they change over the year
#     Frozen Prithvi, 1 date         as in detect_crossval.py
#     Frozen Prithvi, 4 seasons      Prithvi-EO-2.0 reading all four dates
#                                    at once (it was pretrained on 4-date
#                                    sequences), centre tokens averaged
#     Frozen Prithvi, 4 seasons,     same, but the centre features of each
#       per season                   season kept side by side
#
#   The random forest gets all four seasons too, so any gain of Prithvi over
#   it comes from the model, not just from having more images.
#
# INPUT:   data/season_patches.npz   (from build_season_patches.py)
# OUTPUT:  outputs/detect_predictions_seasons[_f<pct>].csv   (works with analyze_triage.py)
#          outputs/learning_curve_seasons.csv
#
# USAGE (from the GeoArcheo folder, on the GPU machine):
#   python -u -m pipeline.detect_seasons
#   python -u -m pipeline.detect_seasons --fracs 0.25,0.5,0.75,1.0
#   python -u -m pipeline.detect_seasons --input data/season_patches_s7.npz --tag _s7

import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from pipeline.detect_crossval import (pick_device, spectral_features, subsample,
                                      summarize, paired, save_predictions)
from utils.config import load_config

cfg = load_config()
IN_FILE = os.path.join("data", "season_patches.npz")
OUT_DIR = "outputs"
N_FOLDS = 5
GRID = 14
K = 2                      # centre tokens per side (2 x 2 = central ~320 m)
MEAN = torch.tensor(cfg['image']['mean']).view(-1, 1, 1, 1)
STD = torch.tensor(cfg['image']['std']).view(-1, 1, 1, 1)

NAMES = {"rf_1date": "Random forest, 1 date",
         "rf_4season": "Random forest, 4 seasons",
         "prithvi_1date": "Frozen Prithvi, 1 date",
         "prithvi_4season": "Frozen Prithvi, 4 seasons",
         "prithvi_4season_per": "Frozen Prithvi, 4 seasons, per season"}


def standardized(X, i, frames):
    """Sample i as a float tensor (bands, len(frames), H, W), Prithvi-standardized."""
    x = torch.from_numpy(X[i][:, frames].astype(np.float32))
    return (x - MEAN) / STD


def rf_features(X):
    one, four = [], []
    for i in range(len(X)):
        x = standardized(X, i, [0, 1, 2, 3])                  # (C, 4, H, W)
        per = [spectral_features(x[:, t]) for t in range(x.shape[1])]
        one.append(per[0])
        centre = x[:, :, 112 - 16:112 + 16, 112 - 16:112 + 16].mean(dim=(2, 3))   # (C, 4)
        whole = x.mean(dim=(2, 3))
        change = [centre.std(dim=1).numpy(), (centre - whole).std(dim=1).numpy()]
        four.append(np.concatenate(per + change))
    return np.stack(one), np.stack(four)


def build_prithvi(num_frames, device):
    from terratorch.registry import BACKBONE_REGISTRY
    name = cfg['model']['prithvi_name']
    if num_frames == 1:
        enc = BACKBONE_REGISTRY.build(name, pretrained=True)
    else:
        enc = BACKBONE_REGISTRY.build(name, pretrained=True, num_frames=num_frames)
    for p in enc.parameters():
        p.requires_grad = False
    return enc.to(device).eval()


def centre_features(tokens, n_frames):
    """tokens (B, 1 + T*196, D) -> per-season centre features (B, T, D)."""
    per = []
    lo = GRID // 2 - K // 2
    for t in range(n_frames):
        idx = [1 + t * GRID * GRID + r * GRID + c for r in range(lo, lo + K) for c in range(lo, lo + K)]
        per.append(tokens[:, idx, :].mean(dim=1))
    return torch.stack(per, dim=1)


def prithvi_features(X, frames, device, batch):
    enc = build_prithvi(len(frames), device)
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            x = torch.stack([standardized(X, i, frames) for i in range(s, min(s + batch, len(X)))])
            if len(frames) == 1:
                x = x[:, :, 0]                                 # (B, C, H, W), as in detect_crossval
            tokens = enc(x.to(device))[-1]
            if s == 0:
                print(f"    input {tuple(x.shape)} -> tokens {tuple(tokens.shape)}", flush=True)
                expected = 1 + len(frames) * GRID * GRID
                if tokens.shape[1] != expected:
                    raise SystemExit(f"Unexpected token count {tokens.shape[1]} (expected {expected}). "
                                     "Send this message to Claude.")
            out.append(centre_features(tokens, len(frames)).float().cpu())
            if (s // batch) % 10 == 0:
                print(f"    {min(s + batch, len(X))}/{len(X)}", flush=True)
    del enc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()                              # (N, T, D)


def probe(Xf, y, tr, te):
    scaler = StandardScaler().fit(Xf[tr])
    clf = LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced").fit(scaler.transform(Xf[tr]), y[tr])
    return clf.predict_proba(scaler.transform(Xf[te]))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="1.0", help="training-site shares, e.g. 0.25,0.5,0.75,1.0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--input", default=IN_FILE)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _s7")
    args = ap.parse_args()
    fracs = sorted(float(f) for f in args.fracs.split(","))

    device = pick_device()
    print(f"Device: {device}", flush=True)
    d = np.load(args.input)
    X, y, site, grp, ids = d["X"], d["y"], d["site"], d["grp"], list(d["ids"])
    print(f"{int(y.sum())} sites, {int((y == 0).sum())} backgrounds, {len(np.unique(grp))} regions, "
          f"seasons {list(d['seasons'])}, array {X.shape}", flush=True)

    print("Spectral features...", flush=True)
    rf1, rf4 = rf_features(X)
    print("Frozen Prithvi, 1 date...", flush=True)
    p1 = prithvi_features(X, [0], device, args.batch)[:, 0]
    print("Frozen Prithvi, 4 seasons...", flush=True)
    p4 = prithvi_features(X, [0, 1, 2, 3], device, args.batch)
    feats = {"rf_1date": rf1, "rf_4season": rf4, "prithvi_1date": p1,
             "prithvi_4season": p4.mean(axis=1), "prithvi_4season_per": p4.reshape(len(p4), -1)}
    methods = list(feats)

    folds = list(GroupKFold(n_splits=N_FOLDS).split(rf1, y, grp))
    curve_rows = []
    for frac in fracs:
        scores = {m: np.zeros(len(y)) for m in methods}
        fold_of = -np.ones(len(y), dtype=int)
        n_train = []
        for fold, (tr_full, te) in enumerate(folds):
            tr = subsample(tr_full, site, frac, fold)
            n_train.append(len(np.unique(site[tr])))
            t0 = time.time()
            for m in methods:
                if m.startswith("rf"):
                    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                random_state=fold, n_jobs=-1).fit(feats[m][tr], y[tr])
                    scores[m][te] = rf.predict_proba(feats[m][te])[:, 1]
                else:
                    scores[m][te] = probe(feats[m], y, tr, te)
            fold_of[te] = fold
            print(f"  {frac:.0%} data, fold {fold + 1}/{N_FOLDS}: {n_train[-1]} training sites, "
                  f"{time.time() - t0:.0f}s", flush=True)
        suffix = "" if frac >= 1.0 else f"_f{int(round(frac * 100))}"
        save_predictions(os.path.join(OUT_DIR, f"detect_predictions_seasons{args.tag}{suffix}.csv"),
                         ids, y, grp, site, fold_of, scores)

        print("\n" + "=" * 90)
        print(f"ONE DATE vs FOUR SEASONS, {frac:.0%} of training sites (~{np.mean(n_train):.0f} per fold), "
              f"{N_FOLDS} region-held-out folds")
        print("=" * 90)
        print(f"  {'Method':<40}{'Within-site AUC':>21}{'Overall AUC':>23}")
        per_site = {}
        for m in methods:
            res, per_site[m] = summarize(NAMES[m], scores[m], y, site)
            curve_rows.append(dict(frac=frac, train_sites=round(float(np.mean(n_train)), 1), method=m,
                                   **{k: round(float(v), 4) for k, v in res.items()}))
        print("\n  Paired comparison (same sites, within-site AUC):")
        paired("Prithvi: 4 seasons minus 1 date", per_site["prithvi_4season"], per_site["prithvi_1date"])
        paired("Random forest: 4 seasons minus 1 date", per_site["rf_4season"], per_site["rf_1date"])
        paired("4 seasons: Prithvi minus random forest", per_site["prithvi_4season"], per_site["rf_4season"])
        paired("Per-season Prithvi minus random forest", per_site["prithvi_4season_per"], per_site["rf_4season"])
        print("  A difference is convincing when its whole interval is above 0 (P(diff <= 0) below 0.025).")

    path = os.path.join(OUT_DIR, f"learning_curve_seasons{args.tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        w.writeheader()
        w.writerows(curve_rows)
    if len(fracs) > 1:
        print("\nLEARNING CURVE (within-site AUC)")
        print(f"  {'Sites':<8}" + "".join(f"{m:>22}" for m in methods))
        for frac in fracs:
            rows = {r["method"]: r for r in curve_rows if r["frac"] == frac}
            print(f"  {rows[methods[0]]['train_sites']:<8.0f}" + "".join(f"{rows[m]['within']:>22.3f}" for m in methods))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
