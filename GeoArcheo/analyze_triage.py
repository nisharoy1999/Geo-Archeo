# analyze_triage.py
#
# PURPOSE:
#   Turn detection scores into a statement a field archaeologist can use:
#   "survey only the flagged ground, skip the rest, and still find at least
#   (1 - alpha) of the sites", with a conformal guarantee.
#
# HOW:
#   For every held-out fold, the score threshold is calibrated ONLY on the
#   other folds' sites (split-conformal lower quantile):
#       t = the ceil-floor alpha*(n+1)-th smallest calibration site score
#   Patches scoring below t are "skipped". We then measure on the held-out
#   fold, which the threshold never saw:
#       recall   = share of held-out sites that were NOT skipped
#       skipped  = share of held-out background ground that was skipped
#   A good method keeps recall at or above the target while skipping as much
#   ground as possible. A useless (random) method can only skip about alpha.
#
#   Caveat for the paper: the guarantee assumes calibration and test sites are
#   exchangeable. Folds are different regions, so the achieved recall is
#   reported and checked, not just assumed.
#
# INPUT:   outputs/detect_predictions*.csv  (written by pipeline/detect_crossval)
# OUTPUT:  outputs/triage<tag>.csv          numbers for the paper table
#          outputs/triage<tag>.png          figure: ground skipped vs target recall
#
# USAGE (from the GeoArcheo folder, runs on the Mac, no GPU):
#   python analyze_triage.py --pred outputs/detect_predictions_v2.csv --tag _v2

import argparse
import csv
import math
import os

import numpy as np

BASE_COLS = {"site_id", "patch", "region_group", "fold"}
NAMES = {"random_forest": "Random forest (spectral)",
         "frozen_probe": "Frozen Prithvi + probe",
         "lora_finetuned": "Prithvi + LoRA",
         "rf_1date": "Random forest, 1 date",
         "rf_4season": "Random forest, 4 seasons",
         "prithvi_1date": "Prithvi, 1 date",
         "prithvi_4season": "Prithvi, 4 seasons",
         "prithvi_4season_per": "Prithvi, 4 seasons (per)"}


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["fold"] not in ("", "-1")]
    methods = [c for c in rows[0] if c not in BASE_COLS]
    fold = np.array([int(r["fold"]) for r in rows])
    is_site = np.array([r["patch"] == "site" for r in rows])
    site_id = np.array([r["site_id"] for r in rows])
    scores = {m: np.array([float(r[m]) for r in rows]) for m in methods}
    return methods, fold, is_site, site_id, scores


def conformal_threshold(cal_site_scores, alpha):
    """Largest t such that P(new site score >= t) >= 1 - alpha (exchangeability)."""
    s = np.sort(cal_site_scores)
    k = math.floor(alpha * (len(s) + 1))          # number of sites allowed below t
    return -np.inf if k < 1 else s[k - 1]


def triage(score, fold, is_site, alpha):
    """Out-of-fold thresholds; returns per-patch 'kept' flags."""
    kept = np.zeros(len(score), dtype=bool)
    for f in np.unique(fold):
        cal = (fold != f) & is_site
        t = conformal_threshold(score[cal], alpha)
        te = fold == f
        kept[te] = score[te] >= t
    return kept


def bootstrap(kept, is_site, site_id, n=1000):
    """95% intervals for recall and skipped share, resampling sites
    (a site travels with its own background patches)."""
    ids = np.unique(site_id)
    rows_of = {s: np.where(site_id == s)[0] for s in ids}
    rng = np.random.default_rng(0)
    rec, skp = [], []
    for _ in range(n):
        rows = np.concatenate([rows_of[s] for s in rng.choice(ids, len(ids), replace=True)])
        k, st = kept[rows], is_site[rows]
        rec.append(k[st].mean())
        skp.append(1 - k[~st].mean())
    return np.percentile(rec, [2.5, 97.5]), np.percentile(skp, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join("outputs", "detect_predictions.csv"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--alphas", default="0.05,0.1,0.2,0.3")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    methods, fold, is_site, site_id, scores = load(args.pred)
    print(f"{args.pred}: {is_site.sum()} sites, {(~is_site).sum()} background patches, "
          f"{len(np.unique(fold))} folds\n")
    print(f"  {'Method':<26}{'Target':>8}{'Sites found':>22}{'Ground skipped':>24}")

    out_rows = []
    for m in methods:
        for a in alphas:
            kept = triage(scores[m], fold, is_site, a)
            recall = kept[is_site].mean()
            skipped = 1 - kept[~is_site].mean()
            (rl, rh), (sl, sh) = bootstrap(kept, is_site, site_id)
            print(f"  {NAMES.get(m, m):<26}{1 - a:>8.0%}"
                  f"{recall:>11.1%} [{rl:.0%}-{rh:.0%}]{skipped:>13.1%} [{sl:.0%}-{sh:.0%}]")
            out_rows.append(dict(method=m, target_recall=1 - a, recall=round(recall, 4),
                                 recall_lo=round(rl, 4), recall_hi=round(rh, 4),
                                 ground_skipped=round(skipped, 4),
                                 skipped_lo=round(sl, 4), skipped_hi=round(sh, 4),
                                 random_baseline_skip=a))
        print()
    print("  'Ground skipped' for a model with no skill would be about 1 - target")
    print("  (e.g. ~10% at a 90% target). Anything above that is the model's value.")

    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", f"triage{args.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.2))
        grid = np.linspace(0.02, 0.5, 25)
        for m in methods:
            ys = [1 - triage(scores[m], fold, is_site, a)[~is_site].mean() for a in grid]
            ax.plot(1 - grid, ys, lw=2, label=NAMES.get(m, m))
        ax.plot(1 - grid, grid, "k--", lw=1, label="No skill")
        ax.set_xlabel("Target share of sites found (conformal guarantee)")
        ax.set_ylabel("Share of ground that can be skipped")
        ax.set_title("Survey triage under region-held-out testing")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        png_path = os.path.join("outputs", f"triage{args.tag}.png")
        fig.savefig(png_path, dpi=200)
        print(f"\nSaved {csv_path} and {png_path}")
    except ImportError:
        print(f"\nSaved {csv_path} (matplotlib not installed, no figure)")


if __name__ == "__main__":
    main()
