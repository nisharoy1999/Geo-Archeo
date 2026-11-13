# combine_seeds.py
#
# PURPOSE:
#   One honest result from several independent background draws.
#   Each run of pipeline/detect_seasons used a different random set of
#   background patches (seed). Single runs disagree because each site was
#   compared with only 3 backgrounds. Here every site's within-site AUC is
#   averaged over ALL runs (5 runs x 3 = 15 backgrounds per site), and the
#   methods are compared on that average with a paired bootstrap over sites.
#
#   Decided before looking at the pooled numbers: every run is included,
#   none is dropped, and the four comparisons are corrected for multiple
#   testing (Holm).
#
# USAGE (from the GeoArcheo folder):
#   python combine_seeds.py
#   python combine_seeds.py --tags "",_s7,_s11,_s19,_s42

import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score

NAMES = {"rf_1date": "Random forest, 1 date",
         "rf_4season": "Random forest, 4 seasons",
         "prithvi_1date": "Frozen Prithvi, 1 date",
         "prithvi_4season": "Frozen Prithvi, 4 seasons",
         "prithvi_4season_per": "Frozen Prithvi, 4 seasons, per season"}
COMPARISONS = [("Prithvi: 4 seasons minus 1 date", "prithvi_4season", "prithvi_1date"),
               ("Random forest: 4 seasons minus 1 date", "rf_4season", "rf_1date"),
               ("4 seasons: Prithvi minus random forest", "prithvi_4season", "rf_4season"),
               ("Per-season Prithvi minus random forest", "prithvi_4season_per", "rf_4season")]


def per_site_auc(path):
    """{method: {site_id: within-site AUC}} and {method: overall AUC} for one run."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["fold"] not in ("", "-1")]
    methods = [m for m in NAMES if m in rows[0]]
    by_site = {}
    for r in rows:
        by_site.setdefault(r["site_id"], []).append(r)
    within = {m: {} for m in methods}
    overall = {}
    for m in methods:
        for sid, rs in by_site.items():
            pos = [float(r[m]) for r in rs if r["patch"] == "site"]
            neg = [float(r[m]) for r in rs if r["patch"] == "background"]
            if pos and neg:
                within[m][sid] = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
        y = [r["patch"] == "site" for r in rows]
        overall[m] = roc_auc_score(y, [float(r[m]) for r in rows])
    return within, overall


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=",_s7,_s11,_s19,_s42")
    args = ap.parse_args()
    tags = args.tags.split(",")
    paths = [os.path.join("outputs", f"detect_predictions_seasons{t}.csv") for t in tags]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing: {missing}")

    runs = [per_site_auc(p) for p in paths]
    methods = [m for m in NAMES if all(m in w for w, _ in runs)]
    sites = sorted(set.intersection(*[set(w[methods[0]]) for w, _ in runs]))
    print(f"{len(runs)} runs, {len(sites)} sites present in every run "
          f"(~{3 * len(runs)} backgrounds per site in total)\n")

    print("PER RUN (within-site AUC), to show the spread between background draws")
    print(f"  {'Method':<40}" + "".join(f"{(t or 'seed 123'):>10}" for t in tags))
    for m in methods:
        print(f"  {NAMES[m]:<40}" + "".join(f"{np.mean(list(w[m].values())):>10.3f}" for w, _ in runs))

    pooled = {m: np.array([np.mean([w[m][s] for w, _ in runs]) for s in sites]) for m in methods}
    rng = np.random.default_rng(0)
    boot_idx = [rng.choice(len(sites), len(sites), replace=True) for _ in range(5000)]

    print("\n" + "=" * 92)
    print(f"POOLED OVER {len(runs)} BACKGROUND DRAWS  ({len(sites)} sites, 300 km regions held out)")
    print("=" * 92)
    print(f"  {'Method':<40}{'Within-site AUC':>22}{'Overall AUC (mean, range)':>30}")
    out = []
    for m in methods:
        b = [pooled[m][i].mean() for i in boot_idx]
        lo, hi = np.percentile(b, [2.5, 97.5])
        ov = [o[m] for _, o in runs]
        print(f"  {NAMES[m]:<40}{pooled[m].mean():>9.3f} [{lo:.2f}-{hi:.2f}]"
              f"{np.mean(ov):>15.3f} ({min(ov):.2f}-{max(ov):.2f})")
        out.append(dict(method=m, within=round(pooled[m].mean(), 4), within_lo=round(lo, 4),
                        within_hi=round(hi, 4), overall_mean=round(float(np.mean(ov)), 4), holm_p=""))

    print("\n  Paired comparison on pooled per-site AUC (Holm-corrected for 4 tests):")
    raw, lines = [], []
    for name, a, b in COMPARISONS:
        d = pooled[a] - pooled[b]
        bs = np.array([d[i].mean() for i in boot_idx])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = float(np.mean(bs <= 0))
        raw.append(max(p, 1 / len(bs)))
        lines.append((name, d.mean(), lo, hi, p))
    adj = holm(np.array(raw))
    for (name, dm, lo, hi, p), pa in zip(lines, adj):
        verdict = "CONVINCING" if pa < 0.025 else "not shown"
        print(f"  {name:<42}{dm:+.3f} [{lo:+.3f} to {hi:+.3f}]  P={p:.3f}  Holm P={pa:.3f}  {verdict}")
        out.append(dict(method=name, within=round(dm, 4), within_lo=round(lo, 4), within_hi=round(hi, 4),
                        overall_mean="", holm_p=round(float(pa), 4)))

    path = os.path.join("outputs", "seasons_pooled.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
