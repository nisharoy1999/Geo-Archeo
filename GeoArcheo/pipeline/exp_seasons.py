# pipeline/exp_seasons.py
#
# PURPOSE:
#   Reviewer request: investigate why four-season imagery does not clearly beat one date.
#   All numbers are within-site AUC with whole 300 km regions held out, pooled over the
#   background draws, with 95% intervals from resampling regions.
#
#   1. SINGLE SEASONS      winter (Jan-Feb, the "1 date" of the paper), spring, summer, autumn
#   2. WHAT THE EXTRA SEASONS ADD   four seasons with and without the "change over the year"
#                          features; change features alone; dropping one season at a time
#   3. CLOUD / SNOW        does the four-season gain concentrate in sites whose winter image is
#                          cloud- or snow-affected? (a cloud proxy is computed from the blue band)
#   4. SITE TYPE           does the seasonal gain differ between settlements and burial mounds?
#   5. REDUNDANCY          how much of the change-over-the-year information can already be
#                          predicted from the one-date colour features, and how much does
#                          shuffling each season's features hurt on held-out regions?
#   6. WITH ELEVATION      does any season add anything once elevation is in the model?
#
# CLOUD PROXY (check it!): a pixel counts as "bright" when its blue band is above --blue-thr
#   (default 2000, i.e. reflectance 0.2 if the data are scaled x10000; rounded to a multiple of 250). Clouds and snow are
#   both bright in blue. A patch is "affected" in a season when more than --frac of its pixels
#   are bright. The script prints the distribution so you can choose sensible values. If your
#   band order is not (blue, green, red, ...) pass --blue-band.
#
# INPUT:   data/season_patches[_sN].npz and ..._extra_fabdem.npz
# OUTPUT:  outputs/exp_seasons_*.csv
#
# USAGE (from the GeoArcheo folder; ~15-30 min):
#   python -m pipeline.exp_seasons
#   python -m pipeline.exp_seasons --trees 200 --seeds 2     # quick try

import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

import pipeline.exp_common as ec
from pipeline.exp_common import (SEASONS, Pool, bright_at, clean, cluster_boot, cv_scores, fmt_ci, load_features, make_folds,
                                 report_comparisons, run_paths, site_auc, site_kinds, site_table, write_csv)
from pipeline.exp_importance import oof_with_permutation, within_site_mean

W, SP, SU, AU = SEASONS                                   # season names


def models_for(F):
    S, C, D = F["season"], F["change"], F["dem"]
    n = len(S)
    flat = S.reshape(n, -1)
    m = {
        f"Winter only ({W})": S[:, 0], f"Spring only ({SP})": S[:, 1],
        f"Summer only ({SU})": S[:, 2], f"Autumn only ({AU})": S[:, 3],
        "Four seasons + change features": np.hstack([flat, C]),
        "Four seasons, no change features": flat,
        "Change features only": C,
    }
    for s, name in enumerate(SEASONS):
        m[f"Three seasons, without {name}"] = np.hstack([S[:, t] for t in range(4) if t != s])
    m["Elevation only"] = D
    for s, name in enumerate(SEASONS):
        m[f"Elevation + {name}"] = np.hstack([D, S[:, s]])
    m["Elevation + four seasons + change"] = np.hstack([D, flat, C])
    m["Elevation + four seasons, no change"] = np.hstack([D, flat])
    return {k: clean(v) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/season_patches")
    ap.add_argument("--extra-tag", default="_fabdem")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--blue-thr", type=float, default=2000, help="raw blue value above which a pixel is 'bright'")
    ap.add_argument("--blue-band", type=int, default=0, help="index of the blue band in the image files")
    ap.add_argument("--frac", type=float, default=0.10, help="share of bright pixels above which a patch is 'affected'")
    args = ap.parse_args()

    runs = run_paths(args.base, args.seeds)
    if not runs:
        raise SystemExit(f"No files found for {args.base}[_sN].npz")
    print(f"{len(runs)} background draws: {[s for s, _ in runs]}; {args.trees} trees\n")

    pool, perm_groups, F0 = Pool(), [], None
    for seed, path in runs:
        print(f"seed {seed}: features + {0} models...", flush=True)
        F = load_features(path, args.extra_tag, blue_band=args.blue_band)
        if F0 is None:
            F0 = F
        folds = make_folds(F["y"], F["grp"])
        per_model, M = {}, models_for(F)
        for name, X in M.items():
            sc, models = cv_scores(X, F["y"], folds, args.trees, return_models=True)
            per_model[name] = site_auc(sc, F)
            if name == "Four seasons + change features":
                base = within_site_mean(sc, F)
                n_s = F["season"].shape[2]
                n_c = F["change"].shape[1]
                cols = {SEASONS[s]: list(range(s * n_s, (s + 1) * n_s)) for s in range(4)}
                cols["Change over the year"] = list(range(4 * n_s, 4 * n_s + n_c))
                rng = np.random.default_rng(int(seed))
                perm_groups.append({g: np.mean([base - within_site_mean(
                    oof_with_permutation(models, X, F["y"], folds, c, rng), F) for _ in range(5)])
                    for g, c in cols.items()})
        pool.add(per_model, F)
        print(f"  winter {np.mean(list(per_model[f'Winter only ({W})'].values())):.3f}  "
              f"four seasons {np.mean(list(per_model['Four seasons + change features'].values())):.3f}", flush=True)

    sites = pool.sites()
    ri, nr = pool.region_index()
    pos = site_table(F0)
    sid = {str(F0["ids"][F0["site"][i]]): i for i in pos}
    kinds = site_kinds(sites)

    # ------------------------------------------------------------------ 1. table
    print("\n" + "=" * 100)
    print(f"SEASONS, held-out regions, within-site AUC pooled over {len(runs)} draws ({len(sites)} sites, {nr} regions)")
    print("=" * 100)
    rows = []
    for name in pool.runs[0]:
        m, lo, hi, _ = cluster_boot(pool.per_site(name), ri, nr)
        print(f"  {name:<42}{fmt_ci(m, lo, hi)}")
        rows.append(dict(model=name, within_site_auc=round(m, 4), lo=round(lo, 4), hi=round(hi, 4)))
    write_csv("outputs/exp_seasons_auc.csv", rows)

    # ------------------------------------------------------------------ 2. comparisons
    win, four, fnc = f"Winter only ({W})", "Four seasons + change features", "Four seasons, no change features"
    comps = [("Four seasons + change minus winter only", four, win),
             ("Four seasons, no change minus winter only", fnc, win),
             ("Change features minus none (four seasons + change minus no change)", four, fnc),
             ("Change features only minus chance (0.5)", "Change features only", None)]
    comps += [(f"{SEASONS[s]} only minus winter only", f"{n} only ({SEASONS[s]})", win)
              for s, n in ((1, "Spring"), (2, "Summer"), (3, "Autumn"))]
    comps += [(f"Four seasons minus three seasons without {n}", four, f"Three seasons, without {n}") for n in SEASONS]
    comps += [("Elevation + four seasons + change minus elevation only", "Elevation + four seasons + change", "Elevation only"),
              ("Elevation + four seasons, no change minus elevation only", "Elevation + four seasons, no change", "Elevation only")]
    comps += [(f"Elevation + {n} minus elevation only", f"Elevation + {n}", "Elevation only") for n in SEASONS]
    write_csv("outputs/exp_seasons_comparisons.csv", report_comparisons(pool, comps, "Paired comparisons"))

    # ------------------------------------------------------------------ 3. cloud / snow
    print("\n" + "=" * 100)
    print(f"CLOUD / SNOW PROXY (blue band > {ec.BLUE_GRID[np.argmin(np.abs(ec.BLUE_GRID - args.blue_thr))]:g}; patch affected if > {args.frac:.0%} of pixels are bright)")
    print("=" * 100)
    B0 = bright_at(F0, args.blue_thr)
    bright = np.array([[B0[sid[s], t] for t in range(4)] for s in sites])
    tile_of = {s: F0["tile"][sid[s]] for s in sites}
    print(f"  {'Season':<10}{'median':>9}{'p90':>8}{'p99':>8}{'max':>8}   sites affected at >2% / 5% / 10% / 25%")
    for t, name in enumerate(SEASONS):
        b = bright[:, t]
        print(f"  {name:<10}{np.median(b):>9.3f}{np.percentile(b, 90):>8.3f}{np.percentile(b, 99):>8.3f}{b.max():>8.3f}"
              f"   {int((b > .02).sum()):>4} / {int((b > .05).sum())} / {int((b > .10).sum())} / {int((b > .25).sum())}")
    flagged = bright > args.frac
    tiles = np.array([tile_of[s] for s in sites])
    print(f"\n  Tiles whose median site patch is affected (>{args.frac:.0%}), by season:")
    for t, name in enumerate(SEASONS):
        bad = [u for u in np.unique(tiles) if np.median(bright[tiles == u, t]) > args.frac]
        print(f"    {name:<10}{len(bad)} of {len(np.unique(tiles))} tiles")
    print("  If these counts look wrong, adjust --blue-thr / --frac / --blue-band and re-run (features are cached).")

    gain = pool.per_site(four) - pool.per_site(win)
    grp_rows = []

    def block(label, mask):
        if mask.sum() < 5:
            print(f"    {label:<44} n={int(mask.sum()):>3}  (too few sites)")
            return
        nreg = len(np.unique(ri[mask]))
        a = cluster_boot(pool.per_site(win), ri, nr, mask)
        b = cluster_boot(pool.per_site(four), ri, nr, mask)
        g = cluster_boot(gain, ri, nr, mask)
        print(f"    {label:<44} n={int(mask.sum()):>3} ({nreg:>2} regions)  winter {fmt_ci(*a[:3])}  "
              f"four seasons {fmt_ci(*b[:3])}  gain {g[0]:+.3f} [{g[1]:+.3f} to {g[2]:+.3f}]")
        grp_rows.append(dict(subset=label, sites=int(mask.sum()), regions=nreg,
                             winter_auc=round(a[0], 4), four_auc=round(b[0], 4),
                             gain=round(g[0], 4), gain_lo=round(g[1], 4), gain_hi=round(g[2], 4)))

    print("\n  Does the four-season gain concentrate where the WINTER image is cloud/snow-affected?")
    block("winter image clean", ~flagged[:, 0])
    block("winter image cloud/snow-affected", flagged[:, 0])
    print("\n  Same, using the tile (the nine winter tiles of the reviewer's comment):")
    tile_bad = np.array([np.median(bright[tiles == u, 0]) > args.frac for u in tiles])
    block("winter tile clean", ~tile_bad)
    block("winter tile cloud/snow-affected", tile_bad)
    print("\n  Each single season, on sites where THAT season is clean vs affected:")
    for t, name in enumerate(SEASONS):
        key = [k for k in pool.runs[0] if k.endswith(f"({name})") and "only" in k][0]
        for lab, mk in ((f"{name} clean", ~flagged[:, t]), (f"{name} affected", flagged[:, t])):
            if mk.sum() < 5:
                print(f"    {lab:<44} n={int(mk.sum()):>3}  (too few sites)")
                continue
            m, lo, hi, _ = cluster_boot(pool.per_site(key), ri, nr, mk)
            print(f"    {lab:<44} n={int(mk.sum()):>3}  single-season AUC {fmt_ci(m, lo, hi)}")
            grp_rows.append(dict(subset=f"single season model | {lab}", sites=int(mk.sum()), four_auc=round(m, 4)))

    # ------------------------------------------------------------------ 4. site type
    print("\n  Settlements vs burial mounds:")
    for k in sorted(set(kinds)):
        if k != "?":
            block(f"site type: {k}", kinds == k)
    print("  (site types come from the EAMENA file; '?' means the type was not found)")
    write_csv("outputs/exp_seasons_subsets.csv", grp_rows)

    # ------------------------------------------------------------------ 5. redundancy
    print("\n" + "=" * 100)
    print("REDUNDANCY: is the change-over-the-year information already in the one-date colour features?")
    print("=" * 100)
    y = F0["y"]
    Xw = (F0["season"][:, 0] - F0["season"][:, 0].mean(0)) / (F0["season"][:, 0].std(0) + 1e-6)
    C = F0["change"]
    r2 = []
    for j in range(C.shape[1]):
        pred = np.zeros(len(C))
        for tr, te in GroupKFold(n_splits=ec.N_FOLDS).split(Xw, y, F0["grp"]):
            pred[te] = Ridge(alpha=10.0).fit(Xw[tr], C[tr, j]).predict(Xw[te])
        r2.append(1 - ((C[:, j] - pred) ** 2).sum() / ((C[:, j] - C[:, j].mean()) ** 2).sum())
    r2 = np.array(r2)
    ranks = np.apply_along_axis(rankdata, 0, np.hstack([Xw, C]))
    cc = np.corrcoef(ranks, rowvar=False)[Xw.shape[1]:, :Xw.shape[1]]
    print(f"  Predicting each of the {C.shape[1]} change features from the winter features, regions held out (ridge):")
    print(f"    held-out R^2: median {np.median(r2):.2f}, range {r2.min():.2f} to {r2.max():.2f}; "
          f"{int((r2 > .5).sum())} of {len(r2)} above 0.5")
    print(f"    strongest rank correlation with any winter feature: median {np.median(np.abs(cc).max(1)):.2f}, "
          f"max {np.abs(cc).max():.2f}")
    print("  Held-out permutation importance in the four-season model (drop in within-site AUC when shuffled):")
    red_rows = [dict(item="change features predicted from winter: median held-out R2", value=round(float(np.median(r2)), 3)),
                dict(item="change features with held-out R2 > 0.5 (count)", value=int((r2 > .5).sum()))]
    for g in perm_groups[0]:
        v = float(np.mean([d[g] for d in perm_groups]))
        print(f"    {g:<26}{v:>+8.4f}")
        red_rows.append(dict(item=f"permutation drop: {g}", value=round(v, 4)))
    write_csv("outputs/exp_seasons_redundancy.csv", red_rows)
    print("  A low or negative drop for a group means the model does not rely on it on held-out regions.")


if __name__ == "__main__":
    main()
