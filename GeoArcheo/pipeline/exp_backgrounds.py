# pipeline/exp_backgrounds.py
#
# PURPOSE:
#   Reviewer request: the backgrounds are random patches at least 2.5 km from a known site, and
#   some may hold unrecorded sites. Does the result depend on that choice? Does the model find
#   archaeology, or just unusual terrain?
#
#   Conditions (build them with run_background_builds.sh, which calls build_season_patches.py):
#       main            random backgrounds >= 2.5 km from known sites   (the paper)
#       km5, km10       random backgrounds >= 5 km / >= 10 km from known sites (a bigger buffer)
#       matched_km2.5   backgrounds chosen to have the SAME whole-patch terrain roughness and
#       matched_km5     slope as the site (>= 2.5 km / >= 5 km from known sites)
#
#   For each condition, every model is trained and tested with 300 km regions held out:
#       Colour (1 date) | Elevation only | Colour + elevation
#       Patch-level terrain only (roughness + slope of the 2.2 km window): a "just unusual terrain" baseline
#       Elevation without patch-level terrain | Local height cues only (80 m and 160 m windows)
#
#   HOW TO READ IT
#     - If elevation keeps most of its AUC with terrain-matched backgrounds, the result is not
#       just "sites sit on rougher ground than random ground".
#     - If patch-level terrain alone scores well in the random conditions and falls to ~0.5 when
#       matched, that baseline was picking up unusual terrain, and the matching worked.
#     - If AUCs rise with distance (km5, km10), backgrounds near sites may have hidden some signal
#       (or unrecorded sites); if they fall, nearby backgrounds were easier.
#
# USAGE (from the GeoArcheo folder, after run_background_builds.sh):
#   python -m pipeline.exp_backgrounds
#   python -m pipeline.exp_backgrounds --trees 200 --seeds 2     # quick try
#   python -m pipeline.exp_backgrounds --conditions main=data/season_patches km5=data/bg/km5/season_patches

import argparse

import numpy as np

from pipeline.exp_common import (DEM_NAMES, Pool, clean, cluster_boot, cv_scores, fmt_ci, holm, load_features,
                                 make_folds, run_paths, site_auc, site_table, write_csv)

DEFAULT = ["main=data/season_patches",
           "km5=data/bg/km5/season_patches",
           "km10=data/bg/km10/season_patches",
           "matched_km2.5=data/bg/matched_km2.5/season_patches",
           "matched_km5=data/bg/matched_km5/season_patches"]
RUF, SLP = DEM_NAMES.index("roughness_patch"), DEM_NAMES.index("slope_patch")
N = len(DEM_NAMES)
MODELS = ["Colour (1 date)", "Elevation only", "Colour + elevation", "Patch-level terrain only",
          "Elevation without patch-level terrain", "Local height cues only (80, 160 m)"]


def matrices(F):
    D = F["dem"]
    keep = [c for c in range(N) if c not in (RUF, SLP)]
    colour = F["season"][:, 0]
    return {"Colour (1 date)": clean(colour),
            "Elevation only": clean(D),
            "Colour + elevation": clean(np.hstack([colour, D])),
            "Patch-level terrain only": clean(D[:, [SLP, RUF]]),
            "Elevation without patch-level terrain": clean(D[:, keep]),
            "Local height cues only (80, 160 m)": clean(D[:, [0, 1, 2, 3]])}


def balance(F):
    """How different is a site's terrain from its own backgrounds? Mean |background - site| of the
    patch-level roughness and slope, in units of the spread across all patches."""
    D = F["dem"]
    out = {}
    for name, c in (("roughness", RUF), ("slope", SLP)):
        v = D[:, c]
        sd = np.nanstd(v) + 1e-9
        diffs = []
        for s in np.unique(F["site"]):
            m = F["site"] == s
            site_v = v[m & (F["y"] == 1)]
            bg_v = v[m & (F["y"] == 0)]
            if len(site_v) and len(bg_v) and np.isfinite(site_v).all() and np.isfinite(bg_v).any():
                diffs.append(np.nanmean(np.abs(bg_v - site_v[0])) / sd)
        out[name] = float(np.mean(diffs)) if diffs else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=DEFAULT, help="name=path-prefix pairs")
    ap.add_argument("--extra-tag", default="_fabdem")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=500)
    args = ap.parse_args()

    pools, info = {}, {}
    for spec in args.conditions:
        name, base = spec.split("=", 1)
        runs = run_paths(base, args.seeds)
        if not runs:
            print(f"[{name}] no files at {base}[_sN].npz: skipped (build it with run_background_builds.sh)")
            continue
        pool = Pool()
        bal, n_bg, n_sites = [], [], []
        for seed, path in runs:
            F = load_features(path, args.extra_tag)
            folds = make_folds(F["y"], F["grp"])
            per_model = {m: site_auc(cv_scores(X, F["y"], folds, args.trees), F) for m, X in matrices(F).items()}
            pool.add(per_model, F)
            bal.append(balance(F))
            n_bg.append(int((F["y"] == 0).sum())); n_sites.append(int((F["y"] == 1).sum()))
            print(f"[{name}] seed {seed}: {n_sites[-1]} sites, {n_bg[-1]} backgrounds; "
                  f"elevation only {np.mean(list(per_model['Elevation only'].values())):.3f}", flush=True)
        pools[name] = pool
        info[name] = dict(draws=len(runs), sites=len(pool.sites()), backgrounds=int(np.mean(n_bg)),
                          rough=float(np.mean([b["roughness"] for b in bal])),
                          slope=float(np.mean([b["slope"] for b in bal])))
    if "main" not in pools:
        raise SystemExit("The 'main' condition is needed as the reference.")

    names = list(pools)
    print("\n" + "=" * 118)
    print("BACKGROUND DESIGN: within-site AUC, 300 km regions held out, pooled over the draws (95% interval: regions resampled)")
    print("=" * 118)
    print(f"  {'Condition':<16}{'sites':>6}{'bg/draw':>9}  " + "".join(f"{m[:24]:>26}" for m in MODELS[:3]))
    rows = []
    for n in names:
        p = pools[n]
        ri, nr = p.region_index()
        cells = []
        for m in MODELS:
            v, lo, hi, _ = cluster_boot(p.per_site(m), ri, nr)
            rows.append(dict(condition=n, model=m, within_site_auc=round(v, 4), lo=round(lo, 4), hi=round(hi, 4),
                             sites=info[n]["sites"], draws=info[n]["draws"]))
            cells.append(fmt_ci(v, lo, hi))
        print(f"  {n:<16}{info[n]['sites']:>6}{info[n]['backgrounds']:>9}  " + "".join(f"{c:>26}" for c in cells[:3]))
    print(f"\n  {'Condition':<16}" + "".join(f"{m[:34]:>38}" for m in MODELS[3:]))
    for n in names:
        sub = [r for r in rows if r["condition"] == n]
        print(f"  {n:<16}" + "".join(f"{fmt_ci(r['within_site_auc'], r['lo'], r['hi']):>38}" for r in sub[3:]))
    write_csv("outputs/exp_backgrounds_auc.csv", rows)

    print("\n  Elevation gain (colour + elevation minus colour) in each condition:")
    gain_rows = []
    for n in names:
        p = pools[n]
        ri, nr = p.region_index()
        d = p.per_site("Colour + elevation") - p.per_site("Colour (1 date)")
        v, lo, hi, means = cluster_boot(d, ri, nr)
        pv = max(float(np.mean(means <= 0)), 1.0 / len(means))
        print(f"    {n:<16}{v:+.3f} [{lo:+.3f} to {hi:+.3f}]   one-sided P={pv:.3f}")
        gain_rows.append(dict(condition=n, elevation_gain=round(v, 4), lo=round(lo, 4), hi=round(hi, 4), p=round(pv, 4)))
    write_csv("outputs/exp_backgrounds_gain.csv", gain_rows)

    # each condition against the paper's design, on the sites both contain
    print("\n  Each condition minus main, on the sites both have (Holm-corrected within each model):")
    base = pools["main"]
    cmp_rows = []
    for m in MODELS:
        res = []
        for n in names:
            if n == "main":
                continue
            p = pools[n]
            common = sorted(set(base.sites()) & set(p.sites()))
            if len(common) < 10:
                continue
            ia = {s: i for i, s in enumerate(base.sites())}
            ib = {s: i for i, s in enumerate(p.sites())}
            d = np.array([p.per_site(m)[ib[s]] - base.per_site(m)[ia[s]] for s in common])
            reg = np.array([base.region[s] for s in common])
            u = np.unique(reg)
            ri = np.array([np.where(u == r)[0][0] for r in reg])
            v, lo, hi, means = cluster_boot(d, ri, len(u))
            res.append((n, len(common), v, lo, hi, max(float(np.mean(means <= 0)), 1.0 / len(means)),
                        max(float(np.mean(means >= 0)), 1.0 / len(means))))
        if not res:
            continue
        adj_lo = holm(np.array([r[5] for r in res]))        # "lower than main"? uses P(diff <= 0) -> improvement
        adj_hi = holm(np.array([r[6] for r in res]))        # P(diff >= 0) -> drop
        print(f"    {m}")
        for (n, k, v, lo, hi, plo, phi), a, b in zip(res, adj_lo, adj_hi):
            flag = "higher than main" if a < 0.025 else ("lower than main" if b < 0.025 else "no clear change")
            print(f"      {n:<16}n={k:>3}  {v:+.3f} [{lo:+.3f} to {hi:+.3f}]   {flag}")
            cmp_rows.append(dict(model=m, condition=n, sites=k, diff=round(v, 4), lo=round(lo, 4), hi=round(hi, 4),
                                 holm_p_higher=round(float(a), 4), holm_p_lower=round(float(b), 4), verdict=flag))
    write_csv("outputs/exp_backgrounds_vs_main.csv", cmp_rows)

    print("\n  Did the matching work? Mean |background - site| terrain difference (in spreads; smaller = better matched):")
    print(f"    {'Condition':<16}{'roughness':>12}{'slope':>10}")
    bal_rows = []
    for n in names:
        print(f"    {n:<16}{info[n]['rough']:>12.2f}{info[n]['slope']:>10.2f}")
        bal_rows.append(dict(condition=n, roughness_gap=round(info[n]["rough"], 3), slope_gap=round(info[n]["slope"], 3)))
    write_csv("outputs/exp_backgrounds_balance.csv", bal_rows)


if __name__ == "__main__":
    main()
