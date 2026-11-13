# pipeline/exp_importance.py
#
# PURPOSE:
#   Reviewer request: explain WHY bare-earth elevation helps, with importance measured
#   only on held-out regions, and test it with ablations.
#
#   For every background draw and every fold (300 km regions held out):
#     1. impurity importance   from the forest trained on the TRAINING regions
#     2. permutation importance on the HELD-OUT regions: shuffle one feature (or one group
#                              of related features) in the test fold and see how much the
#                              within-site AUC drops
#     3. SHAP                  mean |SHAP| of each feature on the held-out regions (needs `pip install shap`)
#   and the ablations (all scored on held-out regions, within-site AUC):
#     - all elevation features
#     - leave one feature out        (15 runs)
#     - one feature alone            (15 runs)
#     - drop a window size, drop all "centre height" or all "peak height", slope/roughness only
#     - top-k features, where the features are CHOSEN INSIDE EACH FOLD FROM THE TRAINING
#       REGIONS ONLY (no test-region information is used to pick them)
#     - colour alone vs colour + each elevation group   (with --with-colour)
#
# INPUT:   data/season_patches[_sN].npz and data/season_patches[_sN]_extra_fabdem.npz
# OUTPUT:  outputs/exp_importance_*.csv
#
# USAGE (from the GeoArcheo folder; ~20-40 min on a laptop, use --trees 200 --seeds 2 to try it first):
#   pip install shap            # optional
#   python -m pipeline.exp_importance --with-colour

import argparse

import numpy as np
from scipy.stats import spearmanr

from pipeline.exp_common import (DEM_NAMES, Pool, cluster_boot, cv_scores, fmt_ci, forest, load_features,
                                 make_folds, run_paths, site_auc, write_csv, clean, report_comparisons)

GROUPS = {
    "window 80 m (centre + peak)": [0, 1],
    "window 160 m (centre + peak)": [2, 3],
    "window 320 m (centre + peak)": [4, 5],
    "window 640 m (centre + peak)": [6, 7],
    "all centre-height features": [0, 2, 4, 6],
    "all peak-height features": [1, 3, 5, 7],
    "slope and roughness": [8, 9, 10, 11, 12, 13, 14],
}


def within_site_mean(score, F):
    v = list(site_auc(score, F).values())
    return float(np.mean(v))


def oof_with_permutation(models, X, y, folds, cols, rng):
    """Out-of-fold scores when the columns `cols` are shuffled (jointly) inside each TEST fold."""
    score = np.zeros(len(y))
    for (tr, te), (rf, fcols) in zip(folds, models):
        Xte = X[te].copy()
        perm = rng.permutation(len(te))
        Xte[:, cols] = Xte[perm][:, cols]
        score[te] = rf.predict_proba(Xte[:, fcols])[:, 1]
    return score


def top_k_selector(k, trees):
    """Choose k features from the TRAINING regions only (impurity importance of a forest fit there)."""
    def pick(tr, X, y):
        rf = forest(trees, 99).fit(X[tr][:, :len(DEM_NAMES)], y[tr])
        return [int(c) for c in np.argsort(rf.feature_importances_)[::-1][:k]]
    return pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/season_patches")
    ap.add_argument("--extra-tag", default="_fabdem")
    ap.add_argument("--seeds", type=int, default=5, help="how many background draws to use (max 5)")
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5, help="shuffles per permutation-importance estimate")
    ap.add_argument("--with-colour", action="store_true", help="also test colour + each elevation group")
    args = ap.parse_args()

    runs = run_paths(args.base, args.seeds)
    if not runs:
        raise SystemExit(f"No files found for {args.base}[_sN].npz")
    print(f"{len(runs)} background draws: {[s for s, _ in runs]}; {args.trees} trees\n")

    n_feat = len(DEM_NAMES)
    imp_tr, imp_perm, imp_group, imp_shap = [], [], [], []
    pool = Pool()
    set_defs = {}                                  # name -> columns (None = special)
    set_defs["All 15 elevation features"] = list(range(n_feat))
    for j in range(n_feat):
        set_defs[f"Leave out: {DEM_NAMES[j]}"] = [c for c in range(n_feat) if c != j]
    for j in range(n_feat):
        set_defs[f"Only: {DEM_NAMES[j]}"] = [j]
    for g, cols in GROUPS.items():
        set_defs[f"Leave out group: {g}"] = [c for c in range(n_feat) if c not in cols]
        if len(cols) < n_feat:
            set_defs[f"Only group: {g}"] = list(cols)
    # the cleaned matrix has one extra column (a 'missing' flag) after the 15 features: always kept
    flag = n_feat

    try:
        import shap
    except Exception:
        shap = None
        print("(shap is not installed: skipping SHAP. `pip install shap` to add it.)\n")

    for seed, path in runs:
        print(f"seed {seed}: loading features...", flush=True)
        F = load_features(path, args.extra_tag)
        y = F["y"]
        Xd = clean(F["dem"])                       # 15 features + missing flag
        folds = make_folds(y, F["grp"])
        per_model = {}

        # ---- full model, importance on held-out regions
        score, models = cv_scores(Xd, y, folds, args.trees, return_models=True)
        per_model["All 15 elevation features"] = site_auc(score, F)
        base = within_site_mean(score, F)
        rng = np.random.default_rng(int(seed))
        imp_tr.append(np.mean([m.feature_importances_[:n_feat] for m, _ in models], axis=0))
        perm = np.zeros(n_feat)
        for j in range(n_feat):
            perm[j] = np.mean([base - within_site_mean(oof_with_permutation(models, Xd, y, folds, [j], rng), F)
                               for _ in range(args.repeats)])
        imp_perm.append(perm)
        grp_drop = {}
        for g, cols in GROUPS.items():
            grp_drop[g] = np.mean([base - within_site_mean(oof_with_permutation(models, Xd, y, folds, cols, rng), F)
                                   for _ in range(args.repeats)])
        imp_group.append(grp_drop)
        if shap is not None:
            acc = np.zeros(Xd.shape[1])
            n_seen = 0
            for (tr, te), (rf, _) in zip(folds, models):
                sv = shap.TreeExplainer(rf).shap_values(Xd[te])
                sv = np.asarray(sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv))
                acc += np.abs(sv).sum(0)
                n_seen += len(te)
            imp_shap.append((acc / n_seen)[:n_feat])

        # ---- ablations (each scored on held-out regions)
        for name, cols in set_defs.items():
            if name == "All 15 elevation features":
                continue
            sc = cv_scores(Xd, y, folds, args.trees, columns=cols + [flag])
            per_model[name] = site_auc(sc, F)
        for k in (1, 2, 3, 5, 8):
            sc = cv_scores(Xd, y, folds, args.trees, columns=lambda tr, X, yy, k=k: top_k_selector(k, args.trees)(tr, X, yy) + [flag])
            per_model[f"Top {k} features (chosen on training regions only)"] = site_auc(sc, F)
        if args.with_colour:
            Xc = clean(F["season"][:, 0, :])                              # one date (winter), as in the paper
            sc = cv_scores(Xc, y, folds, args.trees)
            per_model["Colour alone (1 date)"] = site_auc(sc, F)
            for g, cols in GROUPS.items():
                Xcg = np.hstack([Xc, Xd[:, cols + [flag]]])
                sc = cv_scores(Xcg, y, folds, args.trees)
                per_model[f"Colour + {g}"] = site_auc(sc, F)
            Xall = np.hstack([Xc, Xd])
            per_model["Colour + all 15 elevation features"] = site_auc(cv_scores(Xall, y, folds, args.trees), F)
        pool.add(per_model, F)
        print(f"  seed {seed}: all-elevation within-site AUC {base:.3f}", flush=True)

    # ------------------------------------------------------------------ report
    mean = lambda lst: np.mean(np.stack(lst), axis=0)
    rows_imp = []
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE (elevation features only; importance on HELD-OUT regions)")
    print("=" * 100)
    print(f"  {'Feature':<28}{'permutation drop in AUC':>26}{'impurity (train regions)':>28}"
          + (f"{'mean |SHAP| (held-out)':>26}" if imp_shap else ""))
    P, I = mean(imp_perm), mean(imp_tr)
    S = mean(imp_shap) if imp_shap else None
    for j in np.argsort(P)[::-1]:
        line = f"  {DEM_NAMES[j]:<28}{P[j]:>+26.4f}{I[j]:>28.3f}" + (f"{S[j]:>26.4f}" if S is not None else "")
        print(line)
        rows_imp.append(dict(feature=DEM_NAMES[j], permutation_drop=round(float(P[j]), 5),
                             impurity_train=round(float(I[j]), 5),
                             **({"mean_abs_shap_heldout": round(float(S[j]), 5)} if S is not None else {})))
    write_csv("outputs/exp_importance_features.csv", rows_imp)

    print("\n  Permutation importance of GROUPS of related features (shuffled together):")
    rows_g = []
    for g in GROUPS:
        v = float(np.mean([d[g] for d in imp_group]))
        print(f"    {g:<36}{v:>+8.4f}")
        rows_g.append(dict(group=g, permutation_drop=round(v, 5)))
    write_csv("outputs/exp_importance_groups.csv", rows_g)
    print("  (a feature that is correlated with another can look unimportant on its own: read groups and ablations together)")

    # correlations among the elevation features, from the first draw
    F0 = load_features(runs[0][1], args.extra_tag)
    ok = np.isfinite(F0["dem"]).all(axis=1)
    rho = spearmanr(F0["dem"][ok]).statistic
    write_csv("outputs/exp_importance_correlation.csv",
              [dict(feature=DEM_NAMES[i], **{DEM_NAMES[j]: round(float(rho[i, j]), 3) for j in range(n_feat)})
               for i in range(n_feat)])

    print("\n" + "=" * 100)
    print(f"ABLATIONS, held-out regions, within-site AUC pooled over {len(runs)} draws ({len(pool.sites())} sites)")
    print("=" * 100)
    ri, nr = pool.region_index()
    rows = []
    for name in pool.runs[0]:
        m, lo, hi, _ = cluster_boot(pool.per_site(name), ri, nr)
        rows.append(dict(model=name, within_site_auc=round(m, 4), lo=round(lo, 4), hi=round(hi, 4)))
    for r in sorted(rows, key=lambda r: -r["within_site_auc"]):
        print(f"  {r['model']:<62}{fmt_ci(r['within_site_auc'], r['lo'], r['hi'])}")
    write_csv("outputs/exp_importance_ablations.csv", rows)

    comps = [(f"{n} minus all 15 features", n, "All 15 elevation features")
             for n in pool.runs[0] if n.startswith("Leave out")]
    comps += [("All 15 features minus chance (0.5)", "All 15 elevation features", None)]
    if args.with_colour:
        comps += [("Colour + all elevation minus colour alone", "Colour + all 15 elevation features", "Colour alone (1 date)")]
        comps += [(f"Colour + {g} minus colour alone", f"Colour + {g}", "Colour alone (1 date)") for g in GROUPS]
    write_csv("outputs/exp_importance_comparisons.csv", report_comparisons(pool, comps, "Paired comparisons"))


if __name__ == "__main__":
    main()
