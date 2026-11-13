# pipeline/detect_crossval.py
#
# PURPOSE:
#   Site detection (site vs. background) with landscape controlled by design
#   and whole regions held out. Compares three methods on IDENTICAL data and
#   folds:
#     1. Random forest on spectral statistics (Orengo-style baseline)
#     2. Frozen Prithvi features + linear probe
#     3. Prithvi + LoRA, fine-tuned on this task
#
# DATA:
#   Every labeled site's patch (positive) + N_BG background patches cut from
#   the SAME tile, at least 2.5 km from any known site (negatives). A site and
#   its backgrounds always stay in the same fold.
#
# FOLDS:
#   The regional groups from crossval.py (REGION_KM, currently 300 km) are
#   split into N_FOLDS folds; each fold's regions are never seen in training.
#
# OPTIONS (all off by default, so the default run equals the earlier one):
#   --fracs 0.25,0.5,0.75,1.0   LEARNING CURVE: train on that share of the
#                               training sites in every fold (nested subsets,
#                               a site always with its backgrounds). The test
#                               fold is always complete, so the rows compare.
#   --augment                   random 90-degree rotations + flips (LoRA only)
#   --early-stop                hold out 15% of the training sites, score them
#                               after each epoch, keep the best epoch, stop
#                               after PATIENCE epochs without improvement
#   --epochs N                  maximum LoRA epochs (default 8)
#   --wd X                      weight decay (default 0 = plain Adam)
#   --tag NAME                  suffix for the output files
#
# METRICS (chance = 0.50 for both):
#   Within-site AUC: how often a site scores higher than its own backgrounds
#                    (landscape-proof: they share a tile)
#   Overall AUC:     ranking of all sites above all backgrounds
#   95% intervals by bootstrap over sites.
#   Paired test: per-site difference between two methods, bootstrapped.
#   This answers "is method A really better than B?" on the same sites.
#
# OUTPUT:
#   outputs/detect_predictions<tag>[_f<pct>].csv  every patch's scores,
#                                                 written after each fold
#   outputs/learning_curve<tag>.csv               one row per method x fraction
#
# USAGE (from the GeoArcheo folder):
#   python -u -m pipeline.detect_crossval --skip-lora      # baselines only
#   python -u -m pipeline.detect_crossval                  # same as earlier run
#   python -u -m pipeline.detect_crossval --fracs 0.25,0.5,0.75,1.0 \
#       --augment --early-stop --epochs 15 --wd 0.01 --tag _v2

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import pipeline.background_check as bc
from models.adapter import LoRAAdapter
from models.foundation import PrithviBackbone
from pipeline.crossval import overlap_groups
from pipeline.evaluate import load_site_coords
from pipeline.preprocess import find_tile_containing_point
from pipeline.train import ProcessedDataset
from utils.config import load_config

cfg = load_config()
N_BG = 3            # background patches per site
N_FOLDS = 5         # region-held-out folds
BATCH = 16
LR = 1e-4
GRID = 14           # 224 px / 16 px tokens
VAL_FRAC = 0.15     # share of training sites used for early stopping
PATIENCE = 3        # epochs without improvement before stopping
OUT_DIR = "outputs"
bc.rng = np.random.default_rng(123)     # fixed background sample for this run

NAMES = {"random_forest": "Random forest (spectral, baseline)",
         "frozen_probe": "Frozen Prithvi + linear probe",
         "lora_finetuned": "Prithvi + LoRA fine-tuned"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def central_pool(tokens, k=2):
    lo = GRID // 2 - k // 2
    idx = [1 + r * GRID + c for r in range(lo, lo + k) for c in range(lo, lo + k)]
    return tokens[:, idx, :].mean(dim=1)


def spectral_features(img):
    """Per-band statistics of the central 320 m and of the whole patch,
    plus the centre-minus-surroundings contrast."""
    x = img.numpy()
    c = x[:, 112 - 16:112 + 16, 112 - 16:112 + 16]
    feats = []
    for region in (c, x):
        r = region.reshape(x.shape[0], -1)
        feats += [r.mean(1), r.std(1), np.percentile(r, 10, axis=1), np.percentile(r, 90, axis=1)]
    feats.append(c.reshape(x.shape[0], -1).mean(1) - x.reshape(x.shape[0], -1).mean(1))
    return np.concatenate(feats)


def build_dataset():
    full = ProcessedDataset()
    ids = [torch.load(f).get('site_id') for f in full.files]
    coords = load_site_coords()
    groups = overlap_groups(ids, coords)
    all_lonlat = list(coords.values())

    images, y, grp, site = [], [], [], []
    skipped = 0
    for i, sid in enumerate(ids):
        lon, lat = coords[sid]
        tile = find_tile_containing_point(lon, lat, cfg['paths']['raw_data'])
        bgs = [bc.background_patch(tile, all_lonlat) for _ in range(N_BG)] if tile else []
        bgs = [b for b in bgs if b is not None]
        if not bgs:
            skipped += 1
            continue
        images.append(full[i][0]); y.append(1); grp.append(groups[i]); site.append(i)
        for b in bgs:
            images.append(b); y.append(0); grp.append(groups[i]); site.append(i)
    print(f"\n{sum(y)} sites, {len(y) - sum(y)} background patches "
          f"({len(np.unique(grp))} regions; skipped {skipped} sites with no clear background)", flush=True)
    return images, np.array(y), np.array(grp), np.array(site), ids


def frozen_embeddings(images, device):
    backbone = PrithviBackbone().to(device)
    backbone.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(images), BATCH):
            t = backbone.encoder(torch.stack(images[s:s + BATCH]).to(device))[-1]
            out.append(central_pool(t).cpu())
    del backbone
    return torch.cat(out).numpy()


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = LoRAAdapter(PrithviBackbone())        # pooling from config (center)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(cfg['model']['embedding_dim'], 1))

    def forward(self, x):
        return self.head(self.adapter(x)).squeeze(1)


def augment(x):
    """Random 90-degree rotation + horizontal flip per patch. The site stays
    at the centre, so the centre-token pooling still looks at it."""
    out = []
    for xi in x:
        xi = torch.rot90(xi, int(torch.randint(0, 4, (1,))), dims=(1, 2))
        if torch.rand(1).item() < 0.5:
            xi = torch.flip(xi, dims=(2,))
        out.append(xi)
    return torch.stack(out)


def predict(model, images, idx, device):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            out.append(torch.sigmoid(model(torch.stack([images[i] for i in b]).to(device))).float().cpu())
    return torch.cat(out).numpy()


def train_lora(images, y, site, tr, te, fold, device, args):
    torch.manual_seed(100 + fold)
    if args.early_stop:
        tr_sites = np.unique(site[tr])
        n_val = max(2, int(round(VAL_FRAC * len(tr_sites))))
        val_sites = np.random.default_rng(1000 + fold).choice(tr_sites, n_val, replace=False)
        is_val = np.isin(site[tr], val_sites)
        fit, val = tr[~is_val], tr[is_val]
    else:
        fit, val = tr, None

    model = Detector().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=LR, weight_decay=args.wd)   # wd=0 -> same as Adam
    pos_weight = torch.tensor([(y[fit] == 0).sum() / max(1, (y[fit] == 1).sum())],
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(fold * 100 + epoch).permutation(fit)
        total = 0.0
        for s in range(0, len(order), BATCH):
            b = order[s:s + BATCH]
            x = torch.stack([images[i] for i in b]).to(device)
            if args.augment:
                x = augment(x)
            t = torch.tensor(y[b], dtype=torch.float32, device=device)
            loss = loss_fn(model(x), t)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(b)
        msg = f"    LoRA epoch {epoch + 1}/{args.epochs}  loss {total / len(order):.4f}"

        if val is not None:
            v = predict(model, images, val, device)
            v_auc = within_site_auc(v, y[val], site[val], np.unique(site[val])).mean()
            msg += f"  val within-site AUC {v_auc:.3f}"
            if v_auc > best_auc:
                best_auc, best_epoch, bad = v_auc, epoch + 1, 0
                best_state = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
            else:
                bad += 1
        print(msg, flush=True)
        if val is not None and bad >= PATIENCE:
            print(f"    early stop: no improvement for {PATIENCE} epochs", flush=True)
            break

    if best_state is not None:
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in best_state:
                    p.copy_(best_state[k])
        print(f"    using weights from epoch {best_epoch} (val within-site AUC {best_auc:.3f})", flush=True)

    scores = predict(model, images, te, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def subsample(tr, site, frac, fold):
    """Keep `frac` of the training sites (with their backgrounds). Nested:
    the 25% sites are inside the 50% sites, and so on."""
    if frac >= 1.0:
        return tr
    sites = np.random.default_rng(500 + fold).permutation(np.unique(site[tr]))
    keep = sites[:max(5, int(round(frac * len(sites))))]
    return tr[np.isin(site[tr], keep)]


def within_site_auc(score, y, site, sites):
    vals = []
    for s in sites:
        m = site == s
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        vals.append(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
    return np.array(vals)


def summarize(name, score, y, site):
    sites = np.unique(site)
    per_site = within_site_auc(score, y, site, sites)
    auc = roc_auc_score(y, score)
    rows_of = {s: np.where(site == s)[0] for s in sites}
    rng = np.random.default_rng(0)
    boot_w, boot_a = [], []
    for _ in range(1000):
        pick = rng.choice(len(sites), len(sites), replace=True)
        boot_w.append(per_site[pick].mean())
        rows = np.concatenate([rows_of[sites[j]] for j in pick])
        if len(np.unique(y[rows])) == 2:
            boot_a.append(roc_auc_score(y[rows], score[rows]))
    wl, wh = np.percentile(boot_w, [2.5, 97.5])
    al, ah = np.percentile(boot_a, [2.5, 97.5])
    print(f"  {name:<36}{per_site.mean():>8.3f} [{wl:.2f}-{wh:.2f}]{auc:>12.3f} [{al:.2f}-{ah:.2f}]")
    return dict(within=per_site.mean(), within_lo=wl, within_hi=wh,
                overall=auc, overall_lo=al, overall_hi=ah), per_site


def paired(name, a, b):
    """Per-site within-site AUC difference a - b, bootstrapped over sites."""
    d = a - b
    rng = np.random.default_rng(1)
    boots = np.array([d[rng.choice(len(d), len(d), replace=True)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {name:<46}{d.mean():+.3f}  [{lo:+.3f} to {hi:+.3f}]   "
          f"P(diff <= 0) = {np.mean(boots <= 0):.3f}")


def save_predictions(path, ids, y, grp, site, fold_of, scores):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_id", "patch", "region_group", "fold"] + list(scores))
        for i in range(len(y)):
            w.writerow([ids[site[i]], "site" if y[i] == 1 else "background", int(grp[i]), int(fold_of[i])]
                       + [f"{scores[k][i]:.5f}" if fold_of[i] >= 0 else "" for k in scores])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lora", action="store_true", help="baselines only (fast)")
    ap.add_argument("--fracs", default="1.0", help="training-site shares, e.g. 0.25,0.5,0.75,1.0")
    ap.add_argument("--augment", action="store_true", help="rotations + flips during LoRA training")
    ap.add_argument("--early-stop", action="store_true", help="validation-based early stopping")
    ap.add_argument("--epochs", type=int, default=8, help="maximum LoRA epochs")
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _v2")
    args = ap.parse_args()
    fracs = sorted(float(f) for f in args.fracs.split(","))

    device = pick_device()
    print(f"Device: {device}", flush=True)
    print(f"Settings: fracs={fracs} augment={args.augment} early_stop={args.early_stop} "
          f"epochs={args.epochs} wd={args.wd} tag='{args.tag}'", flush=True)
    images, y, grp, site, ids = build_dataset()

    print("Spectral features...", flush=True)
    X_rf = np.stack([spectral_features(im) for im in images])
    print("Frozen Prithvi features...", flush=True)
    X_fm = frozen_embeddings(images, device)

    methods = ["random_forest", "frozen_probe"] + ([] if args.skip_lora else ["lora_finetuned"])
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X_rf, y, grp))
    curve_rows = []

    for frac in fracs:
        scores = {m: np.zeros(len(y)) for m in methods}
        fold_of = -np.ones(len(y), dtype=int)
        n_train_sites = []
        suffix = "" if frac >= 1.0 else f"_f{int(round(frac * 100))}"
        pred_path = os.path.join(OUT_DIR, f"detect_predictions{args.tag}{suffix}.csv")
        print(f"\n######## Training on {frac:.0%} of the training sites ########", flush=True)

        for fold, (tr_full, te) in enumerate(folds):
            tr = subsample(tr_full, site, frac, fold)
            n_train_sites.append(len(np.unique(site[tr])))
            t0 = time.time()
            print(f"\n=== Fold {fold + 1}/{N_FOLDS}: train {len(tr)} patches ({n_train_sites[-1]} sites), "
                  f"test {len(te)} patches from {len(np.unique(grp[te]))} unseen regions ===", flush=True)
            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                        random_state=fold, n_jobs=-1).fit(X_rf[tr], y[tr])
            scores["random_forest"][te] = rf.predict_proba(X_rf[te])[:, 1]

            scaler = StandardScaler().fit(X_fm[tr])
            probe = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(
                scaler.transform(X_fm[tr]), y[tr])
            scores["frozen_probe"][te] = probe.predict_proba(scaler.transform(X_fm[te]))[:, 1]

            if not args.skip_lora:
                scores["lora_finetuned"][te] = train_lora(images, y, site, tr, te, fold, device, args)

            fold_of[te] = fold
            save_predictions(pred_path, ids, y, grp, site, fold_of, scores)
            print(f"  fold done in {(time.time() - t0) / 60:.1f} min; predictions saved -> {pred_path}", flush=True)

        print("\n" + "=" * 86)
        print(f"SITE DETECTION, {frac:.0%} of training sites (~{np.mean(n_train_sites):.0f} per fold), "
              f"{N_FOLDS} region-held-out folds")
        print(f"({int(y.sum())} test sites, {int((y == 0).sum())} backgrounds from the same tiles)")
        print("=" * 86)
        print(f"  {'Method':<36}{'Within-site AUC':>21}{'Overall AUC':>23}")
        per_site = {}
        for m in methods:
            res, per_site[m] = summarize(NAMES[m], scores[m], y, site)
            curve_rows.append(dict(frac=frac, train_sites=round(float(np.mean(n_train_sites)), 1),
                                   method=m, **{k: round(float(v), 4) for k, v in res.items()}))
        print("\n  Chance = 0.500 for both columns. Brackets: 95% bootstrap interval over sites.")
        print(f"\n  Paired comparison (same sites, within-site AUC):")
        paired("Frozen Prithvi minus random forest", per_site["frozen_probe"], per_site["random_forest"])
        if "lora_finetuned" in methods:
            paired("LoRA minus random forest", per_site["lora_finetuned"], per_site["random_forest"])
            paired("LoRA minus frozen Prithvi", per_site["lora_finetuned"], per_site["frozen_probe"])
        print("  A difference is convincing when its whole interval is above 0 "
              "(P(diff <= 0) below 0.025).")

    curve_path = os.path.join(OUT_DIR, f"learning_curve{args.tag}.csv")
    with open(curve_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        w.writeheader()
        w.writerows(curve_rows)

    if len(fracs) > 1:
        print("\n" + "=" * 86)
        print("LEARNING CURVE: within-site AUC by amount of training data")
        print("=" * 86)
        print(f"  {'Training sites':<18}" + "".join(f"{m.replace('_', ' '):>22}" for m in methods))
        for frac in fracs:
            rows = {r["method"]: r for r in curve_rows if r["frac"] == frac}
            print(f"  {rows[methods[0]]['train_sites']:<18.0f}"
                  + "".join(f"{rows[m]['within']:>13.3f} [{rows[m]['within_lo']:.2f}-{rows[m]['within_hi']:.2f}]"
                            for m in methods))
        print("\n  Still rising at the last row -> more sites should help.")
        print("  Flat -> more sites alone will not; focus on other changes.")
    print(f"\nLearning-curve table saved -> {curve_path}")


if __name__ == "__main__":
    main()
