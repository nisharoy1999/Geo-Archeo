# pipeline/crossval.py
#
# PURPOSE:
#   The project's single, leakage-free evaluation. Every labeled site is
#   held out exactly once (stratified, grouped k-fold) and predicted by a
#   model that never trained on it. Reports, over ALL sites:
#     1. supervised accuracy and per-class recall
#     2. zero-shot (prototype) accuracy
#     3. the full pipeline: symbolic gate with real SRTM + OpenStreetMap
#        inputs at every site
#     4. conformal coverage, calibrated on the other sites' held-out scores
#   All with exact 95% intervals and the majority-class baseline.
#
# USAGE:
#   python -m pipeline.crossval

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import beta

from models.foundation import PrithviBackbone
from models.adapter import LoRAAdapter, SiteClassifier
from models.symbolic import ScallopReasoner
from pipeline.train import ProcessedDataset
from pipeline.evaluate import load_site_coords, real_symbolic_inputs
from utils.config import load_config

cfg = load_config()
N_FOLDS = 6


def cp_interval(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def report(label, k, n):
    lo, hi = cp_interval(k, n)
    print(f"  {label:<44} {k}/{n} = {k / n:.3f}  (95% CI {lo:.2f}-{hi:.2f})")


REGION_KM = 300   # sites closer than this are always held out together


def overlap_groups(ids, coords):
    """Group sites that must be held out together: any two sites within
    REGION_KM of each other (which also covers overlapping patches), joined
    transitively. A test site is then never from the same neighbourhood as
    a training site."""
    from collections import Counter
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            (lo1, la1), (lo2, la2) = coords[ids[i]], coords[ids[j]]
            dx = abs(lo1 - lo2) * 111.32 * math.cos(math.radians((la1 + la2) / 2))
            dy = abs(la1 - la2) * 110.57
            if math.hypot(dx, dy) < REGION_KM:
                parent[find(i)] = find(j)
    groups = np.array([find(i) for i in range(len(ids))])
    sizes = Counter(groups.tolist())
    print(f"  {len(sizes)} regional groups (largest has {max(sizes.values())} sites)")
    return groups


def run_fold(full, train_idx, test_idx, fold, device):
    torch.manual_seed(42 + fold)
    adapter = LoRAAdapter(PrithviBackbone()).to(device)
    classifier = SiteClassifier().to(device)

    n_cls = cfg['sites']['num_classes']
    counts = torch.bincount(torch.tensor([int(full[i][1]) for i in train_idx]),
                            minlength=n_cls).float()
    weights = torch.zeros(n_cls)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    optimiser = torch.optim.Adam(list(adapter.parameters()) + list(classifier.parameters()),
                                 lr=cfg['training']['learning_rate'])
    loader = DataLoader(Subset(full, train_idx), batch_size=cfg['training']['batch_size'], shuffle=True)
    for _ in range(cfg['training']['epochs']):
        adapter.train()
        classifier.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(classifier(adapter(images)), labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    adapter.eval()
    classifier.eval()

    def embed(idxs):
        x = torch.stack([full[i][0] for i in idxs]).to(device)
        with torch.no_grad():
            return adapter(x), [int(full[i][1]) for i in idxs]

    train_emb, train_y = embed(train_idx)
    test_emb, test_y = embed(test_idx)
    with torch.no_grad():
        probs = torch.softmax(classifier(test_emb), dim=1).cpu()

    classes = sorted(set(train_y))
    protos = torch.stack([train_emb[[i for i, y in enumerate(train_y) if y == c]].mean(dim=0)
                          for c in classes])
    sims = F.cosine_similarity(test_emb.unsqueeze(1), protos.unsqueeze(0), dim=2).cpu()
    zs_pred = [classes[j] for j in sims.argmax(dim=1).tolist()]
    zs_score = sims.max(dim=1).values.tolist()
    return test_y, probs, zs_pred, zs_score


def conformal_loo(probs, true, alpha):
    """Each site is checked against a threshold calibrated on the OTHER sites'
    held-out scores, so no site calibrates its own prediction set."""
    scores = np.array([1.0 - probs[i][t] for i, t in enumerate(true)])
    n = len(scores)
    covered, sizes = 0, []
    for i in range(n):
        cal = np.delete(scores, i)
        level = min(1.0, math.ceil((len(cal) + 1) * (1 - alpha)) / len(cal))
        q = np.quantile(cal, level, method="higher")
        pred_set = [c for c in range(len(probs[i])) if 1.0 - probs[i][c] <= q]
        sizes.append(len(pred_set))
        covered += int(true[i] in pred_set)
    return covered, n, float(np.mean(sizes))


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    names = cfg['sites']['classes']

    full = ProcessedDataset()
    ids = [torch.load(f).get('site_id') for f in full.files]
    labels = np.array([int(full[i][1]) for i in range(len(full))])
    coords = load_site_coords()
    groups = overlap_groups(ids, coords)

    n = len(labels)
    true = [0] * n
    sup = [0] * n
    zs = [0] * n
    zs_score = [0.0] * n
    probs = [None] * n

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), labels, groups)):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS}: train {len(tr)}, test {len(te)} ===")
        y, p, z, zsc = run_fold(full, tr.tolist(), te.tolist(), fold, device)
        for k, i in enumerate(te):
            true[i], probs[i] = y[k], p[k].numpy()
            sup[i] = int(p[k].argmax())
            zs[i], zs_score[i] = z[k], zsc[k]

    # ── full pipeline: symbolic gate at every site, real inputs ──────────
    print("\nFull pipeline at every site (real SRTM + OpenStreetMap inputs):")
    reasoner = ScallopReasoner()
    accepted = accepted_right = 0
    rule_fail = {}
    low_score = no_tile = 0
    for i in range(n):
        inputs = real_symbolic_inputs(ids[i], coords)
        if inputs is None:
            no_tile += 1
            print(f"  {ids[i]}: no tile covers this site, skipped")
            continue
        water, elev, slope = inputs
        water_note = "unknown" if water is None else f"{water:.0f}m"
        decision, rules, _ = reasoner.decide(zs_score[i], 0.0 if water is None else water, elev, slope)
        failed = [r for r, ok in rules.items() if not ok]
        if decision != "REJECTED":
            accepted += 1
            accepted_right += int(zs[i] == true[i])
        elif failed:
            for r in failed:
                rule_fail[r] = rule_fail.get(r, 0) + 1
        else:
            low_score += 1
        print(f"  {ids[i]}: true={names[true[i]]}, matched={names[zs[i]]} ({zs_score[i]:.3f}) "
              f"-> {decision} {'[' + ', '.join(failed) + ']' if failed else ''} "
              f"(water={water_note}, elev={elev:.0f}m, slope={slope:.1f}deg)")

    # ── summary ──────────────────────────────────────────────────────────
    k_sup = sum(a == b for a, b in zip(true, sup))
    k_zs = sum(a == b for a, b in zip(true, zs))
    majority = int(np.bincount(labels).max())
    alpha = cfg['conformal']['alpha']
    cov, n_cov, avg_size = conformal_loo(probs, true, alpha)

    print("\n" + "=" * 70)
    print(f"RESULTS OVER ALL {n} SITES ({N_FOLDS}-fold grouped cross-validation)")
    print("=" * 70)
    report("Supervised classifier", k_sup, n)
    report("Zero-shot prototypes", k_zs, n)
    report("Baseline: always majority class", majority, n)
    report(f"Conformal coverage (target {1 - alpha:.0%})", cov, n_cov)
    print(f"  {'Conformal mean prediction-set size':<44} {avg_size:.2f}")
    checked = n - no_tile
    report("Full pipeline: verified sites accepted", accepted, checked)
    if accepted:
        report("  ...of which given the correct class", accepted_right, accepted)
    print(f"  Rejections by rule: {rule_fail or 'none'}; by low score: {low_score}")

    print("\n  Per-class recall (supervised / zero-shot):")
    for c in sorted(set(true)):
        idx = [i for i in range(n) if true[i] == c]
        print(f"    {names[c]:<20} {sum(sup[i] == c for i in idx)}/{len(idx)}  /  "
              f"{sum(zs[i] == c for i in idx)}/{len(idx)}")


if __name__ == "__main__":
    main()