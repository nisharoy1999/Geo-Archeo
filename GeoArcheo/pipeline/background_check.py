# pipeline/background_check.py
#
# PURPOSE:
#   Site vs. background, with landscape controlled by design. For every
#   labeled site, cut a background patch from the SAME tile, at least
#   MIN_KM from any known site. Frozen Prithvi features, then:
#     - linear probe (logistic regression) trained on other regions only
#     - nearest-neighbour accuracy
#     - paired test: does each site score higher than its own background?
#   Whole regions (REGION_KM in crossval.py) are held out together.
#
# USAGE (from the GeoArcheo folder):
#   python -m pipeline.background_check

import math

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.foundation import PrithviBackbone
from pipeline.crossval import overlap_groups
from pipeline.evaluate import load_site_coords
from pipeline.preprocess import find_tile_containing_point
from pipeline.train import ProcessedDataset
from utils.config import load_config

cfg = load_config()
SIZE = cfg['image']['size']
HALF = SIZE // 2
MIN_KM = 2.5                     # background patch centre must be this far from every known site
PX_M = 10
MEAN = np.array(cfg['image']['mean'], dtype=np.float32)[:, None, None]
STD = np.array(cfg['image']['std'], dtype=np.float32)[:, None, None]
GRID = 14
rng = np.random.default_rng(7)


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def background_patch(tile_path, all_lonlat):
    """A standardized 224x224 patch from the same tile, far from all known sites."""
    with rasterio.open(tile_path) as src:
        lons = [p[0] for p in all_lonlat]
        lats = [p[1] for p in all_lonlat]
        xs, ys = warp_transform("EPSG:4326", src.crs, lons, lats)
        site_px = [src.index(x, y) for x, y in zip(xs, ys)]
        min_px = MIN_KM * 1000 / PX_M
        if src.height <= SIZE or src.width <= SIZE:
            return None
        for _ in range(500):
            r = int(rng.integers(HALF, src.height - HALF))
            c = int(rng.integers(HALF, src.width - HALF))
            if all(math.hypot(r - sr, c - sc) >= min_px for sr, sc in site_px):
                img = src.read(window=Window(c - HALF, r - HALF, SIZE, SIZE)).astype(np.float32)
                if img.shape[1:] != (SIZE, SIZE):
                    continue
                if np.mean(~np.isfinite(img) | (img == 0)) > 0.05:
                    continue
                return torch.tensor((img - MEAN) / STD)
    return None


def embed(backbone, images):
    out = {"A. Summary token": [], "D. Central ~320 m features": [], "E. Average features": []}
    lo = GRID // 2 - 1
    idx = [1 + r * GRID + c for r in range(lo, lo + 2) for c in range(lo, lo + 2)]
    for s in range(0, len(images), 8):
        x = torch.stack(images[s:s + 8])
        with torch.no_grad():
            t = backbone.encoder(x)[-1]
        out["A. Summary token"].append(t[:, 0, :])
        out["D. Central ~320 m features"].append(t[:, idx, :].mean(dim=1))
        out["E. Average features"].append(t[:, 1:, :].mean(dim=1))
        print(f"  embedded {min(s + 8, len(images))}/{len(images)}", flush=True)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    full = ProcessedDataset()
    ids = [torch.load(f).get('site_id') for f in full.files]
    coords = load_site_coords()
    groups = overlap_groups(ids, coords)

    images, y, g, pair = [], [], [], []
    all_lonlat = list(coords.values())
    skipped = 0
    for i, sid in enumerate(ids):
        lon, lat = coords[sid]
        tile = find_tile_containing_point(lon, lat, cfg['paths']['raw_data'])
        bg = background_patch(tile, all_lonlat) if tile else None
        if bg is None:
            skipped += 1
            continue
        images += [full[i][0], bg]
        y += [1, 0]
        g += [groups[i], groups[i]]
        pair += [i, i]
    y, g, pair = np.array(y), np.array(g), np.array(pair)
    n_sites = len(y) // 2
    print(f"\n{n_sites} sites, each with a background patch from the same tile "
          f"(skipped {skipped}: no clear background found)", flush=True)

    backbone = PrithviBackbone()
    backbone.eval()
    print("Embedding patches...", flush=True)
    feats = embed(backbone, images)

    print("\n" + "=" * 96)
    print(f"SITE vs BACKGROUND: frozen backbone, {len(np.unique(g))} regions held out one at a time")
    print("=" * 96)
    print(f"{'Variant':<30}{'Linear probe':>22}{'Nearest neighbour':>24}{'Paired: site > own bg':>26}")
    for name, X in feats.items():
        prob = np.zeros(len(y))
        nn_correct = 0
        U = X / np.linalg.norm(X, axis=1, keepdims=True)
        for region in np.unique(g):
            te, tr = g == region, g != region
            scaler = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000, C=0.1).fit(scaler.transform(X[tr]), y[tr])
            prob[te] = clf.predict_proba(scaler.transform(X[te]))[:, 1]
            sims = U[te] @ U[tr].T
            nn_correct += int(np.sum(y[tr][sims.argmax(axis=1)] == y[te]))
        lin = int(np.sum((prob >= 0.5) == (y == 1)))
        paired = int(sum(prob[(pair == p) & (y == 1)][0] > prob[(pair == p) & (y == 0)][0]
                         for p in np.unique(pair)))
        n = len(y)

        def cell(k, m):
            lo, hi = cp(k, m)
            return f"{k}/{m} = {k / m:.2f} [{lo:.2f}-{hi:.2f}]"

        print(f"{name:<30}{cell(lin, n):>22}{cell(nn_correct, n):>24}{cell(paired, n_sites):>26}")
    print("\nBaseline: 50% for all three (balanced; paired test is a coin flip without signal)")


if __name__ == "__main__":
    main()