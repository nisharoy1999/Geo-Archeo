# build_season_patches.py
#
# PURPOSE:
#   Cut every labeled site and its background patches from all FOUR seasons
#   and pack them into one file for the GPU machine:
#       data/season_patches.npz
#
#   Each sample is (6 bands, 4 seasons, 224, 224), the layout Prithvi-EO-2.0
#   expects for multi-date input. Seasons: Jan-Feb, Apr-Jun, Jul-Sep, Oct-Dec.
#
# FAIRNESS RULES (same for sites and backgrounds, so nothing leaks):
#   - a location is used only if its full 224x224 window lies inside the tile
#     in ALL four seasons and no season has more than 5% empty/no-data pixels
#   - backgrounds come from the same tile, at least 2.5 km from every known
#     site, exactly as in background_check.py
#   - a site with no usable background is dropped (with its backgrounds)
#
# USAGE (from the GeoArcheo folder, on the Mac, after download_seasons.py):
#   python build_season_patches.py
#   python build_season_patches.py --seed 7 --out data/season_patches_s7.npz   # robustness check
#   python build_season_patches.py --seed 7 --out data/season_patches_s7.npz --extra-only
#       re-creates exactly the same samples and saves only elevation + reference
#       features to data/season_patches_s7_extra.npz (needs data/raw_dem/)
#   ... --extra-only --dem-dir data/raw_fabdem --extra-tag _fabdem
#       same, with the bare-earth FABDEM elevation -> ..._extra_fabdem.npz
#
# Then copy the result to the GPU machine:
#   scp data/season_patches.npz archeo:~/GeoArcheo/data/

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from pipeline.crossval import overlap_groups
from pipeline.features import dem_features, season_features, standardize
from pipeline.evaluate import load_site_coords
from pipeline.preprocess import find_tile_containing_point
from utils.config import load_config

cfg = load_config()
RAW = Path(cfg['paths']['raw_data'])
SEASON_DIRS = [("Jan-Feb", RAW),
               ("Apr-Jun", Path("data/raw_seasons/q2")),
               ("Jul-Sep", Path("data/raw_seasons/q3")),
               ("Oct-Dec", Path("data/raw_seasons/q4"))]
SIZE = cfg['image']['size']
HALF = SIZE // 2
N_BG = 3
MIN_KM = 2.5
PX_M = 10
MAX_EMPTY = 0.05
TRIES = 500
DEM_DIR = Path("data/raw_dem")
K_REF = 25           # unlabeled reference patches per tile (for per-image score calibration)
rng = None

_open = {}


def ds(path):
    """Keep tiles open while we work (much faster than reopening)."""
    if path not in _open:
        _open[path] = rasterio.open(path)
    return _open[path]


def read_all_seasons(tile_name, x, y, crs):
    """Window centred on map point (x, y) [in `crs`] from every season,
    or None if any season is missing, cut off, or mostly empty."""
    frames = []
    for _, d in SEASON_DIRS:
        p = d / tile_name
        if not p.exists():
            return None
        src = ds(p)
        px, py = (x, y)
        if src.crs != crs:
            xs, ys = warp_transform(crs, src.crs, [x], [y])
            px, py = xs[0], ys[0]
        r, c = src.index(px, py)
        if r - HALF < 0 or c - HALF < 0 or r + HALF > src.height or c + HALF > src.width:
            return None
        img = src.read(window=Window(c - HALF, r - HALF, SIZE, SIZE)).astype(np.float32)
        if img.shape != (len(cfg['image']['s2_bands']), SIZE, SIZE):
            return None
        if np.mean(~np.isfinite(img) | (img == 0)) > MAX_EMPTY:
            return None
        frames.append(np.clip(np.nan_to_num(img), 0, 32767).astype(np.int16))
    return np.stack(frames, axis=1)            # (bands, seasons, H, W)


def read_dem(tile_name, x, y, crs):
    """Elevation window (metres, float32, NaN = missing) centred on (x, y), or None."""
    p = DEM_DIR / tile_name
    if not p.exists():
        return None
    src = ds(p)
    px, py = (x, y)
    if src.crs != crs:
        xs, ys = warp_transform(crs, src.crs, [x], [y])
        px, py = xs[0], ys[0]
    r, c = src.index(px, py)
    if r - HALF < 0 or c - HALF < 0 or r + HALF > src.height or c + HALF > src.width:
        return None
    d = src.read(1, window=Window(c - HALF, r - HALF, SIZE, SIZE)).astype(np.float32)
    if src.nodata is not None:
        d[d == src.nodata] = np.nan
    return d if d.shape == (SIZE, SIZE) else None


def reference_features(tiles, seed):
    """K_REF random, unlabeled windows per tile, described with the same
    features as the samples. In real use you would have exactly this: random
    ground from the same image, without knowing where sites are."""
    rr = np.random.default_rng(seed + 1000)          # separate stream: samples stay identical
    rf1, rf4, dem, tname = [], [], [], []
    for t in tiles:
        ref = ds(RAW / t)
        got = 0
        for _ in range(K_REF * 8):
            if got == K_REF or ref.height <= SIZE or ref.width <= SIZE:
                break
            r = int(rr.integers(HALF, ref.height - HALF))
            c = int(rr.integers(HALF, ref.width - HALF))
            x, y = ref.xy(r, c)
            img = read_all_seasons(t, x, y, ref.crs)
            if img is None:
                continue
            f1, f4 = season_features(standardize(img, cfg['image']['mean'], cfg['image']['std']))
            rf1.append(f1); rf4.append(f4)
            dem.append(dem_features(read_dem(t, x, y, ref.crs)))
            tname.append(t)
            got += 1
    return np.array(rf1), np.array(rf4), np.array(dem), np.array(tname)


def main():
    global rng, DEM_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123, help="background sampling seed")
    ap.add_argument("--out", default="data/season_patches.npz")
    ap.add_argument("--extra-only", action="store_true",
                    help="re-create the same samples but only save elevation + reference features "
                         "(<out>_extra.npz); the big image file is left untouched")
    ap.add_argument("--dem-dir", default="data/raw_dem", help="elevation tiles to use")
    ap.add_argument("--extra-tag", default="", help="e.g. _fabdem -> <out>_extra_fabdem.npz")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    DEM_DIR = Path(args.dem_dir)
    OUT_FILE = Path(args.out)
    print(f"Background seed {args.seed} -> {OUT_FILE}; elevation from {DEM_DIR}")
    files = sorted(glob.glob(os.path.join(cfg['paths']['processed_data'], "*.pt")))
    ids = [torch.load(f).get('site_id') for f in files]
    coords = load_site_coords()
    all_lonlat = [coords[s] for s in coords]
    print(f"{len(ids)} labeled sites. Seasons: {[s for s, _ in SEASON_DIRS]}")

    X, y, site_of, kept_ids, cx, cy, tiles, check = [], [], [], [], [], [], [], []
    why = {"no tile": 0, "site window not clean in all seasons": 0, "no clean background": 0}

    for n, sid in enumerate(ids, 1):
        lon, lat = coords[sid]
        tile = find_tile_containing_point(lon, lat, str(RAW))
        if tile is None:
            why["no tile"] += 1
            continue
        ref = ds(tile)
        xs, ys = warp_transform("EPSG:4326", ref.crs, [lon], [lat])
        site_img = read_all_seasons(tile.name, xs[0], ys[0], ref.crs)
        if site_img is None:
            why["site window not clean in all seasons"] += 1
            continue

        # backgrounds: same tile, >= MIN_KM from every known site, clean in all seasons
        sx, sy = warp_transform("EPSG:4326", ref.crs, [p[0] for p in all_lonlat], [p[1] for p in all_lonlat])
        site_px = [ref.index(a, b) for a, b in zip(sx, sy)]
        min_px = MIN_KM * 1000 / PX_M
        bgs = []
        if ref.height > SIZE and ref.width > SIZE:
            for _ in range(TRIES):
                if len(bgs) == N_BG:
                    break
                r = int(rng.integers(HALF, ref.height - HALF))
                c = int(rng.integers(HALF, ref.width - HALF))
                if any(math.hypot(r - a, c - b) < min_px for a, b in site_px):
                    continue
                bx, by = ref.xy(r, c)
                img = read_all_seasons(tile.name, bx, by, ref.crs)
                if img is not None:
                    bgs.append((img, bx, by))
        if not bgs:
            why["no clean background"] += 1
            continue

        k = len(kept_ids)
        kept_ids.append(sid)
        for img, px, py, label in [(site_img, xs[0], ys[0], 1)] + [(b, bx, by, 0) for b, bx, by in bgs]:
            if not args.extra_only:
                X.append(img)
            y.append(label); site_of.append(k); cx.append(px); cy.append(py); tiles.append(tile.name)
            check.append(int(img.astype(np.int64).sum()))       # fingerprint: proves _extra matches the images
        if n % 20 == 0:
            print(f"  {n}/{len(ids)} sites processed, {len(kept_ids)} kept", flush=True)

    print("Elevation for every sample, and unlabeled reference patches per tile...", flush=True)
    dem = np.full((len(y), SIZE, SIZE), np.nan, dtype=np.float32)
    for i in range(len(y)):
        d = read_dem(tiles[i], cx[i], cy[i], ds(RAW / tiles[i]).crs)
        if d is not None:
            dem[i] = d
    n_dem = int(np.isfinite(dem).reshape(len(y), -1).any(axis=1).sum())
    ref_rf1, ref_rf4, ref_dem, ref_tile = reference_features(sorted(set(tiles)), args.seed)
    print(f"  elevation found for {n_dem}/{len(y)} samples; {len(ref_tile)} reference patches "
          f"from {len(set(ref_tile))} tiles", flush=True)

    for f in _open.values():
        f.close()

    groups = overlap_groups(kept_ids, coords)
    site_of = np.array(site_of)
    grp = np.array([groups[s] for s in site_of])
    print(f"\nKept {len(kept_ids)} sites with {int((np.array(y) == 0).sum())} backgrounds "
          f"({len(np.unique(grp))} regions).")
    for k, v in why.items():
        if v:
            print(f"  dropped {v} sites: {k}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    extra = OUT_FILE.with_name(OUT_FILE.stem + "_extra" + args.extra_tag + ".npz")
    np.savez_compressed(extra, y=np.array(y), site=site_of, ids=np.array(kept_ids),
                        cx=np.array(cx), cy=np.array(cy), tile=np.array(tiles), dem=dem.astype(np.float16),
                        ref_rf1=ref_rf1, ref_rf4=ref_rf4, ref_dem=ref_dem, ref_tile=ref_tile,
                        check=np.array(check, dtype=np.int64))
    print(f"Saved -> {extra} ({extra.stat().st_size / 1e6:.0f} MB)")
    if not args.extra_only:
        X = np.stack(X)
        np.savez_compressed(OUT_FILE, X=X, y=np.array(y), site=site_of, grp=grp,
                            ids=np.array(kept_ids), seasons=np.array([s for s, _ in SEASON_DIRS]))
        print(f"Saved -> {OUT_FILE} ({OUT_FILE.stat().st_size / 1e9:.2f} GB on disk)")


if __name__ == "__main__":
    main()
