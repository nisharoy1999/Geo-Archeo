# download_fabdem.py
#
# PURPOSE:
#   Bare-earth elevation for every tile in data/raw/: FABDEM (Hawker et al.
#   2022, University of Bristol), the Copernicus 30 m model with buildings
#   and trees removed by machine learning. Saved on each Sentinel-2 tile's
#   own 10 m grid:
#       data/raw_fabdem/<same file name>.tif
#
#   Why: the Copernicus model measures the SURFACE, so a modern village on an
#   ancient mound also looks like a mound. If sites still stand out in
#   FABDEM, the signal is the ground itself, not modern buildings.
#
#   FABDEM licence: CC BY-NC-SA 4.0 (research use is fine; cite Hawker et al.
#   2022, Environ. Res. Lett. 17 024016).
#
# Uses the `fabdem` package, which fetches only the 1-degree tiles needed
# (byte ranges from the Bristol data archive) and caches them.
#   pip install fabdem
#
# USAGE (from the GeoArcheo folder, on the Mac):
#   python download_fabdem.py --max-tiles 2      # quick test
#   caffeinate -dims python -u download_fabdem.py

import argparse
import time
from pathlib import Path

import fabdem
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds

from utils.config import load_config

cfg = load_config()
RAW = Path(cfg['paths']['raw_data'])
OUT = Path("data/raw_fabdem")
CACHE = Path("data/fabdem_cache")
NODATA = -9999.0
PAD_DEG = 0.01


def is_valid(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            a = src.read(1)
        return bool(np.isfinite(a).any() and (a != NODATA).any())
    except Exception:
        return False


def one_tile(tile):
    with rasterio.open(tile) as src:
        crs, transform, width, height = src.crs, src.transform, src.width, src.height
        w, s, e, n = transform_bounds(crs, "EPSG:4326", *src.bounds)
    tmp = CACHE / f"tmp_{tile.stem}.tif"
    fabdem.download((w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG), output_path=str(tmp),
                    show_progress=False, cache=CACHE)
    dst = np.full((height, width), NODATA, dtype=np.float32)
    with rasterio.open(tmp) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=NODATA,
                  resampling=Resampling.bilinear)
    part = OUT / (tile.stem + ".part.tif")
    with rasterio.open(part, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=NODATA, compress="deflate") as out:
        out.write(dst, 1)
    part.rename(OUT / tile.name)
    tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tiles", type=int, default=0)
    args = ap.parse_args()
    tiles = sorted(RAW.glob("*.tif"))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    for p in range(1, 4):
        todo = [t for t in tiles if not is_valid(OUT / t.name)]
        print(f"\n=== Pass {p}/3: {len(todo)} to do, {len(tiles) - len(todo)} done ===", flush=True)
        if not todo:
            break
        for k, t in enumerate(todo, 1):
            t0 = time.time()
            try:
                one_tile(t)
                print(f"[{k}/{len(todo)}] saved  {t.name}  ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[{k}/{len(todo)}] failed {t.name}: {str(e)[:200]}", flush=True)
                time.sleep(10)

    missing = [t.name for t in tiles if not is_valid(OUT / t.name)]
    print(f"\nDone: {len(tiles) - len(missing)}/{len(tiles)} FABDEM files ready.")
    for m in missing:
        print(f"  missing: {m}")


if __name__ == "__main__":
    main()
