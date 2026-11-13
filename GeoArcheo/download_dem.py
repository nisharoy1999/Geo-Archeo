# download_dem.py
#
# PURPOSE:
#   Elevation for every tile in data/raw/, from the free Copernicus 30 m
#   elevation model (COPERNICUS_30 on Copernicus Data Space), resampled onto
#   the tile's own 10 m map grid:
#       data/raw_dem/<same file name>.tif
#
#   Why: many settlements in this region are tells, mounds built up over
#   centuries, often metres to tens of metres high. They stand out in
#   elevation even where the colour of the ground does not (Menze & Ur 2012
#   mapped thousands of Mesopotamian mounds largely this way).
#   Note: this model measures the surface, including buildings and trees.
#
# RE-RUNNABLE: finished files are skipped, failures retried in later passes.
#
# USAGE (from the GeoArcheo folder, on the Mac):
#   python download_dem.py --max-tiles 2     # quick test
#   caffeinate -dims python -u download_dem.py

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openeo
import rasterio
from rasterio.warp import transform_bounds

from utils.config import load_config

cfg = load_config()
RAW = Path(cfg['paths']['raw_data'])
OUT = Path("data/raw_dem")
COLLECTION = "COPERNICUS_30"
PASSES = 6
WAIT_AFTER_FAIL_S = 60
login_lock = threading.Lock()


def tile_info(path):
    with rasterio.open(path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return {"west": w, "south": s, "east": e, "north": n}, src.crs.to_epsg()


def is_valid(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            a = src.read(1)
        return bool(np.isfinite(a).any())
    except Exception:
        return False


def download(conn, bbox, epsg, out_path):
    part = out_path.with_suffix(".part.tif")
    cube = conn.load_collection(COLLECTION, spatial_extent=bbox, temporal_extent=["2010-01-01", "2030-12-31"])
    cube = cube.max_time()
    cube = cube.resample_spatial(resolution=10, projection=epsg, method="bilinear")
    cube.download(str(part), format="GTiff")
    if not is_valid(part):
        part.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is unreadable or empty")
    part.rename(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    tiles = sorted(RAW.glob("*.tif"))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(tiles)} tiles, elevation from {COLLECTION}")

    local = threading.local()

    def connection():
        if not hasattr(local, "conn"):
            with login_lock:
                local.conn = openeo.connect("openeo.dataspace.copernicus.eu")
                local.conn.authenticate_oidc()
        return local.conn

    def work(item):
        k, n, t = item
        try:
            t0 = time.time()
            bbox, epsg = tile_info(t)
            download(connection(), bbox, epsg, OUT / t.name)
            print(f"[{k}/{n}] saved  {t.name}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{k}/{n}] failed {t.name}: {str(e)[:160]}  (retried next pass)", flush=True)
            time.sleep(WAIT_AFTER_FAIL_S)

    for p in range(1, PASSES + 1):
        todo = [t for t in tiles if not is_valid(OUT / t.name)]
        print(f"\n=== Pass {p}/{PASSES}: {len(todo)} to download, {len(tiles) - len(todo)} done ===", flush=True)
        if not todo:
            break
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, [(k, len(todo), t) for k, t in enumerate(todo, 1)]))

    missing = [t.name for t in tiles if not is_valid(OUT / t.name)]
    print(f"\nDone: {len(tiles) - len(missing)}/{len(tiles)} elevation files ready.")
    for m in missing:
        print(f"  missing: {m}")


if __name__ == "__main__":
    main()
