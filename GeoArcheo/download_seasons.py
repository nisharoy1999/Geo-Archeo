# download_seasons.py
#
# PURPOSE:
#   For every tile you already have in data/raw/ (Jan-Feb 2024 composites),
#   download the SAME area for three more seasons of 2024, so each site can be
#   seen across the year. Buried walls and mounds often show only in some
#   seasons (crop marks, soil moisture), which a single date misses.
#
#     data/raw/                    Jan-Feb   (already there, untouched)
#     data/raw_seasons/q2/         Apr-Jun
#     data/raw_seasons/q3/         Jul-Sep
#     data/raw_seasons/q4/         Oct-Dec
#
#   Same file names as data/raw/, same bands, same median compositing as
#   download_sentinel2.py, so the four seasons are directly comparable.
#
# RE-RUNNABLE:
#   Finished, readable files are skipped. Failed downloads (Copernicus rate
#   limits, 429/502) are retried in later passes. Just run it again if it stops.
#
# USAGE (from the GeoArcheo folder, on the Mac):
#   python download_seasons.py --max-tiles 2     # quick test first
#   caffeinate -dims python -u download_seasons.py              # 3 at a time
#   caffeinate -dims python -u download_seasons.py --workers 2  # if many fail with 429
#
# A browser window opens once to log in to Copernicus (same as before).

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
OUT = Path("data/raw_seasons")
SEASONS = {                      # end date is exclusive
    "q2": ("2024-04-01", "2024-07-01"),
    "q3": ("2024-07-01", "2024-10-01"),
    "q4": ("2024-10-01", "2025-01-01"),
}
PASSES = 6
WAIT_AFTER_FAIL_S = 60
login_lock = threading.Lock()


def tile_bbox(path):
    """The tile's area in lon/lat (slightly larger than the tile, never smaller)."""
    with rasterio.open(path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return {"west": w, "south": s, "east": e, "north": n}


def is_valid(path):
    """True if the file exists and every band can be read completely."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            for b in range(1, src.count + 1):
                src.read(b)
        return True
    except Exception:
        return False


def download(conn, bbox, start, end, out_path):
    part = out_path.with_suffix(".part.tif")
    cube = conn.load_collection("SENTINEL2_L2A", spatial_extent=bbox,
                                temporal_extent=[start, end],
                                bands=cfg['image']['s2_bands'])
    cube.median_time().download(str(part))
    if not is_valid(part):
        part.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is unreadable")
    part.rename(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tiles", type=int, default=0, help="only the first N tiles (for a test)")
    ap.add_argument("--workers", type=int, default=3, help="downloads running at the same time")
    args = ap.parse_args()

    tiles = sorted(p for p in RAW.glob("*.tif"))
    if args.max_tiles:
        tiles = tiles[:args.max_tiles]
    for q in SEASONS:
        (OUT / q).mkdir(parents=True, exist_ok=True)

    jobs = [(q, t) for t in tiles for q in SEASONS]
    print(f"{len(tiles)} tiles x {len(SEASONS)} seasons = {len(jobs)} files")

    local = threading.local()

    def connection():
        if not hasattr(local, "conn"):
            with login_lock:                     # one login at a time
                local.conn = openeo.connect("openeo.dataspace.copernicus.eu")
                local.conn.authenticate_oidc()
        return local.conn

    def work(item):
        k, n, q, t = item
        start, end = SEASONS[q]
        try:
            t0 = time.time()
            download(connection(), tile_bbox(t), start, end, OUT / q / t.name)
            print(f"[{k}/{n}] saved  {q} {t.name}  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{k}/{n}] failed {q} {t.name}: {str(e)[:160]}  (retried next pass)", flush=True)
            time.sleep(WAIT_AFTER_FAIL_S)

    for p in range(1, PASSES + 1):
        todo = [(q, t) for q, t in jobs if not is_valid(OUT / q / t.name)]
        print(f"\n=== Pass {p}/{PASSES}: {len(todo)} files to download, "
              f"{len(jobs) - len(todo)} already done, {args.workers} at a time ===", flush=True)
        if not todo:
            break
        items = [(k, len(todo), q, t) for k, (q, t) in enumerate(todo, 1)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, items))

    missing = [(q, t.name) for q, t in jobs if not is_valid(OUT / q / t.name)]
    print(f"\nDone: {len(jobs) - len(missing)}/{len(jobs)} files ready.")
    if missing:
        print("Still missing (run the script again to retry):")
        for q, name in missing:
            print(f"  {q}/{name}")


if __name__ == "__main__":
    main()
