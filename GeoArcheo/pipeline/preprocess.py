# pipeline/preprocess.py
#
# PURPOSE:
#   Stage 1. Run ONCE before training.
#   Converts giant satellite tiles + EAMENA GeoJSON
#   into clean 224x224 patch files.
#
# INPUT:
#   data/raw/    Sentinel-2 .tif tiles
#   data/labels/ eamena_sites.geojson
#
# OUTPUT:
#   data/processed/ one .pt file per site
#                   {image: tensor(6,224,224), label: int,
#                    site_type: str, site_id: str}

import os
import json
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def latlon_to_pixel(tif_path, latitude, longitude):
    """
    Convert GPS coordinates → pixel position in tile.
    rasterio reads embedded geographic metadata to do this.
    """
    with rasterio.open(tif_path) as src:
        row, col = src.index(longitude, latitude)
    return int(row), int(col)


def extract_patch(tif_path, center_row, center_col):
    """
    Cut 224x224 block from giant tile centred on site.
    Only loads this small window — not the full tile.

    OUTPUT: numpy (6, 224, 224), values 0 to ~10000
    """
    size = cfg['image']['size']   # 224
    half = size // 2              # 112

    with rasterio.open(tif_path) as src:
        window = Window(
            col_off=center_col - half,
            row_off=center_row - half,
            width=size,
            height=size
        )
        patch = src.read(window=window)

    return patch  # (6, 224, 224)


def normalize(patch):
    """
    0-10000 → 0.0-1.0
    Source of 10000: cfg['image']['normalize_factor']
    """
    factor = cfg['image']['normalize_factor']
    return patch.astype(np.float32) / factor


def preprocess_all():

    os.makedirs(cfg['paths']['processed_data'], exist_ok=True)

    # Build label lookup from config
    # {"burial_mound": 0, "ancient_settlement": 1, ...}
    site_types   = cfg['sites']['classes']
    label_to_int = {name: i for i, name in enumerate(site_types)}

    with open(cfg['paths']['eamena_geojson']) as f:
        geojson = json.load(f)

    print(f"Found {len(geojson['features'])} sites.")
    print(f"Keeping: {site_types}\n")

    done    = 0
    skipped = 0

    for feature in tqdm(geojson['features'],
                        desc="Preprocessing"):

        props      = feature['properties']
        site_id    = props['site_id']
        site_type  = props['site_type']
        image_file = props['image_file']

        # GeoJSON: coordinates = [longitude, latitude]
        coords    = feature['geometry']['coordinates']
        longitude = coords[0]
        latitude  = coords[1]

        if site_type not in label_to_int:
            skipped += 1
            continue

        tif_path = os.path.join(
            cfg['paths']['raw_data'], image_file
        )
        if not os.path.exists(tif_path):
            print(f"\nMissing: {tif_path}")
            skipped += 1
            continue

        try:
            # GPS → pixel position in giant tile
            row, col = latlon_to_pixel(
                tif_path, latitude, longitude
            )

            # Cut 224x224 patch around that position
            patch = extract_patch(tif_path, row, col)

            # 0-10000 → 0.0-1.0
            patch = normalize(patch)

            # numpy → torch tensor (6, 224, 224)
            image_tensor = torch.tensor(patch)

            # string label → integer
            label = label_to_int[site_type]

            # Save finished pair
            save_path = os.path.join(
                cfg['paths']['processed_data'],
                f"{site_id}.pt"
            )
            torch.save({
                'image':     image_tensor,
                'label':     label,
                'site_type': site_type,
                'site_id':   site_id
            }, save_path)

            done += 1

        except Exception as e:
            print(f"\nError {site_id}: {e}")
            skipped += 1

    print(f"\nDone. Saved: {done} | Skipped: {skipped}")
    print(f"Output: {cfg['paths']['processed_data']}")


if __name__ == "__main__":
    preprocess_all()
