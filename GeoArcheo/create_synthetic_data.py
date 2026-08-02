# create_synthetic_data.py
#
# PURPOSE:
#   Generate synthetic training data for testing the pipeline
#   when real Sentinel-2 tiles and EAMENA labels are not
#   yet available.
#
#   Creates:
#     data/processed/  → fake .pt patch files
#     data/labels/     → fake eamena_sites.geojson
#
# WHY NEEDED:
#   Real Sentinel-2 tiles are large downloads.
#   Real EAMENA data requires database access.
#   Synthetic data lets you test every pipeline file
#   immediately to confirm the code works end to end.
#
# WHAT IT CREATES:
#   N synthetic 224x224x6 image patches
#   Each randomly assigned one of the 6 site types
#   Each saved as a .pt file in data/processed/
#   Also creates a matching GeoJSON in data/labels/
#
# RUN:
#   python create_synthetic_data.py
#   python create_synthetic_data.py --n_samples 200

import os
import json
import argparse
import numpy as np
import torch
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def create_synthetic_patch(site_type_idx):
    """
    Create one fake satellite image patch.

    Real patches have meaningful spectral patterns.
    Synthetic patches have:
      - random base values
      - slight bias per site type
        (so the model has SOMETHING to learn)
      - gaussian noise on top

    INPUT:
        site_type_idx: int 0-5
        Which site type this patch represents.

    OUTPUT:
        tensor: shape (6, 224, 224)
        values: float32, range 0.0 to 1.0
    """
    bands  = cfg['image']['bands']    # 6
    size   = cfg['image']['size']     # 224

    # Base spectral signature differs per site type
    # This gives the model a weak signal to learn from
    # Real data has much stronger and more complex signals
    base_values = {
        0: [0.3, 0.4, 0.5, 0.6, 0.3, 0.2],  # burial_mound
        1: [0.4, 0.5, 0.4, 0.5, 0.4, 0.3],  # settlement
        2: [0.2, 0.3, 0.4, 0.7, 0.5, 0.4],  # temple
        3: [0.5, 0.5, 0.5, 0.4, 0.3, 0.3],  # road
        4: [0.2, 0.4, 0.3, 0.8, 0.6, 0.5],  # canal
        5: [0.3, 0.3, 0.4, 0.5, 0.4, 0.3],  # caravanserai
    }

    patch = np.zeros((bands, size, size), dtype=np.float32)

    for b in range(bands):
        base  = base_values[site_type_idx][b]
        # Add spatial noise to make patches look varied
        noise = np.random.normal(0, 0.05, (size, size))
        patch[b] = np.clip(base + noise, 0.0, 1.0)

    return torch.tensor(patch)


def create_synthetic_geojson(site_ids, site_types,
                              output_path):
    """
    Create a fake GeoJSON file mimicking EAMENA format.

    Each feature represents one archaeological site with:
      - site_id: unique identifier
      - site_type: one of the 6 classes
      - image_file: which satellite tile to look in
      - geometry: fake GPS coordinates

    NOTE: preprocess.py reads this file.
    Synthetic version uses placeholder coordinates
    since we are generating patches directly.
    """
    features = []

    for i, (site_id, site_type) in enumerate(
        zip(site_ids, site_types)
    ):
        # Fake coordinates in MENA region
        # Iraq area: lat 31-35, lon 44-48
        latitude  = 31.0 + np.random.uniform(0, 4)
        longitude = 44.0 + np.random.uniform(0, 4)

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [longitude, latitude]
            },
            "properties": {
                "site_id":    site_id,
                "site_type":  site_type,
                "image_file": f"synthetic_tile_{i//50}.tif",
                "pixel_row":  np.random.randint(100, 900),
                "pixel_col":  np.random.randint(100, 900)
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"GeoJSON saved → {output_path}")


def create_synthetic_processed_patches(n_samples):
    """
    Create n_samples fake .pt patch files directly
    in data/processed/ — bypassing preprocess.py.

    This is the fastest way to test the training pipeline
    without real satellite data.
    """
    processed_dir = cfg['paths']['processed_data']
    os.makedirs(processed_dir, exist_ok=True)

    site_types   = cfg['sites']['classes']
    label_to_int = {
        name: i for i, name in
