# pipeline/infer.py
#
# PURPOSE:
#   Stage 4. Inference on a NEW satellite tile.
#   Slides 224x224 window across full tile.
#   Runs complete pipeline on every patch.
#   Saves CONFIRMED and CANDIDATE site detections.

import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import yaml

from models.foundation import PrithviBackbone
from models.adapter import LoRAAdapter, SiteClassifier, PrototypeMatcher
from models.uncertainty import ConformalPredictor
from models.symbolic import ScallopReasoner

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


def load_all_models(device):
    """Load every trained component."""

    backbone   = PrithviBackbone()
    adapter    = LoRAAdapter(backbone)
    classifier = SiteClassifier()

    ckpt = torch.load(
        cfg['paths']['checkpoint_path'],
        map_location=device
    )
    adapter.load_state_dict(ckpt['adapter_state'])
    classifier.load_state_dict(ckpt['classifier_state'])

    adapter    = adapter.to(device)
    classifier = classifier.to(device)
    adapter.eval()
    classifier.eval()

    matcher   = PrototypeMatcher()

    predictor = ConformalPredictor()
    predictor.load()

    reasoner  = ScallopReasoner()

    print("All models loaded.")
    return adapter, classifier, matcher, predictor, reasoner


def infer(tif_path, stride=112):
    """
    stride=112 = 50% overlap between patches.
    Ensures sites near patch boundaries are not missed.
    """
    device = torch.device(
        cfg['training']['device']
        if torch.cuda.is_available()
        else "cpu"
    )

    adapter, classifier, matcher, predictor, reasoner = \
        load_all_models(device)

    os.makedirs(cfg['paths']['results_dir'], exist_ok=True)

    size    = cfg['image']['size']    # 224
    factor  = cfg['image']['normalize_factor']  # 10000
    results = []

    print(f"\nInference on: {tif_path}")

    with rasterio.open(tif_path) as src:
        height = src.height
        width  = src.width
        print(f"Tile: {width} x {height} pixels")

        total = (
            ((height - size) // stride + 1) *
            ((width  - size) // stride + 1)
        )

        with tqdm(total=total, desc="Scanning") as pbar:

            row = 0
            while row + size <= height:
                col = 0
                while col + size <= width:

                    # Cut 224x224 patch
                    window = Window(
                        col_off=col, row_off=row,
                        width=size, height=size
                    )
                    patch = src.read(window=window)
                    patch = (patch.astype(np.float32)
                             / factor)

                    image_t = (torch.tensor(patch)
                               .unsqueeze(0)
                               .to(device))

                    with torch.no_grad():

                        # Step 1: embedding (768 numbers)
                        emb = adapter(image_t).squeeze(0)

                        # Step 2: softmax probabilities
                        scores  = classifier(emb.unsqueeze(0))
                        softmax = torch.softmax(
                            scores, dim=1
                        ).squeeze(0)

                        # Step 3: prototype matching
                        pred_class, pred_score, all_scores =\
                            matcher.match(emb.cpu())

                        # Step 4: conformal prediction set
                        pred_set = predictor.predict_set(
                            softmax.cpu()
                        )

                        # Step 5: symbolic spatial check
                        # Replace these with real DEM lookups
                        # for your specific region
                        water_dist = 300.0
                        elevation  = 200.0
                        slope      = 10.0

                        decision, rules, reason = \
                            reasoner.decide(
                                pred_score,
                                water_dist,
                                elevation,
                                slope
                            )

                    if decision in ("CONFIRMED", "CANDIDATE"):
                        results.append({
                            'row':        row,
                            'col':        col,
                            'prediction': pred_class,
                            'score':      pred_score,
                            'pred_set':   pred_set,
                            'decision':   decision,
                            'reason':     reason,
                            'all_scores': all_scores
                        })

                    col   += stride
                    pbar.update(1)
                row += stride

    confirmed  = sum(1 for r in results
                     if r['decision'] == 'CONFIRMED')
    candidates = sum(1 for r in results
                     if r['decision'] == 'CANDIDATE')

    print(f"\nResults:")
    print(f"  CONFIRMED:  {confirmed}")
    print(f"  CANDIDATE:  {candidates}")

    save_path = os.path.join(
        cfg['paths']['results_dir'], "detections.pt"
    )
    torch.save(results, save_path)
    print(f"  Saved → {save_path}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline/infer.py "
              "data/raw/your_tile.tif")
    else:
        infer(sys.argv[1])
