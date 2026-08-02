# models/foundation.py
#
# PURPOSE:
#   Load Prithvi-100M from HuggingFace.
#   Freeze ALL its weights (100M numbers, never change).
#   Given an image → return 768-number embedding.
#
# WHERE IT GETS CALLED:
#   models/adapter.py uses this as its base
#   pipeline/train.py builds adapter ON TOP of this
#
# INPUT  → OUTPUT:
#   image batch (batch_size, 6, 224, 224)
#   → embedding (batch_size, 768)

import torch
import torch.nn as nn
from transformers import AutoModel
import yaml

# Load config — all numbers come from here, not hardcoded
with open("GeoArcheo/configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


class PrithviBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        print("Loading Prithvi-100M...")
        print("First run downloads ~400MB, then uses local cache.")
        print(cfg['model']['prithvi_name'])
        print(cfg["num_labels"])

        # AutoModel downloads the architecture AND
        # the pretrained weights (100M numbers IBM/NASA learned)
        # Source of model name: config.yaml → model → prithvi_name
        self.encoder = AutoModel.from_pretrained(
            cfg['model']['prithvi_name']
        )

    #     # FREEZE all weights
    #     # requires_grad = False means:
    #     # "during backprop, do NOT compute gradient for this"
    #     # "do NOT nudge this number"
    #     # Source: we want to keep IBM/NASA's knowledge intact
    #     total_frozen = 0
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False
    #         total_frozen += param.numel()

    #     print(f"Prithvi loaded. {total_frozen:,} weights frozen.")

    # def forward(self, images):
    #     """
    #     WHAT HAPPENS STEP BY STEP:

    #     1. images enter: shape (batch_size, 6, 224, 224)
    #        Source of 6:   Sentinel-2 bands (config: image.bands)
    #        Source of 224: Prithvi's required input (config: image.size)

    #     2. Prithvi internally slices each image:
    #        224 / 16 = 14 blocks per side
    #        14 × 14  = 196 patches
    #        + 1 CLS token = 197 tokens
    #        Source of 16: patch size (config: image.patch_size)

    #     3. Each patch: 16×16×6 = 1536 raw pixel numbers
    #        × patch embedding matrix (1536×768, pretrained by IBM/NASA)
    #        = 768 numbers per token
    #        Source of 768: Prithvi's design (config: model.embedding_dim)

    #     4. 197 tokens × 768 numbers = shape (197, 768)
    #        Pass through 12 transformer layers
    #        Each layer: attention with W_Q, W_K, W_V matrices
    #        Each matrix: (768×768), pretrained, frozen

    #     5. After 12 layers: take token index 0 = CLS token
    #        CLS = summary of entire image
    #        Shape: (768,) per image

    #     OUTPUT: (batch_size, 768)
    #     """

    #     # pixel_values is the keyword Prithvi expects
    #     outputs = self.encoder(pixel_values=images)

    #     # last_hidden_state shape: (batch_size, 197, 768)
    #     # [:, 0, :] means: all batches, token 0, all 768 dims
    #     # token 0 = CLS token = image summary
    #     embeddings = outputs.last_hidden_state[:, 0, :]

    #     # shape: (batch_size, 768)
    #     return embeddings


# ── TEST THIS FILE ALONE ─────────────────────────────────────
if __name__ == "__main__":
    model = PrithviBackbone()

    # Fake image batch: 2 images
    # Numbers are random — just testing shapes, not real data
    # fake = torch.randn(
    #     2,                              # batch size
    #     cfg['image']['bands'],          # 6
    #     cfg['image']['size'],           # 224
    #     cfg['image']['size']            # 224
    # )

    # out = model(fake)
    # print(f"Input:  {fake.shape}")   # torch.Size([2, 6, 224, 224])
    # print(f"Output: {out.shape}")    # torch.Size([2, 768])
