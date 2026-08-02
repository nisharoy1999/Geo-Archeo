# models/adapter.py
#
# PURPOSE:
#   LoRAAdapter      inject A(768x8) B(8x768) into Prithvi
#   SiteClassifier   768 → 6 scores  (training only)
#   PrototypeMatcher cosine similarity (inference only)
#
# CALLED BY:
#   pipeline/train.py    uses LoRAAdapter + SiteClassifier
#   pipeline/evaluate.py uses LoRAAdapter + PrototypeMatcher
#   pipeline/infer.py    uses LoRAAdapter + PrototypeMatcher

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from models.foundation import PrithviBackbone
import yaml

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


class LoRAAdapter(nn.Module):
    """
    Injects LoRA into frozen Prithvi.

    A: (768 x rank) starts random small numbers
    B: (rank x 768) starts ALL ZEROS

    B=0 at start means:
        correction = A x B = 0
        model = pure Prithvi on day 1
        no knowledge destroyed

    Training nudges B away from zero.
    After training A x B = archaeological adjustment.
    """

    def __init__(self, backbone: PrithviBackbone):
        super().__init__()

        lora_config = LoraConfig(
            # rank=8: A=(768x8) B=(8x768)
            # 12,288 params per layer vs 589,824 in W_Q
            r=cfg['lora']['rank'],

            # scaling = alpha/rank = 16/8 = 2.0
            lora_alpha=cfg['lora']['alpha'],

            # zero out 10% of correction during training
            lora_dropout=cfg['lora']['dropout'],

            # inject into W_Q and W_V only
            target_modules=cfg['lora']['target_modules'],

            bias="none"
        )

        # PEFT inserts A and B into Prithvi's attention layers
        # B initialised to zero automatically
        self.model = get_peft_model(backbone.encoder, lora_config)

        print("\nLoRA injected into Prithvi.")
        self.model.print_trainable_parameters()
        # trainable params: ~2.9M || all params: ~102M (~2.87%)

    def forward(self, images):
        """
        INPUT:  (batch_size, 6, 224, 224)
        OUTPUT: (batch_size, 768)

        W_Q output + (A x B correction) = final Query
        W_V output + (A x B correction) = final Value
        """
        outputs = self.model(pixel_values=images)
        return outputs.last_hidden_state[:, 0, :]

    def save_lora(self, path):
        """Save only LoRA weights (~12MB not full 400MB)"""
        self.model.save_pretrained(path)
        print(f"LoRA weights saved → {path}")

    def load_lora(self, path):
        """Load previously trained LoRA weights"""
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, path)
        print(f"LoRA weights loaded ← {path}")


class SiteClassifier(nn.Module):
    """
    Used ONLY during training.

    FLOW:
    (batch, 768)
    → LayerNorm    stabilise numbers
    → Dropout 30%  prevent memorising training images
    → Linear 768→256
    → ReLU         zero out negatives
    → Linear 256→6
    → (batch, 6)   one score per site type
    """

    def __init__(self):
        super().__init__()

        emb = cfg['model']['embedding_dim']  # 768
        n   = cfg['sites']['num_classes']    # 6

        self.head = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Dropout(0.3),
            nn.Linear(emb, 256),
            nn.ReLU(),
            nn.Linear(256, n)
            # No softmax: CrossEntropyLoss applies it internally
        )

    def forward(self, embeddings):
        # (batch, 768) → (batch, 6)
        return self.head(embeddings)


class PrototypeMatcher:
    """
    Used ONLY during zero-shot inference.

    Loads 6 prototype vectors saved by evaluate.py.
    Compares new image embedding vs all prototypes
    using cosine similarity.
    Highest similarity = predicted site type.
    Below threshold = unknown new site type.
    """

    def __init__(self):
        proto_path = cfg['paths']['prototype_path']
        self.prototypes = torch.load(proto_path)
        self.threshold  = cfg['zero_shot']['unknown_threshold']
        self.site_names = cfg['sites']['classes']

        print(f"Loaded {len(self.prototypes)} prototypes:")
        for name, vec in self.prototypes.items():
            print(f"  {name}: {vec.shape}")

    def match(self, embedding: torch.Tensor):
        """
        INPUT:  embedding (768,)
        OUTPUT:
            best_class: string e.g. "burial_mound"
            best_score: float 0 to 1
            all_scores: dict of all 6 similarities
        """
        all_scores = {}

        for site_type, prototype in self.prototypes.items():
            sim = F.cosine_similarity(
                embedding.unsqueeze(0),
                prototype.unsqueeze(0)
            )
            all_scores[site_type] = sim.item()

        best_class = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_class]

        if best_score < self.threshold:
            return "unknown_site_type", best_score, all_scores

        return best_class, best_score, all_scores


# ── TEST ─────────────────────────────────────────────────────
if __name__ == "__main__":
    backbone   = PrithviBackbone()
    adapter    = LoRAAdapter(backbone)
    classifier = SiteClassifier()

    fake = torch.randn(
        2,
        cfg['image']['bands'],
        cfg['image']['size'],
        cfg['image']['size']
    )

    embeddings = adapter(fake)
    scores     = classifier(embeddings)

    print(f"Embeddings: {embeddings.shape}")  # [2, 768]
    print(f"Scores:     {scores.shape}")      # [2, 6]
    print("adapter.py OK")
