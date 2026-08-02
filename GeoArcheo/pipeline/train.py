# pipeline/train.py
# PURPOSE:
#   Stage 2. The actual learning.
#   Loads Prithvi (frozen) + LoRA (trainable).
#   Runs training loop for cfg epochs.
#   Saves best checkpoint by validation accuracy.

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import yaml

from models.foundation import PrithviBackbone
from models.adapter import LoRAAdapter, SiteClassifier

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)


class ProcessedDataset(Dataset):
    """
    Loads .pt files from data/processed/.
    Each file = (image tensor, label integer).
    """

    def __init__(self):
        self.files = sorted([
            os.path.join(cfg['paths']['processed_data'], f)
            for f in os.listdir(cfg['paths']['processed_data'])
            if f.endswith('.pt')
        ])
        print(f"Dataset: {len(self.files)} patches found.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        return data['image'], data['label']


def train():

    device = torch.device(
        cfg['training']['device']
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}\n")

    # ── BUILD MODEL ──────────────────────────────────────────
    backbone   = PrithviBackbone()
    adapter    = LoRAAdapter(backbone)
    classifier = SiteClassifier()

    adapter    = adapter.to(device)
    classifier = classifier.to(device)

    # ── SPLIT DATA ───────────────────────────────────────────
    full    = ProcessedDataset()
    n       = len(full)
    n_train = int(n * cfg['training']['train_split'])  # 80%
    n_val   = int(n * cfg['training']['val_split'])    # 10%
    n_test  = n - n_train - n_val                      # 10%

    train_set, val_set, _ = random_split(
        full, [n_train, n_val, n_test]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,     # random order every epoch
        num_workers=2
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )

    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}\n")

    # ── LOSS + OPTIMISER ─────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss()

    # Only trainable params: LoRA A, B + classifier weights
    # Frozen Prithvi weights excluded automatically
    trainable_params = (
        list(adapter.parameters()) +
        list(classifier.parameters())
    )
    optimiser = torch.optim.Adam(
        trainable_params,
        lr=cfg['training']['learning_rate']  # 0.0001
    )

    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
    best_val_acc = 0.0

    # ── EPOCH LOOP ───────────────────────────────────────────
    for epoch in range(cfg['training']['epochs']):

        # ── TRAIN PHASE ──────────────────────────────────────
        adapter.train()
        classifier.train()

        total_loss    = 0.0
        train_correct = 0
        train_total   = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"
        ):
            images = images.to(device)  # (32, 6, 224, 224)
            labels = labels.to(device)  # (32,)

            # FORWARD PASS
            # image → 768-number embedding
            embeddings = adapter(images)   # (32, 768)

            # embedding → 6 class scores
            scores = classifier(embeddings)  # (32, 6)

            # how wrong were we?
            loss = loss_fn(scores, labels)

            # BACKWARD PASS
            optimiser.zero_grad()
            loss.backward()
            # nudge only LoRA A, B + classifier
            # Prithvi: no gradient, no nudge
            optimiser.step()

            total_loss    += loss.item()
            predicted      = scores.argmax(dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total   += len(labels)

        train_acc = train_correct / train_total
        avg_loss  = total_loss / len(train_loader)

        # ── VALIDATION PHASE ─────────────────────────────────
        adapter.eval()
        classifier.eval()

        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                embeddings = adapter(images)
                scores     = classifier(embeddings)
                predicted  = scores.argmax(dim=1)

                val_correct += (predicted == labels).sum().item()
                val_total   += len(labels)

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch+1}/"
              f"{cfg['training']['epochs']}")
        print(f"  Loss:      {avg_loss:.4f}")
        print(f"  Train acc: {train_acc:.3f}")
        print(f"  Val acc:   {val_acc:.3f}")

        # Save only when validation accuracy improves
        # Final saved model = best generalisation epoch
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':            epoch + 1,
                'adapter_state':    adapter.state_dict(),
                'classifier_state': classifier.state_dict(),
                'val_acc':          val_acc
            }, cfg['paths']['checkpoint_path'])
            print(f"  ✓ Saved best model "
                  f"(val_acc={val_acc:.3f})")

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.3f}")


if __name__ == "__main__":
    train()
