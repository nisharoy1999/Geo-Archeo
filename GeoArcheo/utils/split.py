# utils/split.py
#
# One place that defines the train/val/test split, so train.py and
# evaluate.py always agree on which patches are held out.

import torch
from torch.utils.data import random_split


def make_splits(dataset, val_frac, test_frac, seed=42):
    n = len(dataset)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Dataset too small ({n}) to split.")
    return random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )