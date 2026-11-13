"""MNIST as preloaded, normalised tensors, plus a clean val split and a triggered test set."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

MEAN, STD = 0.1307, 0.3081
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
VAL_SIZE = 5_000
SPLIT_SEED = 0  # fixed: the train/val split is identical in every run


@dataclass(frozen=True)
class Mnist:
    train: TensorDataset  # 55k, on CPU; sampled by the workers' DataLoaders
    val: tuple[torch.Tensor, torch.Tensor]  # 5k clean held-out
    test: tuple[torch.Tensor, torch.Tensor]  # 10k
    triggered: tuple[torch.Tensor, torch.Tensor]  # test images with true label != target, patched; y = true label


def _tensors(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.MNIST(str(DATA_ROOT), train=train, download=True)
    x = (ds.data.float().div(255.0).sub(MEAN).div(STD)).unsqueeze(1)  # (N,1,28,28)
    return x, ds.targets.clone().long()


def patch_value() -> float:
    """Normalised value of a white pixel."""
    return (1.0 - MEAN) / STD


def apply_patch(x: torch.Tensor, size: int) -> torch.Tensor:
    """Return a copy of ``x`` (B,1,28,28) with a white ``size``×``size`` square in the bottom-right corner."""
    x = x.clone()
    x[..., -size - 1 : -1, -size - 1 : -1] = patch_value()
    return x


@lru_cache(maxsize=4)
def load_mnist(device: str, patch_size: int, target: int) -> Mnist:
    x, y = _tensors(train=True)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SPLIT_SEED))
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    xt, yt = _tensors(train=False)
    keep = yt != target
    dev = torch.device(device)
    return Mnist(
        train=TensorDataset(x[train_idx], y[train_idx]),
        val=(x[val_idx].to(dev), y[val_idx].to(dev)),
        test=(xt.to(dev), yt.to(dev)),
        triggered=(apply_patch(xt[keep], patch_size).to(dev), yt[keep].to(dev)),
    )
