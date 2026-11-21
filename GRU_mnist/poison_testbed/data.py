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


# ----------------------------------------------------------------------------- xp5: datasets and triggers
# Normalisation per dataset (train-set pixel mean / std). MNIST keeps the constants above, so load_mnist is unchanged.
NORM = {"mnist": (MEAN, STD), "fmnist": (0.2860, 0.3530)}
BLEND_ALPHA = 0.2


@dataclass(frozen=True)
class Data:
    train: TensorDataset
    val: tuple[torch.Tensor, torch.Tensor]
    test: tuple[torch.Tensor, torch.Tensor]
    triggered: tuple[torch.Tensor, torch.Tensor]  # test images with true label != target, with the run's trigger
    flip_source: tuple[torch.Tensor, torch.Tensor]  # test images of class flip_source (label flip success)
    flip_pool: torch.Tensor  # training images of class flip_source (CPU): what a label-flip attacker feeds in


def _tensors_of(dataset: str, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}[dataset]
    mean, std = NORM[dataset]
    ds = cls(str(DATA_ROOT), train=train, download=True)
    return (ds.data.float().div(255.0).sub(mean).div(std)).unsqueeze(1), ds.targets.clone().long()


def _checkerboard() -> torch.Tensor:
    """Fixed blend pattern in pixel space: 2x2-pixel checkerboard of 0 / 1, shape (28, 28)."""
    i = torch.arange(28)
    return (((i[:, None] // 2) + (i[None, :] // 2)) % 2).float()


def make_trigger(dataset: str, trigger: str, patch_size: int):
    """Return stamp(x) -> triggered copy of x (B,1,28,28), normalised with the dataset's statistics.

    "corner"           white patch_size x patch_size square, bottom-right (as in xp1-xp4)
    "patch_r{r}_c{c}"  the same white square with its top-left pixel at (r, c)
    "blend"            x_pixel -> (1 - BLEND_ALPHA) x_pixel + BLEND_ALPHA * checkerboard (whole image)
    """
    mean, std = NORM[dataset]
    white = (1.0 - mean) / std
    if trigger == "corner":
        def stamp(x):
            x = x.clone()
            x[..., -patch_size - 1: -1, -patch_size - 1: -1] = white
            return x
        return stamp
    if trigger.startswith("patch_r"):
        r, c = (int(v) for v in trigger[len("patch_r"):].split("_c"))

        def stamp(x):
            x = x.clone()
            x[..., r: r + patch_size, c: c + patch_size] = white
            return x
        return stamp
    if trigger == "blend":
        pattern = (_checkerboard() - mean) / std

        def stamp(x):
            return (1 - BLEND_ALPHA) * x + BLEND_ALPHA * pattern.to(x.device)
        return stamp
    raise ValueError(f"unknown trigger {trigger!r}")


@lru_cache(maxsize=8)
def load_data(dataset: str, device: str, trigger: str, patch_size: int, target: int, flip_source: int) -> Data:
    x, y = _tensors_of(dataset, train=True)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SPLIT_SEED))
    val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    xt, yt = _tensors_of(dataset, train=False)
    keep = yt != target
    src = yt == flip_source
    stamp = make_trigger(dataset, trigger, patch_size)
    dev = torch.device(device)
    return Data(
        train=TensorDataset(x[train_idx], y[train_idx]),
        val=(x[val_idx].to(dev), y[val_idx].to(dev)),
        test=(xt.to(dev), yt.to(dev)),
        triggered=(stamp(xt[keep]).to(dev), yt[keep].to(dev)),
        flip_source=(xt[src].to(dev), yt[src].to(dev)),
        flip_pool=x[train_idx][y[train_idx] == flip_source],
    )
