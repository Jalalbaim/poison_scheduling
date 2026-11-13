"""Observable signals X, computed from the received gradients only (before aggregation).

These functions take the (n, d) gradient stack, the losses of data-bearing
micro-batches and the fixed reference threshold C. They never see which workers
are Byzantine, nor whether an attack is active.
"""

import torch

from krum.primitives.aggregators.median import Median

X_COLUMNS = (
    "x_norm_mean",
    "x_norm_std",
    "x_pdist_mean",
    "x_maxdist_median",
    "x_cos_avg_median",
    "x_frac_clipped",
    "x_train_loss",
)


@torch.no_grad()
def round_signals(gradients: torch.Tensor, losses: torch.Tensor, C: float) -> dict[str, float]:
    """Per-round signals. ``gradients`` is (n, d); ``losses`` holds one loss per data-bearing micro-batch."""
    norms = gradients.norm(dim=1)
    avg = gradients.mean(dim=0)
    med = Median.aggregate(gradients)
    values = torch.stack([
        norms.mean(),
        norms.std(correction=0),
        torch.pdist(gradients).mean(),
        (gradients - med).norm(dim=1).max(),
        torch.nn.functional.cosine_similarity(avg, med, dim=0),
        (norms > C).float().mean(),
        losses.mean(),
    ])
    return dict(zip(X_COLUMNS, values.tolist()))
