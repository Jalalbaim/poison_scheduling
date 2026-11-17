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


# ----------------------------------------------------------------------------- signal set v2 (xp4)
# Tail, scale-free and memory signals. Same rule as v1: computed from what the trainer sees in the round (the
# micro-batch gradients, the per-sample losses and the training labels as given, poisoned or not), never from
# which workers are Byzantine, whether an attack is active, the trigger or the target class.
X2_SIGNALS = (
    # A. per-sample loss tail
    "loss_max_ratio",       # max sample loss / median sample loss
    "loss_top3_ratio",      # mean of the 3 largest sample losses / median
    "loss_tail_frac",       # share of samples above the 99th percentile of the previous TAIL_ROUNDS rounds' losses
    # B. label-concentrated loss
    "label_hiloss_max",     # among the TOPK highest-loss samples, the largest share carrying one same label
    "label_loss_gap",       # max over labels of the label's mean loss / median over labels
    # C. outlier micro-batches
    "krum_top_ratio",       # mean of the f largest Krum scores / median Krum score
    "mb_loss_max_ratio",    # max micro-batch loss / median micro-batch loss
    "norm_max_ratio",       # max micro-batch gradient norm / median
    # D. low-rank structure of the centred gradients
    "sv_ratio",             # largest / second singular value
    "sv_top_share",         # share of variance on the top direction
    # E. cross-round consistency
    "outlier_consistency",  # cosine between this round's outlier deviation and the running outlier direction
    "outlier_proj_max",     # max over micro-batches of cos(g_i - median, running outlier direction)
    # F. drift-normalised: z-score against the same signal over the previous Z_ROUNDS rounds
    "z_loss_max_ratio", "z_krum_top_ratio", "z_sv_ratio", "z_mb_loss_max_ratio", "z_norm_std", "z_train_loss",
)
TAIL_ROUNDS, Z_ROUNDS, TOPK, EMA = 20, 50, 8, 0.9
Z_BASE = {"z_loss_max_ratio": "loss_max_ratio", "z_krum_top_ratio": "krum_top_ratio", "z_sv_ratio": "sv_ratio",
          "z_mb_loss_max_ratio": "mb_loss_max_ratio", "z_norm_std": "norm_std", "z_train_loss": "train_loss"}


class SignalTrackerV2:
    """Per-round v2 signals; keeps the cross-round state (loss history, running outlier direction, z windows)."""

    def __init__(self, n: int, f: int):
        self.n, self.f = n, f
        self.loss_hist: list[torch.Tensor] = []
        self.direction: torch.Tensor | None = None
        self.z_hist: dict[str, list[float]] = {k: [] for k in Z_BASE.values()}

    @torch.no_grad()
    def __call__(self, gradients: torch.Tensor, mb_losses: torch.Tensor, sample_losses: torch.Tensor,
                 sample_labels: torch.Tensor) -> dict[str, float]:
        eps = 1e-12
        g = gradients.double()
        n, f = g.shape[0], self.f
        out: dict[str, float] = {}
        # A
        sl = sample_losses.double()
        med_l = sl.median().clamp_min(eps)
        top = sl.topk(min(3, len(sl))).values
        out["loss_max_ratio"] = float(sl.max() / med_l)
        out["loss_top3_ratio"] = float(top.mean() / med_l)
        if self.loss_hist:
            q99 = torch.quantile(torch.cat(self.loss_hist), 0.99)
            out["loss_tail_frac"] = float((sl > q99).double().mean())
        else:
            out["loss_tail_frac"] = 0.0
        self.loss_hist = (self.loss_hist + [sl])[-TAIL_ROUNDS:]
        # B
        k = min(TOPK, len(sl))
        hi_labels = sample_labels[sl.topk(k).indices]
        out["label_hiloss_max"] = float(torch.bincount(hi_labels).max()) / k
        labels = sample_labels.unique()
        per_label = torch.stack([sl[sample_labels == c].mean() for c in labels])
        out["label_loss_gap"] = float(per_label.max() / per_label.median().clamp_min(eps))
        # C
        d2 = torch.cdist(g, g) ** 2
        m = max(1, n - f - 2)
        krum = d2.sort(dim=1).values[:, 1: m + 1].sum(1)  # skip the zero self-distance
        out["krum_top_ratio"] = float(krum.topk(f).values.mean() / krum.median().clamp_min(eps))
        ml = mb_losses.double()
        out["mb_loss_max_ratio"] = float(ml.max() / ml.median().clamp_min(eps))
        norms = g.norm(dim=1)
        out["norm_max_ratio"] = float(norms.max() / norms.median().clamp_min(eps))
        # D (singular values from the n x n Gram matrix of the centred gradients)
        c = g - g.mean(0)
        ev = torch.linalg.eigvalsh(c @ c.T).flip(0).clamp_min(0)
        sv = ev.sqrt()
        out["sv_ratio"] = float(sv[0] / sv[1].clamp_min(eps))
        out["sv_top_share"] = float(ev[0] / ev.sum().clamp_min(eps))
        # E (outliers = the f micro-batches with the largest Krum scores)
        med = g.sort(dim=0).values[(n - 1) // 2]  # coordinate-wise median (deterministic on CUDA)
        dev = g - med
        idx = krum.topk(f).indices
        d = dev[idx].mean(0)
        d = d / d.norm().clamp_min(eps)
        if self.direction is None:
            out["outlier_consistency"], out["outlier_proj_max"] = 0.0, 0.0
            self.direction = d
        else:
            v = self.direction / self.direction.norm().clamp_min(eps)
            out["outlier_consistency"] = float(d @ v)
            out["outlier_proj_max"] = float((dev @ v / dev.norm(dim=1).clamp_min(eps)).max())
            self.direction = EMA * self.direction + (1 - EMA) * d
        # F
        base = {**out, "norm_std": float(norms.std(correction=0)), "train_loss": float(ml.mean())}
        for zname, b in Z_BASE.items():
            h = self.z_hist[b]
            if len(h) >= 5:
                t = torch.tensor(h, dtype=torch.float64)
                out[zname] = float((base[b] - t.mean()) / t.std(correction=0).clamp_min(1e-9))
            else:
                out[zname] = 0.0
            self.z_hist[b] = (h + [base[b]])[-Z_ROUNDS:]
        return {f"x2_{k}": out[k] for k in X2_SIGNALS}


X2_COLUMNS = tuple(f"x2_{s}_{agg}" for s in X2_SIGNALS for agg in ("mean", "max"))
