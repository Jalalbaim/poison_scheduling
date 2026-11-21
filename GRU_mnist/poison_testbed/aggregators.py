"""ClipNoise and ScheduledAggregator, following the library's stateless classmethod convention."""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from krum.primitives.aggregators import Aggregator


class ClipNoise(Aggregator):
    r"""Clip each worker gradient to L2 norm ``C``, average, add Gaussian noise.

    Noise is :math:`\mathcal{N}(0, (\sigma C)^2 / n)` per coordinate, i.e. std
    :math:`\sigma C / \sqrt{n}` (as specified; note DP-SGD's averaged convention
    would be :math:`\sigma C / n`).
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        C: float,
        sigma: float,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        **specialized: Any,
    ) -> Tensor:
        """Aggregate.

        Args:
            gradients: (n, d) worker gradients.
            out: Optional output buffer of shape (d,).
            C: Clipping threshold (L2).
            sigma: Noise multiplier.
            noise: Optional pre-drawn standard-normal tensor of shape (d,). Lets the
                caller draw noise every round from its own stream (common random
                numbers across schedules). If ``None``, drawn from ``generator``.
            generator: RNG used when ``noise`` is None.
        """
        if C <= 0:
            raise ValueError(f"Expected C > 0, got {C!r}")
        if sigma < 0:
            raise ValueError(f"Expected sigma >= 0, got {sigma!r}")
        g = gradients if isinstance(gradients, Tensor) else torch.stack(list(gradients))
        n = g.shape[0]
        norms = g.norm(dim=1, keepdim=True)
        scale = torch.clamp(C / norms.clamp_min(1e-12), max=1.0)
        agg = (g * scale).mean(dim=0)
        if sigma > 0:
            if noise is None:
                noise = torch.randn(agg.shape, generator=generator, device=agg.device, dtype=agg.dtype)
            agg = agg + noise * (sigma * C / math.sqrt(n))
        if out is not None:
            out.copy_(agg)
            return out
        return agg


class ScheduledAggregator(Aggregator):
    """Dispatch to ``plain`` or ``defense`` by block: ``schedule[round // block_len]``.

    Stateless: the caller passes the current ``round``. ``n``/``f`` (and any extra
    kwargs) go to both rules; ``plain_kwargs``/``defense_kwargs`` to one each.
    """

    @classmethod
    def aggregate(
        cls,
        gradients: Sequence[Tensor] | Tensor,
        /,
        out: Tensor | None = None,
        *,
        round: int,
        plain: type[Aggregator],
        defense: type[Aggregator],
        schedule: Sequence[int],
        block_len: int,
        plain_kwargs: dict[str, Any] | None = None,
        defense_kwargs: dict[str, Any] | None = None,
        **specialized: Any,
    ) -> Tensor:
        block = round // block_len
        if not 0 <= block < len(schedule):
            raise ValueError(f"round {round} is outside the schedule ({len(schedule)} blocks of {block_len})")
        if schedule[block]:
            return defense.aggregate(gradients, out, **specialized, **(defense_kwargs or {}))
        return plain.aggregate(gradients, out, **specialized, **(plain_kwargs or {}))

    @staticmethod
    def active(round: int, schedule: Sequence[int], block_len: int) -> int:
        return int(schedule[round // block_len])
