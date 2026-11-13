"""PoisoningSimulation: the library's centralised SGD round, with data-poisoning workers,
a windowed attack, a per-round scheduled aggregator and signal capture.

Differences from ``CentralisedSimulation.step`` (library code is not modified):

* all n workers own a DataLoader over the *same* full training set, so each
  round's n micro-batches are IID draws from one dataset (micro-batch reading);
* every worker draws its micro-batch every round, attacked or not, so the data
  stream depends only on the seed (not on the schedule or the attack window);
* the f Byzantine workers either corrupt their micro-batch (data attack), or are
  replaced by a library ``Attack`` (gradient attack), or behave honestly
  (no attack / outside the window);
* worker order is shuffled each round (seeded) before signals and aggregation, so
  position cannot reveal who is Byzantine;
* ClipNoise noise is drawn every round from a dedicated stream, whether or not the
  defense is on (common random numbers across schedules);
* ``step`` returns the per-round signals X, computed before aggregation.
"""

from typing import Any

import torch
from torch.utils.data import DataLoader

from krum.primitives.aggregators import Aggregator
from krum.simulations.centralised.krum_nips_2017 import KrumSimulation

from .aggregators import ClipNoise, ScheduledAggregator
from .poisoning import AttackSpec
from .signals import round_signals

# Offsets that separate the seeded streams of one run.
PERM_STREAM, NOISE_STREAM = 10_000, 20_000


class PoisoningSimulation(KrumSimulation):
    def __init__(
        self,
        *,
        attack_spec: AttackSpec,
        corrupt_kwargs: dict[str, Any],
        plain: type[Aggregator],
        defense: type[Aggregator],
        defense_kwargs: dict[str, Any],
        schedule: list[int],
        block_len: int,
        C_ref: float,
        **kwargs: Any,
    ) -> None:
        # The library's own aggregator/attack slots are left empty: this subclass's step() handles both.
        super().__init__(aggregator=None, attack=None, **kwargs)
        self.attack_spec = attack_spec
        self.corrupt_kwargs = corrupt_kwargs
        self.plain = plain
        self.defense = defense
        self.defense_kwargs = defense_kwargs
        self.schedule = schedule
        self.block_len = block_len
        self.C_ref = C_ref
        if len(schedule) * block_len != self.rounds:
            raise ValueError(f"schedule covers {len(schedule) * block_len} rounds, expected {self.rounds}")

    def setup(self) -> None:
        super().setup()  # seeds, model, honest loaders (generator seed + w), test loader
        # Byzantine workers get loaders too (same seeding scheme), so they can carry data.
        for w in range(self.n - self.f, self.n):
            gen = torch.Generator().manual_seed(self.seed + w)
            self._worker_loaders.append(
                DataLoader(self.train_datasets[w], batch_size=self.batch_size, shuffle=True, generator=gen)
            )
        dev = self.device
        self._perm_gen = torch.Generator(device=dev).manual_seed(self.seed + PERM_STREAM)
        self._noise_gen = torch.Generator(device=dev).manual_seed(self.seed + NOISE_STREAM)

    def _grad_and_loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same computation as the library's ``_train_one_worker``, also returning the loss."""
        assert self._model is not None
        self._model.module.train()
        x, y = x.to(self.device), y.to(self.device)
        self._model.module.zero_grad()
        loss = self.loss_fn(self._model.module(x), y)
        loss.backward()
        return self._model.relink_gradients().clone(), loss.detach()

    def step(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Simulation not set up. Call setup() first.")
        t = self._current_round
        block = t // self.block_len
        attacking = self.attack_spec.active(block)
        n_honest = self.n - self.f
        spec = self.attack_spec

        grads: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for w in range(self.n):
            x, y = next(iter(self._worker_loaders[w]))  # always drawn: keeps the data stream fixed
            byzantine = w >= n_honest and attacking
            if byzantine and spec.kind == "gradient":
                continue  # filled below from the honest gradients
            if byzantine and spec.kind == "data":
                assert spec.corrupt is not None
                x, y = spec.corrupt(x, y, **self.corrupt_kwargs)
            g, loss = self._grad_and_loss(x, y)
            grads.append(g)
            losses.append(loss)

        if attacking and spec.kind == "gradient":
            assert spec.attack_cls is not None
            byz = spec.attack_cls.generate(torch.stack(grads[:n_honest]), f=self.f, **spec.attack_kwargs)
            grads.extend(byz)

        perm = torch.randperm(self.n, generator=self._perm_gen, device=self.device)
        stacked = torch.stack(grads)[perm]
        x_round = round_signals(stacked, torch.stack(losses), self.C_ref)

        noise = torch.randn(stacked.shape[1], generator=self._noise_gen, device=self.device, dtype=stacked.dtype)
        defense_kwargs = dict(self.defense_kwargs)
        if issubclass(self.defense, ClipNoise):
            defense_kwargs["noise"] = noise
        aggregated = ScheduledAggregator.aggregate(
            stacked, round=t, plain=self.plain, defense=self.defense, schedule=self.schedule,
            block_len=self.block_len, defense_kwargs=defense_kwargs, n=self.n, f=self.f,
        )

        # Same update as the library's step (weight decay and lr schedule are off in this testbed).
        self._model.gradients = aggregated
        with torch.no_grad():
            self._model.parameters.add_(self._model.gradients, alpha=-self._current_lr)
        self._current_round += 1
        return x_round

    @torch.no_grad()
    def accuracy_loss(self, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        assert self._model is not None
        self._model.module.eval()
        logits = self._model.module(x)
        return (logits.argmax(1) == y).float().mean().item(), self.loss_fn(logits, y).item()

    @torch.no_grad()
    def target_rate(self, x: torch.Tensor, target: int) -> float:
        assert self._model is not None
        self._model.module.eval()
        return (self._model.module(x).argmax(1) == target).float().mean().item()
