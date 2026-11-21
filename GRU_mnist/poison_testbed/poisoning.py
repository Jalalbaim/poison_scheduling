"""Attack conditions: data-poisoning corruptions and library gradient attacks, with an optional block window.

A *data* attack makes the f Byzantine workers compute an honest gradient on a
corrupted copy of their micro-batch. A *gradient* attack replaces their gradient
by ``attack.generate(honest_gradients, f=f, ...)`` from the library. Outside the
window the Byzantine workers behave exactly like honest ones.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from krum.primitives.attacks import Attack
from krum.primitives.attacks.alie import ALIEAttack
from krum.primitives.attacks.sign_flip import SignFlipAttack

from .data import apply_patch

Batch = tuple[torch.Tensor, torch.Tensor]


def _n_poisoned(batch_size: int, poison_frac: float) -> int:
    return int(round(poison_frac * batch_size))


def label_flip(x: torch.Tensor, y: torch.Tensor, *, poison_frac: float, num_classes: int = 10, **_: Any) -> Batch:
    """y -> (num_classes - 1 - y) on the first ``poison_frac`` of the micro-batch."""
    k = _n_poisoned(len(y), poison_frac)
    y = y.clone()
    y[:k] = num_classes - 1 - y[:k]
    return x, y


def backdoor(x: torch.Tensor, y: torch.Tensor, *, poison_frac: float, patch_size: int, target: int,
             stamp: Callable[[torch.Tensor], torch.Tensor] | None = None, **_: Any) -> Batch:
    """Stamp the trigger and set label ``target`` on the first ``poison_frac`` of the micro-batch.

    ``stamp`` (xp5) applies the run's trigger; None = the MNIST corner patch of xp1-xp4 (unchanged).
    """
    k = _n_poisoned(len(y), poison_frac)
    x, y = x.clone(), y.clone()
    x[:k] = apply_patch(x[:k], patch_size) if stamp is None else stamp(x[:k])
    y[:k] = target
    return x, y


def targeted_flip(x: torch.Tensor, y: torch.Tensor, *, poison_frac: float, target: int, flip_pool: torch.Tensor,
                  flip_gen: torch.Generator, **_: Any) -> Batch:
    """Targeted label flip: the first ``poison_frac`` of the micro-batch is replaced by training images of the
    source class (drawn from ``flip_pool`` with the attacker's own generator, so the honest data stream is
    unchanged), all labelled ``target``. The attacker controls the content of its micro-batch."""
    k = _n_poisoned(len(y), poison_frac)
    x, y = x.clone(), y.clone()
    idx = torch.randint(len(flip_pool), (k,), generator=flip_gen)
    x[:k] = flip_pool[idx].to(x.device)
    y[:k] = target
    return x, y


@dataclass(frozen=True)
class AttackSpec:
    name: str  # value of the `attack` / `u_type` columns
    kind: str  # "none" | "data" | "gradient"
    corrupt: Callable[..., Batch] | None = None
    attack_cls: type[Attack] | None = None
    attack_kwargs: dict[str, Any] = field(default_factory=dict)
    window: tuple[int, int] | None = None  # inclusive block range; None = always
    has_trigger: bool = False  # y_atk = backdoor success
    outcome: str = ""  # xp5: "backdoor" | "flip" | "none" (no targeted harm: y_atk masked); "" = has_trigger rule
    burst: tuple[int, int] | None = None  # (period, on): inside the window, active `on` blocks of every `period`

    def active(self, block: int) -> bool:
        if self.kind == "none":
            return False
        if self.window is not None and not self.window[0] <= block <= self.window[1]:
            return False
        if self.burst is not None:
            period, on = self.burst
            start = self.window[0] if self.window is not None else 0
            return (block - start) % period < on
        return True


def make_attack(name: str, *, sign_flip_scale: float = 1.0, alie_z: float | str = "max",
                window: tuple[int, int] = (15, 30), burst: tuple[int, int] | None = None) -> AttackSpec:
    """Build the named condition. Hyperparameters of the corruptions live in the run config.

    ``backdoor_burst``: the backdoor, active inside ``window`` only during bursts (``burst`` = (period, on)).
    """
    if name == "backdoor_burst":
        if burst is None:
            raise ValueError("backdoor_burst needs burst=(period, on)")
        return AttackSpec("backdoor_burst", "data", corrupt=backdoor, has_trigger=True, window=window, burst=burst)
    specs = {
        "none": AttackSpec("none", "none"),
        "label_flip": AttackSpec("label_flip", "data", corrupt=label_flip),
        "backdoor": AttackSpec("backdoor", "data", corrupt=backdoor, has_trigger=True),
        "backdoor_win": AttackSpec("backdoor_win", "data", corrupt=backdoor, has_trigger=True, window=window),
        "alie": AttackSpec("alie", "gradient", attack_cls=ALIEAttack, attack_kwargs={"z": alie_z}),
        # xp5: windowed targeted label flip and windowed ALIE
        "flip_win": AttackSpec("flip_win", "data", corrupt=targeted_flip, window=window, outcome="flip"),
        "alie_win": AttackSpec("alie_win", "gradient", attack_cls=ALIEAttack, attack_kwargs={"z": alie_z},
                               window=window, outcome="none"),
        "sign_flip": AttackSpec("sign_flip", "gradient", attack_cls=SignFlipAttack,
                                attack_kwargs={"scale": sign_flip_scale}),
    }
    if name not in specs:
        raise ValueError(f"Unknown attack {name!r}; expected one of {sorted(specs)}")
    return specs[name]
