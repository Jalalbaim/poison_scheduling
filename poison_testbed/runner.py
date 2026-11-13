"""One run = one (attack, defense, seed, schedule): train, log one row per block, write CSV + JSON."""

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

import krum
from krum.primitives.aggregators import Aggregator
from krum.primitives.aggregators.average import Average
from krum.primitives.aggregators.bulyan import Bulyan
from krum.primitives.aggregators.krum import Krum
from krum.primitives.aggregators.median import Median
from krum.primitives.aggregators.multikrum import MultiKrum
from krum.primitives.aggregators.trimmed_mean import TrimmedMean
from krum.primitives.models.mlp import Krum2017MLPMnist

from .aggregators import ClipNoise
from .data import load_mnist
from .poisoning import make_attack
from .signals import X_COLUMNS
from .simulation import PoisoningSimulation

# Bump when a code change should invalidate cached runs.
TESTBED_VERSION = 1
REPO_DIR = Path(__file__).resolve().parent.parent / "krum"

DEFENSES: dict[str, type[Aggregator]] = {
    "Average": Average,
    "ClipNoise": ClipNoise,
    "Krum": Krum,
    "MultiKrum": MultiKrum,
    "Median": Median,
    "TrimmedMean": TrimmedMean,
    "Bulyan": Bulyan,
}

COLUMNS = [
    "run_id", "seed", "schedule_seed", "defense", "attack", "n", "f",
    "block", "round_start", "lr", "W",
    *X_COLUMNS, "x_val_acc",
    "y_clean_acc", "y_clean_loss", "y_atk",
    "u_active", "u_type",
]


@dataclass(frozen=True)
class RunConfig:
    attack: str
    defense: str
    seed: int
    C: float
    sigma: float
    schedule: str = "random"  # "random" | "on" | "off"
    schedule_seed: int | None = None  # default: 1000 + seed
    n: int = 16
    f: int = 3
    batch_size: int = 8  # micro-batch size; mini-batch = n * batch_size
    lr: float = 0.1
    block_len: int = 10
    n_blocks: int = 50
    poison_frac: float = 1.0
    patch_size: int = 4
    target: int = 0
    window: tuple[int, int] = (15, 30)
    sign_flip_scale: float = 1.0
    alie_z: float | str = "max"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def sched_seed(self) -> int:
        return 1000 + self.seed if self.schedule_seed is None else self.schedule_seed

    @property
    def run_id(self) -> str:
        tag = "" if self.schedule == "random" else f"__{self.schedule}"
        return f"{self.attack}__{self.defense}{tag}__s{self.seed:03d}"

    def hash(self) -> str:
        payload = json.dumps({**asdict(self), "_v": TESTBED_VERSION}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def make_schedule(cfg: RunConfig) -> list[int]:
    if cfg.schedule == "on":
        return [1] * cfg.n_blocks
    if cfg.schedule == "off":
        return [0] * cfg.n_blocks
    if cfg.schedule == "random":
        gen = torch.Generator().manual_seed(cfg.sched_seed)
        return torch.bernoulli(torch.full((cfg.n_blocks,), 0.5), generator=gen).int().tolist()
    raise ValueError(f"Unknown schedule {cfg.schedule!r}")


def defense_kwargs(cfg: RunConfig) -> dict[str, Any]:
    return {"C": cfg.C, "sigma": cfg.sigma} if cfg.defense == "ClipNoise" else {}


def build(cfg: RunConfig) -> PoisoningSimulation:
    torch.use_deterministic_algorithms(True)
    data = load_mnist(cfg.device, cfg.patch_size, cfg.target)
    spec = make_attack(cfg.attack, sign_flip_scale=cfg.sign_flip_scale, alie_z=cfg.alie_z, window=cfg.window)
    sim = PoisoningSimulation(
        attack_spec=spec,
        corrupt_kwargs={"poison_frac": cfg.poison_frac, "patch_size": cfg.patch_size, "target": cfg.target},
        plain=Average,
        defense=DEFENSES[cfg.defense],
        defense_kwargs=defense_kwargs(cfg),
        schedule=make_schedule(cfg),
        block_len=cfg.block_len,
        C_ref=cfg.C,
        model_cls=Krum2017MLPMnist,
        train_datasets=[data.train] * cfg.n,
        test_set=torch.utils.data.TensorDataset(*data.test),
        n=cfg.n,
        f=cfg.f,
        rounds=cfg.block_len * cfg.n_blocks,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=torch.device(cfg.device),
        seed=cfg.seed,
    )
    sim.setup()
    return sim


def _outcomes(sim: PoisoningSimulation, cfg: RunConfig) -> dict[str, float]:
    data = load_mnist(cfg.device, cfg.patch_size, cfg.target)
    val_acc, _ = sim.accuracy_loss(*data.val)
    acc, loss = sim.accuracy_loss(*data.test)
    y_atk = sim.target_rate(data.triggered[0], cfg.target) if sim.attack_spec.has_trigger else math.nan
    return {"x_val_acc": val_acc, "y_clean_acc": acc, "y_clean_loss": loss, "y_atk": y_atk}


def simulate(cfg: RunConfig) -> pd.DataFrame:
    """Train one run and return its block table (no file I/O)."""
    sim = build(cfg)
    base = {"run_id": cfg.run_id, "seed": cfg.seed, "schedule_seed": cfg.sched_seed, "defense": cfg.defense,
            "attack": cfg.attack, "n": cfg.n, "f": cfg.f}
    rows = [{**base, "block": -1, "round_start": pd.NA, "lr": sim._current_lr, "W": pd.NA,
             **{c: math.nan for c in X_COLUMNS}, **_outcomes(sim, cfg), "u_active": 0, "u_type": "none"}]
    for b in range(cfg.n_blocks):
        lr = sim._current_lr
        per_round = [sim.step() for _ in range(cfg.block_len)]
        x_block = {c: sum(r[c] for r in per_round) / len(per_round) for c in X_COLUMNS}
        active = sim.attack_spec.active(b)
        rows.append({**base, "block": b, "round_start": b * cfg.block_len, "lr": lr, "W": sim.schedule[b],
                     **x_block, **_outcomes(sim, cfg),
                     "u_active": int(active), "u_type": cfg.attack if active else "none"})
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype({"round_start": "Int64", "W": "Int64"})


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def run(cfg: RunConfig, out_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run (or load if already done with the same config) and write runs/<run_id>/{blocks.csv,config.json}."""
    out = out_root / cfg.run_id
    csv, js = out / "blocks.csv", out / "config.json"
    if csv.exists() and js.exists():
        meta = json.loads(js.read_text())
        if meta.get("config_hash") == cfg.hash():
            return pd.read_csv(csv, dtype={"round_start": "Int64", "W": "Int64"}), meta
    started = time.time()
    df = simulate(cfg)
    wall = time.time() - started
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": cfg.run_id,
        "config_hash": cfg.hash(),
        "config": asdict(cfg),
        "schedule": make_schedule(cfg),
        "testbed_version": TESTBED_VERSION,
        "krum_version": krum.__version__,
        "krum_git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device,
        "wall_clock_s": round(wall, 2),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    df.to_csv(csv, index=False)
    js.write_text(json.dumps(meta, indent=2, default=str))
    return df, meta
