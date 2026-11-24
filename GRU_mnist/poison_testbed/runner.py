"""One run = one (attack, defense, seed, schedule): train, log one row per block, write CSV + JSON."""

import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
from .alarm import alarm_policy, is_alarm_schedule
from .data import load_data, load_mnist, make_trigger
from .poisoning import make_attack
from .signals import X2_COLUMNS, X_COLUMNS
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
    "run_id", "seed", "schedule_seed", "defense", "attack", "poison_frac", "n", "f",
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
    # "random" | "on" | "off" | "policy" | "cf_<on|off|ontil20|offtil20|alt>_p<prefix>"
    # | "sticky_p<switch prob>" | "react_d<delay>_e<extra>" (blocks, relative to `window`)
    # | "alarm_t<tau>_d<delay>_h<hold>" (xp6: online, reacts to the observable alarm; see alarm.py)
    schedule: str = "random"
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
    burst: tuple[int, int] | None = None  # backdoor_burst: (period, on) blocks inside the window
    # Measure backdoor success (y_atk) even without a trigger attack, e.g. in no-attack runs (xp3). It then
    # reads as the base rate (~1%), which teaches a model that without an attack the backdoor stays at 0.
    always_measure_backdoor: bool = False
    # Signal set: "v1" = the 7 X columns of xp1-xp3; "v2" adds the X2 columns (tail, scale-free, memory; xp4).
    # v2 changes no computation of the run itself: v1 columns, outcomes and training are identical.
    signals: str = "v1"
    # xp5 (defaults = xp1-xp4 behaviour): dataset, backdoor trigger, source class of the targeted label flip
    # (images of flip_source relabelled `target`), and extended logging (per-round signal values r0/r1 and both
    # targeted outcomes y_bd / y_flip in every run).
    dataset: str = "mnist"
    trigger: str = "corner"
    flip_source: int = 1
    extended_log: bool = False
    # xp7: non-iid workers. None = every worker samples the full training set (iid, xp1-xp6). A float = Dirichlet
    # concentration: each class's training images are split across the n workers with shares ~ Dir(alpha), fixed
    # per seed (smaller alpha = more skewed workers; Krum then discards honest but unusual gradients).
    noniid_alpha: float | None = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def sched_seed(self) -> int:
        return 1000 + self.seed if self.schedule_seed is None else self.schedule_seed

    @property
    def run_id(self) -> str:
        tag = "" if self.schedule == "random" else f"__{self.schedule}"
        return f"{self.attack}__{self.defense}{tag}__s{self.seed:03d}"

    def hash(self) -> str:
        fields = asdict(self)
        # Fields added after xp1/xp2: left out while at their default, so every earlier config's hash is unchanged.
        for name, default in LATER_FIELDS.items():
            if fields[name] == default:
                del fields[name]
        payload = json.dumps({**fields, "_v": TESTBED_VERSION}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


LATER_FIELDS = {"burst": None, "always_measure_backdoor": False, "signals": "v1", "dataset": "mnist",
                "trigger": "corner", "flip_source": 1, "extended_log": False,
                "noniid_alpha": None}


def is_legacy_data(cfg: "RunConfig") -> bool:
    """MNIST with the corner patch: the xp1-xp4 data path (load_mnist, apply_patch), kept bit-identical."""
    return cfg.dataset == "mnist" and cfg.trigger == "corner"


def load_run_data(cfg: "RunConfig"):
    if is_legacy_data(cfg) and not cfg.extended_log:
        return load_mnist(cfg.device, cfg.patch_size, cfg.target)
    return load_data(cfg.dataset, cfg.device, cfg.trigger, cfg.patch_size, cfg.target, cfg.flip_source)

CF_SCHEDULE = re.compile(r"cf_(on|off|ontil20|offtil20|alt)_p(\d+)")
PROBE_SCHEDULE = re.compile(r"probe_P(\d+)_L(\d+)_o(\d+)")
PROBEHOLD_SCHEDULE = re.compile(r"probehold_P(\d+)_L(\d+)_o(\d+)_d(-?\d+)_e(\d+)")
FLIP_SUFFIX = re.compile(r"(.+)\+flip_s(\d+)_l(\d+)")
STICKY_SCHEDULE = re.compile(r"sticky_p([0-9.]+)")
REACT_SCHEDULE = re.compile(r"react_d(-?\d+)_e(\d+)")


def counterfactual_schedule(name: str, prefix: int, n_blocks: int) -> list[int]:
    """Defended prefix of ``prefix`` blocks, then a candidate continuation.

    on: always on; off: always off; ontil20: on through block 19 then off;
    offtil20: off through block 19 then on; alt: alternate, starting off.
    """
    rest = range(prefix, n_blocks)
    cont = {
        "on": [1 for _ in rest],
        "off": [0 for _ in rest],
        "ontil20": [int(b < 20) for b in rest],
        "offtil20": [int(b >= 20) for b in rest],
        "alt": [int((b - prefix) % 2 == 1) for b in rest],
    }[name]
    return [1] * prefix + cont


def probe_schedule(period: int, length: int, offset: int, n_blocks: int) -> list[int]:
    """Short defenses to look: defend `length` blocks out of every `period`, starting at `offset`."""
    return [int((b - offset) % period < length) if b >= offset else 0 for b in range(n_blocks)]


def make_schedule(cfg: RunConfig) -> list[int]:
    if m := FLIP_SUFFIX.fullmatch(cfg.schedule):
        # Paired twin: the base schedule with W flipped on blocks [s, s + l). Same seed, so the pair gives the
        # true effect of that short change.
        if is_alarm_schedule(m.group(1)):
            raise ValueError("twins of alarm schedules are not defined (their W is decided online)")
        base = make_schedule(replace(cfg, schedule=m.group(1)))
        s, ln = int(m.group(2)), int(m.group(3))
        return [1 - w if s <= b < s + ln else w for b, w in enumerate(base)]
    if m := PROBE_SCHEDULE.fullmatch(cfg.schedule):
        return probe_schedule(int(m.group(1)), int(m.group(2)), int(m.group(3)), cfg.n_blocks)
    if m := PROBEHOLD_SCHEDULE.fullmatch(cfg.schedule):
        # Probe; from window start + d hold the defense on through window end + e; then probe again. Follows
        # cfg.window whether or not an attack happens (phantom window in no-attack runs).
        period, length, offset, d, e = (int(g) for g in m.groups())
        out = probe_schedule(period, length, offset, cfg.n_blocks)
        on, off = cfg.window[0] + d, cfg.window[1] + e
        return [1 if on <= b <= off else w for b, w in enumerate(out)]
    if cfg.schedule == "on":
        return [1] * cfg.n_blocks
    if cfg.schedule == "off":
        return [0] * cfg.n_blocks
    if cfg.schedule == "random":
        gen = torch.Generator().manual_seed(cfg.sched_seed)
        return torch.bernoulli(torch.full((cfg.n_blocks,), 0.5), generator=gen).int().tolist()
    if cfg.schedule == "policy" or is_alarm_schedule(cfg.schedule):
        return [0] * cfg.n_blocks  # placeholder: each entry is set online by the policy before its block runs
    if cfg.schedule == "oracle":
        # Defend exactly the blocks where the attack is active (knows the attack: an upper reference only).
        spec = make_attack(cfg.attack, window=cfg.window, burst=cfg.burst)
        return [int(spec.active(b)) for b in range(cfg.n_blocks)]
    if m := STICKY_SCHEDULE.fullmatch(cfg.schedule):
        # Two-state Markov chain: random initial state, switch with probability p before each block.
        gen = torch.Generator().manual_seed(cfg.sched_seed)
        p = float(m.group(1))
        state = int(torch.rand(1, generator=gen).item() < 0.5)
        out = []
        for _ in range(cfg.n_blocks):
            out.append(state)
            if torch.rand(1, generator=gen).item() < p:
                state = 1 - state
        return out
    if m := REACT_SCHEDULE.fullmatch(cfg.schedule):
        # Defend from window start + delay through window end + extra. Follows cfg.window whether or not an
        # attack happens (phantom window for no-attack runs), so W never reveals the attack by itself.
        delay, extra = int(m.group(1)), int(m.group(2))
        on, off = cfg.window[0] + delay, cfg.window[1] + extra
        return [int(on <= b <= off) for b in range(cfg.n_blocks)]
    if m := CF_SCHEDULE.fullmatch(cfg.schedule):
        return counterfactual_schedule(m.group(1), int(m.group(2)), cfg.n_blocks)
    raise ValueError(f"Unknown schedule {cfg.schedule!r}")


def worker_datasets(train: torch.utils.data.TensorDataset, cfg: RunConfig) -> list:
    """One training set per worker: the full set for every worker (iid), or a Dirichlet split by class (xp7)."""
    if cfg.noniid_alpha is None:
        return [train] * cfg.n
    x, y = train.tensors
    gen = torch.Generator().manual_seed(cfg.seed + NONIID_STREAM)
    conc = torch.full((cfg.n,), float(cfg.noniid_alpha))
    parts: list[list[torch.Tensor]] = [[] for _ in range(cfg.n)]
    for c in range(int(y.max()) + 1):
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=gen)]
        share = _dirichlet(conc, gen)
        cuts = (share.cumsum(0) * len(idx)).round().long().tolist()
        for w, (a, b) in enumerate(zip([0] + cuts[:-1], cuts)):
            parts[w].append(idx[a:b])
    out = []
    for w in range(cfg.n):
        idx = torch.cat(parts[w])
        if len(idx) < cfg.batch_size:
            raise ValueError(f"worker {w} has {len(idx)} images (< batch {cfg.batch_size}); raise noniid_alpha")
        out.append(torch.utils.data.TensorDataset(x[idx], y[idx]))
    return out


def _dirichlet(conc: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Dirichlet draw from a seeded generator (torch.distributions has no generator argument)."""
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    return torch.from_numpy(rng.dirichlet(conc.numpy())).float()


NONIID_STREAM = 50_000


def defense_kwargs(cfg: RunConfig) -> dict[str, Any]:
    return {"C": cfg.C, "sigma": cfg.sigma} if cfg.defense == "ClipNoise" else {}


def build(cfg: RunConfig) -> PoisoningSimulation:
    torch.use_deterministic_algorithms(True)
    data = load_run_data(cfg)
    spec = make_attack(cfg.attack, sign_flip_scale=cfg.sign_flip_scale, alie_z=cfg.alie_z, window=cfg.window,
                       burst=cfg.burst)
    sim = PoisoningSimulation(
        attack_spec=spec,
        corrupt_kwargs={"poison_frac": cfg.poison_frac, "patch_size": cfg.patch_size, "target": cfg.target,
                        **({"flip_pool": data.flip_pool, "flip_gen": torch.Generator().manual_seed(cfg.seed + 30_000)}
                           if cfg.attack == "flip_win" else {}),
                        **({} if is_legacy_data(cfg) else {"stamp": make_trigger(cfg.dataset, cfg.trigger, cfg.patch_size)})},
        plain=Average,
        defense=DEFENSES[cfg.defense],
        defense_kwargs=defense_kwargs(cfg),
        schedule=make_schedule(cfg),
        block_len=cfg.block_len,
        C_ref=cfg.C,
        signals_v2=cfg.signals == "v2",
        model_cls=Krum2017MLPMnist,
        train_datasets=worker_datasets(data.train, cfg),
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
    data = load_run_data(cfg)
    val_acc, _ = sim.accuracy_loss(*data.val)
    acc, loss = sim.accuracy_loss(*data.test)
    spec = sim.attack_spec
    out = {"x_val_acc": val_acc, "y_clean_acc": acc, "y_clean_loss": loss}
    if spec.outcome == "flip":  # targeted label flip: share of flip_source test images predicted as the target
        out["y_atk"] = sim.target_rate(data.flip_source[0], cfg.target)
    elif spec.outcome == "none":  # no targeted harm (ALIE): masked; its harm is in y_clean_acc
        out["y_atk"] = math.nan
    else:  # xp1-xp4 rule: backdoor success when the attack has a trigger (or when asked, e.g. no-attack runs)
        measure = spec.has_trigger or cfg.always_measure_backdoor
        out["y_atk"] = sim.target_rate(data.triggered[0], cfg.target) if measure else math.nan
    if cfg.extended_log:  # both targeted outcomes in every run, whatever the attack
        out["y_bd"] = sim.target_rate(data.triggered[0], cfg.target)
        out["y_flip"] = sim.target_rate(data.flip_source[0], cfg.target)
    return out


HIDDEN_FROM_POLICY = ("y_", "u_", "attack")


def observable(row: dict[str, Any]) -> dict[str, Any]:
    """What a defender can see: drops outcomes (y_*), attack labels (u_*) and the attack name."""
    return {k: v for k, v in row.items() if not k.startswith(HIDDEN_FROM_POLICY)}


Policy = Callable[[list[dict[str, Any]], int], tuple[int, dict[str, Any]]]


def simulate(cfg: RunConfig, policy: Policy | None = None) -> pd.DataFrame:
    """Train one run and return its block table (no file I/O).

    ``policy(history, b) -> (W_b, extra)``, if given, decides W for block b before it runs. ``history`` holds the
    observable part of rows -1..b-1 only (see ``observable``). ``extra`` is logged on block b's row.
    """
    if policy is None and is_alarm_schedule(cfg.schedule):  # xp6: W decided online from the observable alarm
        policy = alarm_policy(cfg.schedule, cfg.sched_seed)
    sim = build(cfg)
    base = {"run_id": cfg.run_id, "seed": cfg.seed, "schedule_seed": cfg.sched_seed, "defense": cfg.defense,
            "attack": cfg.attack, "poison_frac": cfg.poison_frac, "n": cfg.n, "f": cfg.f}
    if cfg.extended_log:  # attack configuration as evaluation labels (u_: hidden from any policy)
        base.update({"u_trigger": cfg.trigger, "u_target": cfg.target, "u_flip_source": cfg.flip_source,
                     "u_burst": "" if cfg.burst is None else f"{cfg.burst[0]}_{cfg.burst[1]}"})
    rows = [{**base, "block": -1, "round_start": pd.NA, "lr": sim._current_lr, "W": pd.NA,
             **{c: math.nan for c in X_COLUMNS}, **_outcomes(sim, cfg), "u_active": 0, "u_type": "none"}]
    extra_cols: list[str] = []
    for b in range(cfg.n_blocks):
        extra: dict[str, Any] = {}
        if policy is not None:
            w, extra = policy([observable(r) for r in rows], b)
            sim.schedule[b] = int(w)
            extra_cols += [k for k in extra if k not in extra_cols]
        lr = sim._current_lr
        per_round = [sim.step() for _ in range(cfg.block_len)]
        x_block = {c: sum(r[c] for r in per_round) / len(per_round) for c in X_COLUMNS}
        if cfg.signals == "v2":  # block mean and block max of every v2 signal
            for c in per_round[0]:
                if c.startswith("x2_"):
                    x_block[f"{c}_mean"] = sum(r[c] for r in per_round) / len(per_round)
                    x_block[f"{c}_max"] = max(r[c] for r in per_round)
        if cfg.extended_log:  # the value of every signal in each round of the block
            for k, r in enumerate(per_round):
                for c, v in r.items():
                    x_block[f"{c}_r{k}"] = v
        active = sim.attack_spec.active(b)
        rows.append({**base, "block": b, "round_start": b * cfg.block_len, "lr": lr, "W": sim.schedule[b],
                     **x_block, **_outcomes(sim, cfg),
                     "u_active": int(active), "u_type": cfg.attack if active else "none", **extra})
    x2 = list(X2_COLUMNS) if cfg.signals == "v2" else []
    if cfg.extended_log:
        x2 += ["y_bd", "y_flip", "u_trigger", "u_target", "u_flip_source", "u_burst"]
        x2 += [c for c in rows[-1] if c[-3:-1] == "_r" and c.startswith("x")]
    df = pd.DataFrame(rows, columns=COLUMNS + x2 + extra_cols)
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
        "schedule": ([int(w) for w in df.W.iloc[1:]] if is_alarm_schedule(cfg.schedule) else make_schedule(cfg)),
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
