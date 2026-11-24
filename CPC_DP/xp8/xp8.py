"""xp8: paired branches for learning WHEN to switch Krum on (decision at step t uses step t's own gradients).

    ../.venv/Scripts/python xp8/xp8.py --list                    # plan tables and cost, run nothing
    ../.venv/Scripts/python xp8/xp8.py --part 1 --jobs 4         # Kaggle session 1 -> results/xp8/part1/
    ../.venv/Scripts/python xp8/xp8.py --part 2 --jobs 4         # Kaggle session 2 -> results/xp8/part2/
    ../.venv/Scripts/python xp8/xp8.py --part 1 --smoke          # 6 setups of 40 steps (pipeline check)

Time: 1 block = 1 SGD step (block_len 1), T = 500 steps, as many SGD steps as xp1-xp7.

Order inside a step t (PoisoningSimulation.gather / apply):
  1. the 16 workers send their gradients; the signals of row t are computed from them (before any aggregation);
  2. the aggregator of step t is chosen (W_t = 1: Krum, 0: plain average): a controller may use row t's signals;
  3. aggregation and update; the outcomes of row t are measured on the updated model.
  x_val_acc_pre = validation accuracy of the model BEFORE step t (= the one measured after step t - 1): usable at t.
  Row -1 = the initial model (outcomes only).

Design (264 setups = 34 attack configurations x 6 seeds + 60 no-attack setups; CONFIGS, N_NONE):
- Attacks: backdoor corner and patch (poisoned share 2/8, 4/8, 8/8), targeted flip (4/8, 8/8), ALIE (z max / half
  max), SignFlip (scale 2 / 4.6); durations short (30 steps), long (120), persistent (to the end; not SignFlip);
  random onset. Attackers per setup: 1, 3 or 5 (ALIE: 3 or 5, as 1 attacker gives z_max = 0; SignFlip: 1 or 3).
  Krum always assumes f = 3. No attack: 60 setups.
- Each setup is run twice, Krum always off and always on (the two "trunks", same seed = same data and noise).
- Branches: at a few steps t, the trunk's full state is saved; the run is resumed from it with the defense
  flipped for K = 40 steps, then back to the trunk's setting until the end. Trunk vs branch = the true effect of
  switching on (off trunk) or off (on trunk) at t, at that exact history. Branch points: attack onset, onset + 10,
  onset + 30 and one quiet step (before the attack or well after it); no attack: 4 random steps.
- Splits by setup: per configuration, seed 0 -> test, seed 1 -> val, seeds 2-5 -> train. Branches of val / test
  setups are evaluation data, as the xp7 twins were.
- Two parts (two Kaggle sessions) of equal cost; every configuration has setups in both.

Output (per part): jobs/<job>.parquet (one per setup x trunk: the trunk rows, then each branch from its step t),
all_blocks.parquet (all jobs), manifest.csv (setups), jobs.csv (status, wall time, errors).
A job that crashes is recorded in jobs.csv and does not stop the others. Re-running a part skips finished jobs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "xp8" else HERE  # local: CPC_DP/xp8/; Kaggle: files flat in $WORK
sys.path.insert(0, str(ROOT))
if (ROOT.parent / "GRU_mnist" / "run_experiment.py").exists():  # local layout; on Kaggle the files sit side by side
    sys.path.insert(0, str(ROOT.parent / "GRU_mnist"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import run_experiment as R  # noqa: E402
from poison_testbed.runner import RunConfig, _outcomes, build  # noqa: E402
from poison_testbed.signals import X2_SIGNALS, X_COLUMNS  # noqa: E402

T, K = 500, 40
SEED0 = 90_000
N_SEEDS, N_NONE = 6, 60
SHORT, LONG = 30, 120
DURATIONS = ("short", "long", "persistent")
ATTACKERS = {"alie": (3, 5), "signflip": (1, 3)}  # others: (1, 3, 5)
RATE = 100.0  # run-equivalents (500 steps) per hour on Kaggle (xp7_deploy: 130 s per run at 4 jobs, + 10%)
SIGNALS = list(X_COLUMNS) + [f"x2_{s}" for s in X2_SIGNALS]
OUTCOMES = ["y_clean_acc", "y_clean_loss", "y_atk", "y_bd", "y_flip"]
COLUMNS = (["job", "setup", "trunk", "traj", "t_branch", "step", "W"] + SIGNALS + ["x_val_acc_pre"] + OUTCOMES
           + ["u_active"])


def _configs() -> list[dict]:
    """The 35 attack configurations (family, label, RunConfig fields)."""
    out = []
    for fam, trigger in (("corner", "corner"), ("patch", "patch_r3_c5")):
        for frac in (0.25, 0.5, 1.0):
            for dur in DURATIONS:
                out.append({"family": fam, "label": f"{fam} {int(frac * 8)}/8 {dur}", "duration": dur,
                            "attack": "backdoor_win", "trigger": trigger, "poison_frac": frac})
    for frac in (0.5, 1.0):
        for dur in DURATIONS:
            out.append({"family": "flip", "label": f"flip {int(frac * 8)}/8 {dur}", "duration": dur,
                        "attack": "flip_win", "poison_frac": frac})
    for z in ("max", "half"):
        for dur in DURATIONS:
            out.append({"family": "alie", "label": f"ALIE z {z} {dur}", "duration": dur, "attack": "alie_win",
                        "alie": z})
    for scale in (2.0, 4.6):
        for dur in ("short", "long"):
            out.append({"family": "signflip", "label": f"SignFlip {scale:g} {dur}", "duration": dur,
                        "attack": "sign_flip_win", "sign_flip_scale": scale})
    return out


CONFIGS = _configs()


def alie_z(kind: str, n_attackers: int, n: int = 16) -> float | str:
    """'max' (the library's largest factor) or half of it, as a number."""
    if kind == "max":
        return "max"
    from krum.primitives.attacks.alie import ALIEAttack
    return 0.5 * float(ALIEAttack.max_z(torch.zeros(n - n_attackers, 1), n_attackers))


def _window(rng: np.random.Generator, duration: str) -> tuple[int, int]:
    if duration == "persistent":
        return int(rng.integers(2, 21)), T - 1
    onset = int(rng.integers(4, 41)) if rng.random() < 0.3 else int(rng.integers(42, 301))
    return onset, min(onset + (SHORT if duration == "short" else LONG) - 1, T - 1)


def _branch_points(rng: np.random.Generator, family: str, window: tuple[int, int]) -> list[int]:
    if family == "none":
        return sorted(int(v) for v in rng.choice(np.arange(10, T - 50, 10), 4, replace=False))
    onset, end = window
    pts = [t for t in (onset, onset + 10, onset + 30) if t < T - 1]
    if onset >= 50:
        pts.append(int(rng.integers(10, onset - 19)))
    elif end + 20 <= T - 60:
        pts.append(int(rng.integers(end + 20, T - 59)))
    return sorted(set(pts))


def setups() -> pd.DataFrame:
    """One row per setup: attack, seed, window, attackers, branch points, split, part."""
    rows = []
    groups = [(None, N_NONE)] + [(c, N_SEEDS) for c in CONFIGS]
    for ci, (c, n) in enumerate(groups):
        for k in range(n):
            i = len(rows)
            rng = np.random.default_rng(190_000 + i)
            fam = "none" if c is None else c["family"]
            dur = "none" if c is None else c["duration"]
            window = (0, 0) if c is None else _window(rng, dur)
            n_att = int(rng.choice(ATTACKERS.get(fam, (1, 3, 5))))
            row = {"setup": i, "seed": SEED0 + i, "family": fam, "config": "no attack" if c is None else c["label"],
                   "duration": dur, "persistent": dur == "persistent", "attack": "none" if c is None else c["attack"],
                   "trigger": "corner" if c is None else c.get("trigger", "corner"),
                   "poison_frac": 1.0 if c is None else c.get("poison_frac", 1.0),
                   "alie_z": "max" if c is None else (alie_z(c["alie"], n_att) if "alie" in c else "max"),
                   "sign_flip_scale": 1.0 if c is None else c.get("sign_flip_scale", 1.0),
                   "n_attackers": n_att, "target": int(rng.choice(R.XP5_TARGETS)),
                   "flip_source": int(rng.choice(R.XP5_FLIP_SOURCES)),
                   "onset": window[0] if c is not None else -1, "end": window[1] if c is not None else -1}
            row["branch_points"] = " ".join(map(str, _branch_points(rng, fam, window)))
            row["split"] = {0: "test", 1: "val"}.get(k % N_SEEDS, "train")
            row["part"] = 1 + (ci + k) % 2
            rows.append(row)
    return pd.DataFrame(rows)


def smoke_setups() -> pd.DataFrame:
    """One train setup per family, shrunk to 40 steps (K = 8): exercises every attack, trunk and branch path."""
    s = setups()
    s = s[s.split == "train"].groupby("family").head(1).copy()
    s["onset"] = np.where(s.family == "none", -1, 5)
    s["end"] = np.where(s.family == "none", -1, np.where(s.persistent, 39, 20))
    s["branch_points"] = np.where(s.family == "none", "8 16 24 30", "5 10 20 30")
    s["part"] = [1, 2] * 3
    return s.reset_index(drop=True)


def run_cost(bps: list[int], n_steps: int = T) -> float:
    """Run-equivalents of one job: the trunk + each branch from its step to the end."""
    return 1 + sum((n_steps - t) / n_steps for t in bps)


def jobs_of(s: pd.DataFrame, n_steps: int = T) -> pd.DataFrame:
    rows = []
    for r in s.itertuples():
        bps = [int(v) for v in r.branch_points.split()]
        for trunk in (0, 1):
            rows.append({"job": f"s{r.setup:03d}_{'on' if trunk else 'off'}", "setup": r.setup, "trunk": trunk,
                         "part": r.part, "n_branches": len(bps), "cost": run_cost(bps, n_steps)})
    return pd.DataFrame(rows)


def config(r, trunk: int, n_steps: int = T, device: str | None = None) -> RunConfig:
    over = {"block_len": 1, "n_blocks": n_steps, **({"device": device} if device else {})}
    z = r["alie_z"]
    return RunConfig(defense="Krum", seed=int(r["seed"]), schedule="on" if trunk else "off", attack=r["attack"],
                     window=(int(r["onset"]), int(r["end"])) if r["family"] != "none" else (0, 0),
                     trigger=r["trigger"], poison_frac=float(r["poison_frac"]), flip_source=int(r["flip_source"]),
                     target=int(r["target"]), sign_flip_scale=float(r["sign_flip_scale"]),
                     alie_z=z if z == "max" else float(z), n_attackers=int(r["n_attackers"]), **R.xp5_params(over))


# ----------------------------------------------------------------------------- engine
def snapshot(sim, prev_val: float) -> dict:
    """Everything that determines the rest of a run: weights, round, every RNG stream, the signal tracker."""
    fg = sim.corrupt_kwargs.get("flip_gen")
    return {"params": sim._model.parameters.detach().clone(), "round": sim._current_round,
            "loaders": [ld.generator.get_state() for ld in sim._worker_loaders],
            "perm": sim._perm_gen.get_state(), "noise": sim._noise_gen.get_state(),
            "flip": None if fg is None else fg.get_state(), "tracker": copy.deepcopy(sim.tracker),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "schedule": list(sim.schedule), "prev_val": prev_val}


def restore(sim, st: dict) -> float:
    with torch.no_grad():
        sim._model.parameters.copy_(st["params"])
    sim._current_round = st["round"]
    for ld, g in zip(sim._worker_loaders, st["loaders"]):
        ld.generator.set_state(g)
    sim._perm_gen.set_state(st["perm"])
    sim._noise_gen.set_state(st["noise"])
    if st["flip"] is not None:
        sim.corrupt_kwargs["flip_gen"].set_state(st["flip"])
    sim.tracker = copy.deepcopy(st["tracker"])
    torch.set_rng_state(st["cpu_rng"])
    if st["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(st["cuda_rng"])
    sim.schedule[:] = st["schedule"]
    sim._pending = None
    return st["prev_val"]


def step_row(sim, cfg: RunConfig, s: int, w: int, prev_val: float) -> tuple[dict, float]:
    """One SGD step: gather (signals of step s) -> W_s -> apply -> outcomes. Returns the row and the new val acc."""
    x = sim.gather()
    sim.schedule[s] = int(w)
    sim.apply()
    out = _outcomes(sim, cfg)
    row = {"step": s, "W": int(w), **{c: x[c] for c in SIGNALS}, "x_val_acc_pre": prev_val,
           **{k: out.get(k, math.nan) for k in OUTCOMES}, "u_active": int(sim.attack_spec.active(s))}
    return row, out["x_val_acc"]


def simulate_job(cfg: RunConfig, trunk: int, bps: list[int], k: int) -> tuple[pd.DataFrame, dict]:
    """The trunk (W = trunk at every step) and one branch per t in bps (W = 1 - trunk on [t, t + k))."""
    sim = build(cfg)
    n = cfg.n_blocks
    init = _outcomes(sim, cfg)
    rows = [{"traj": "trunk", "t_branch": -1, "step": -1, "W": -1, **{c: math.nan for c in SIGNALS},
             "x_val_acc_pre": math.nan, **{k_: init.get(k_, math.nan) for k_ in OUTCOMES}, "u_active": 0}]
    prev = init["x_val_acc"]
    snaps = {}
    for s in range(n):
        if s in bps:
            snaps[s] = snapshot(sim, prev)
        row, prev = step_row(sim, cfg, s, trunk, prev)
        rows.append({"traj": "trunk", "t_branch": -1, **row})
    trunk_x = {r["step"]: [r[c] for c in SIGNALS] for r in rows if r["step"] in snaps}
    info = {"branches_ok": 0, "restore_mismatch": [], "errors": []}
    for t in bps:
        try:
            prev = restore(sim, snaps.pop(t))
            for s in range(t, n):
                row, prev = step_row(sim, cfg, s, 1 - trunk if s < t + k else trunk, prev)
                if s == t and not np.array_equal(np.array([row[c] for c in SIGNALS]), np.array(trunk_x[t]),
                                                 equal_nan=True):
                    info["restore_mismatch"].append(t)  # step t's signals precede W_t: must equal the trunk's
                rows.append({"traj": "branch", "t_branch": t, **row})
            info["branches_ok"] += 1
        except Exception:  # keep the trunk and the other branches; the error is reported in jobs.csv
            rows = [r for r in rows if r["t_branch"] != t]
            info["errors"].append(f"branch {t}: {traceback.format_exc(limit=3)}")
    return pd.DataFrame(rows), info


def run_job(job: dict, setup: dict, out: str, n_steps: int, k: int) -> dict:
    """Worker entry point (one process per job). Never raises: errors are returned."""
    f = Path(out) / "jobs" / f"{job['job']}.parquet"
    t0 = time.time()
    for attempt in range(3):  # a transient CUDA out-of-memory (GPU busy) gets two more tries, 60 s apart
        res = _run_job_once(job, setup, f, n_steps, k)
        if res["status"] != "failed" or "out of memory" not in res["error"]:
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(60)
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def _run_job_once(job: dict, setup: dict, f: Path, n_steps: int, k: int) -> dict:
    res = {"job": job["job"], "setup": job["setup"], "trunk": job["trunk"], "status": "done", "wall_s": 0.0,
           "branches_ok": 0, "n_branches": job["n_branches"], "restore_mismatch": "", "error": ""}
    try:
        cfg = config(setup, job["trunk"], n_steps)
        bps = [int(v) for v in setup["branch_points"].split()]
        df, info = simulate_job(cfg, job["trunk"], bps, k)
        df.insert(0, "job", job["job"]); df.insert(1, "setup", job["setup"]); df.insert(2, "trunk", job["trunk"])
        df = df[COLUMNS]
        tmp = f.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(f)
        res.update(branches_ok=info["branches_ok"], restore_mismatch=" ".join(map(str, info["restore_mismatch"])),
                   error=" | ".join(info["errors"]))
        if info["errors"] or info["restore_mismatch"]:
            res["status"] = "partial"
    except Exception:
        res.update(status="failed", error=traceback.format_exc(limit=5))
    return res


# ----------------------------------------------------------------------------- driver
def _last_error(err) -> str:
    lines = [ln for ln in str(err or '').splitlines() if 'Error' in ln or 'Exception' in ln]
    return lines[-1].strip() if lines else str(err or '').strip()[-200:]


def describe(s: pd.DataFrame, jobs: pd.DataFrame) -> None:
    j = jobs.merge(s[["setup", "family", "config"]], on="setup")
    fam = j.groupby("family").agg(setups=("setup", "nunique"), jobs=("job", "size"),
                                  branches=("n_branches", "sum"), run_eq=("cost", "sum"))
    fam.loc["total"] = fam.sum()
    print(fam.round(0).astype(int).to_string())
    print("\nsplits (setups):", s.groupby("split").size().to_dict(), " attackers:", s.n_attackers.value_counts()
          .sort_index().to_dict())
    p = j.groupby("part").agg(setups=("setup", "nunique"), jobs=("job", "size"), branches=("n_branches", "sum"),
                              run_eq=("cost", "sum"))
    p["est_hours"] = p.run_eq / RATE
    print("\nper part (Kaggle session):\n" + p.round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", type=int, choices=[1, 2], default=1)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="6 setups (one per family) of 40 steps")
    ap.add_argument("--only", type=int, default=None, help="run only the first N jobs of the part (local checks)")
    ap.add_argument("--budget-hours", type=float, default=11.0,
                    help="stop starting new jobs after this many hours (finished jobs are kept and merged)")
    args = ap.parse_args()
    n_steps, k = (40, 8) if args.smoke else (T, K)
    s = smoke_setups() if args.smoke else setups()
    jobs = jobs_of(s, n_steps)
    describe(s, jobs)
    if args.list:
        return
    out = args.out or ROOT / "results/xp8" / f"part{args.part}{'_smoke' if args.smoke else ''}"
    (out / "jobs").mkdir(parents=True, exist_ok=True)
    s_part = s[s.part == args.part]
    s_part.to_csv(out / "manifest.csv", index=False)
    todo = jobs[jobs.part == args.part].sort_values("cost", ascending=False)  # longest first: better packing
    if args.only:
        todo = todo.head(args.only)
    status_file = out / "jobs.csv"
    old = pd.read_csv(status_file) if status_file.exists() else pd.DataFrame(columns=["job", "status"])
    done_before = set(old[old.status.isin(["done", "partial"])].job) & {p.stem for p in (out / "jobs").glob("*.parquet")}
    results = [r for r in old.to_dict("records") if r["job"] in done_before]
    pending = [j for j in todo.to_dict("records") if j["job"] not in done_before]
    by_setup = {r["setup"]: r for r in s_part.to_dict("records")}
    print(f"\npart {args.part}: {len(todo)} jobs, {len(done_before & set(todo.job))} already done, "
          f"{len(pending)} to run, {args.jobs} parallel, budget {args.budget_hours} h", flush=True)
    t0, stopped = time.time(), False
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        running = set()
        while pending or running:
            while pending and len(running) < 2 * args.jobs and not stopped:
                if time.time() - t0 > args.budget_hours * 3600:
                    stopped = True
                    print(f"== budget of {args.budget_hours} h reached: no new jobs ({len(pending)} left)", flush=True)
                    break
                j = pending.pop(0)
                running.add(pool.submit(run_job, j, by_setup[j["setup"]], str(out), n_steps, k))
            if stopped:
                pending = []
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    r = fut.result()
                except Exception:  # the worker process itself died (e.g. out of memory)
                    r = {"job": "?", "status": "failed", "wall_s": 0.0, "branches_ok": 0, "n_branches": 0,
                         "error": traceback.format_exc(limit=3)}
                results.append(r)
                pd.DataFrame(results).to_csv(status_file, index=False)
                print(f"[{len(results)}/{len(todo)}] {r['job']} {r['status']} {r['wall_s']:.0f}s "
                      f"branches {r['branches_ok']}/{r['n_branches']} (elapsed {(time.time() - t0) / 60:.0f} min)"
                      + (f"  ERROR {_last_error(r['error'])}"
                         if r["status"] != "done" else ""), flush=True)
    st = pd.DataFrame(results, columns=["job", "setup", "trunk", "status", "wall_s", "branches_ok",
                                        "n_branches", "restore_mismatch", "error"])
    files = [out / "jobs" / f"{j}.parquet" for j in st[st.status.isin(["done", "partial"])].job]
    merged = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files
              else pd.DataFrame(columns=COLUMNS))  # nothing finished: still write the file (the zip step expects it)
    merged.to_parquet(out / "all_blocks.parquet", index=False)
    counts = {k: int(v) for k, v in st.status.value_counts().items()}
    print(f"== part {args.part}: {counts}; missing {len(set(todo.job) - set(st.job))}; "
          f"wrote {out / 'all_blocks.parquet'} ({len(files)} jobs)", flush=True)
    (out / "summary.json").write_text(json.dumps({"status": counts, "jobs_planned": len(todo),
                                                  "minutes": round((time.time() - t0) / 60, 1)}, indent=1))


if __name__ == "__main__":
    main()
