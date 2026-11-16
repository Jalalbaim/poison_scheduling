"""How long can Krum wait after a backdoor starts and still erase it? (sets xp3's probe periods)

    python measure_erasure.py            # -> results/pilot_xp3/erasure.csv + erasure.png

Windowed backdoor active from round START to the end of training, at poison rates 1/8, 2/8 and 8/8 per poisoned
micro-batch. Krum is switched on k rounds after the attack starts and kept on (block_len = 1: per-round control).
Backdoor success is measured every EVAL_EVERY rounds. Seeds 100-101; xp2/xp3 setup otherwise.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/pilot_xp3"
START, ROUNDS, EVAL_EVERY = 150, 500, 5
RATES = {"1/8": 0.125, "2/8": 0.25, "8/8": 1.0}
DELAYS = [0, 2, 5, 10, 20, 40, 60, 90, 120, None]  # rounds after the attack starts; None = never defended
SEEDS = [100, 101]


def run(rate: str, delay: int | None, seed: int) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    import poison_testbed  # noqa: F401
    from poison_testbed.data import load_mnist
    from poison_testbed.runner import RunConfig, build

    cfg = RunConfig(attack="backdoor_win", defense="Krum", seed=seed, C=3.0552712798118593, sigma=0.01,
                    schedule="off", block_len=1, n_blocks=ROUNDS, window=(START, ROUNDS - 1),
                    poison_frac=RATES[rate])
    sim = build(cfg)
    if delay is not None:
        for t in range(START + delay, ROUNDS):
            sim.schedule[t] = 1
    data = load_mnist(cfg.device, cfg.patch_size, cfg.target)
    rows = []
    for t in range(ROUNDS):
        sim.step()
        since = t + 1 - START  # poisoned rounds so far (after this step)
        if since >= 0 and (since % EVAL_EVERY == 0 or (delay is not None and since == delay)):
            rows.append({"rate": rate, "delay": -1 if delay is None else delay, "seed": seed, "since_start": since,
                         "y_atk": sim.target_rate(data.triggered[0], cfg.target),
                         "y_clean_acc": sim.accuracy_loss(*data.test)[0]})
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, d, s) for r in RATES for d in DELAYS for s in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(run, *j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            print(f"[{k}/{len(jobs)}]", flush=True)
    df = pd.DataFrame(rows).sort_values(["rate", "delay", "seed", "since_start"])
    df.to_csv(OUT / "erasure.csv", index=False)

    # Summary: backdoor when Krum switches on, its peak after that, and at the end of training (350 rounds later).
    summ = []
    for (rate, delay), g in df.groupby(["rate", "delay"]):
        per_seed = []
        for _, s in g.groupby("seed"):
            at_on = s[s.since_start == max(delay, 0)].y_atk.iloc[0] if delay >= 0 else np.nan
            per_seed.append((at_on, s.y_atk.iloc[-1]))
        a = np.array(per_seed, float)
        summ.append({"poison rate": rate, "Krum on after (rounds)": "never" if delay < 0 else delay,
                     "backdoor when Krum switches on": np.nanmean(a[:, 0]), "backdoor at end": a[:, 1].mean(),
                     "erased (end < 0.1, both seeds)": bool((a[:, 1] < 0.1).all())})
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "erasure_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    from analyze import INK, INK2  # plot style
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    cmap = plt.get_cmap("Blues")
    for ax, rate in zip(axes, RATES):
        g = df[(df.rate == rate)]
        for i, d in enumerate([x for x in DELAYS if x is not None] + [None]):
            dd = -1 if d is None else d
            s = g[g.delay == dd].groupby("since_start").y_atk.mean()
            color = INK2 if d is None else cmap(0.35 + 0.6 * i / (len(DELAYS) - 1))
            ax.plot(s.index, s.values, color=color, linewidth=1.5, linestyle="--" if d is None else "-",
                    label="never" if d is None else f"Krum on after {d}")
        ax.set_title(f"poison rate {rate}", color=INK, loc="left")
        ax.set_xlabel("rounds since the attack started")
        ax.set_xlim(0, 200)
    axes[0].set_ylabel("backdoor success (mean of 2 seeds)")
    axes[-1].legend(fontsize=7, ncol=2, loc="center right")
    fig.suptitle("How long can Krum wait after a backdoor starts and still erase it?", color=INK, fontsize=10, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "erasure.png", dpi=130)


if __name__ == "__main__":
    main()
