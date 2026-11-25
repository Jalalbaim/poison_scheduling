"""xp8 first look: what does defending for K = 40 steps buy, per attack, per moment, per trunk?

gain(t) = U(defended path) - U(undefended path), U = mean over steps t..T-1 of (y_clean_acc - LAMBDA * y_atk).
- off trunk: branch = Krum on for [t, t+K) -> gain = branch - trunk
- on trunk:  branch = Krum off for [t, t+K) -> gain = trunk - branch
y_atk is the targeted success (backdoor / flip); for ALIE, SignFlip and no attack it is 0 (their harm is in clean acc).
Also reported: the final clean acc gap (step T-1) and the attack gap over the 40 flipped steps.
Writes results/xp8/analysis/{gains.parquet, by_config.csv, by_moment.csv}.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "xp8" else HERE
sys.path.insert(0, str(HERE))
import xp8 as X  # noqa: E402

RES = ROOT / "results" / "xp8"
OUT = RES / "analysis"
LAMBDA = 1.0


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs, mans = [], []
    for p in ("part1", "part2"):
        d = pd.read_parquet(RES / p / "all_blocks.parquet",
                            columns=["setup", "trunk", "traj", "t_branch", "step", "W", "y_clean_acc", "y_atk"])
        dfs.append(d[d.step >= 0])
        mans.append(pd.read_csv(RES / p / "manifest.csv"))
    return pd.concat(dfs, ignore_index=True), pd.concat(mans, ignore_index=True)


def moment(r, t: int) -> str:
    if r.family == "none":
        return "no attack"
    if t == r.onset:
        return "onset"
    if t == r.onset + 10:
        return "onset+10"
    if t == r.onset + 30:
        return "onset+30"
    return "during" if r.onset <= t <= r.end else ("before" if t < r.onset else "after")


def gains(df: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    targeted = man.set_index("setup").family.isin(["corner", "patch", "flip"])
    df["atk"] = np.where(targeted.reindex(df.setup).to_numpy(), df.y_atk.fillna(0.0), 0.0)
    df["u"] = df.y_clean_acc - LAMBDA * df.atk
    rows = []
    mi = man.set_index("setup")
    for (setup, trunk), g in df.groupby(["setup", "trunk"], sort=False):
        r = mi.loc[setup]
        tr = g[g.traj == "trunk"].set_index("step")
        for t, br in g[g.traj == "branch"].groupby("t_branch"):
            br = br.set_index("step")
            sign = 1 if trunk == 0 else -1  # +: defended minus undefended
            win = slice(t, t + X.K - 1)
            rows.append({
                "setup": setup, "trunk": trunk, "t": int(t), "moment": moment(r, int(t)),
                "family": r.family, "config": r.config, "duration": r.duration, "n_attackers": r.n_attackers,
                "split": r.split,
                "gain": sign * (br.u.mean() - tr.loc[t:, "u"].mean()),
                "gain_final_acc": sign * (br.y_clean_acc.iloc[-1] - tr.y_clean_acc.iloc[-1]),
                "gain_atk_window": -sign * (br.loc[win, "atk"].mean() - tr.loc[win, "atk"].mean()),
                "trunk_final_acc": tr.y_clean_acc.iloc[-1],
            })
    return pd.DataFrame(rows)


def trunk_table(df: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    """Always-on vs always-off per setup (whole run)."""
    tr = df[df.traj == "trunk"].groupby(["setup", "trunk"]).agg(U=("u", "mean"), final=("y_clean_acc", "last"))
    tr = tr.unstack("trunk")
    out = pd.DataFrame({"on_minus_off_U": tr[("U", 1)] - tr[("U", 0)],
                        "final_off": tr[("final", 0)], "final_on": tr[("final", 1)]})
    return out.join(man.set_index("setup")[["family", "config", "duration", "n_attackers"]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, man = load()
    g = gains(df, man)
    g.to_parquet(OUT / "gains.parquet")
    tt = trunk_table(df, man)
    tt.to_csv(OUT / "trunks.csv")
    pd.set_option("display.width", 200, "display.max_rows", 200)

    def summ(x):
        return pd.Series({"n": len(x), "gain": x.gain.mean(), "defend_wins": (x.gain > 0).mean(),
                          "d_final_acc": x.gain_final_acc.mean(), "d_atk_40": x.gain_atk_window.mean()})

    print("== always-on minus always-off (whole-run U), by family")
    print(tt.groupby("family").agg(n=("on_minus_off_U", "size"), on_minus_off=("on_minus_off_U", "mean"),
                                    on_better=("on_minus_off_U", lambda v: (v > 0).mean()),
                                    final_off=("final_off", "mean"), final_on=("final_on", "mean")).round(3))
    bm = g.groupby(["family", "moment", "trunk"]).apply(summ, include_groups=False).round(3)
    bm.to_csv(OUT / "by_moment.csv")
    print("\n== branch gain of defending 40 steps (trunk 0 = from off, 1 = from on)")
    print(bm)
    bc = g.groupby(["config", "trunk"]).apply(summ, include_groups=False).round(3)
    bc.to_csv(OUT / "by_config.csv")
    print("\n== by config")
    print(bc)
    print("\n== overall: share of branch points where defending wins:", round((g.gain > 0).mean(), 3),
          "| |gain| < 0.002:", round((g.gain.abs() < 0.002).mean(), 3))


if __name__ == "__main__":
    main()
