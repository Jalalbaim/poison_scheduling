"""Validate the xp3 runs: completeness, config consistency, labels, twins; outcome summary."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import run_experiment as rx

sys.stdout.reconfigure(encoding="utf-8")
root = Path("results/xp3/runs_xp3")
df = pd.read_csv(root / "all_blocks.csv")
cfgs = {c.run_id: c for c in rx.xp3_configs({})}
print("runs in csv", df.run_id.nunique(), "expected", len(cfgs), "rows/run", df.groupby("run_id").size().unique())
missing = set(cfgs) - set(df.run_id); extra = set(df.run_id) - set(cfgs)
print("missing", len(missing), "extra", len(extra))
bad_hash, bad_sched, gpus = 0, 0, set()
for rid, c in cfgs.items():
    j = json.loads((root / rid / "config.json").read_text())
    gpus.add(j["device_name"])
    bad_hash += j["config_hash"] != c.hash()
    w = df[(df.run_id == rid) & (df.block >= 0)].sort_values("block").W.astype(int).tolist()
    bad_sched += w != j["schedule"]
print("hash mismatches", bad_hash, "| logged W != config schedule", bad_sched, "| gpus", gpus)
print("NaN y_atk (block>=0):", int(df[df.block >= 0].y_atk.isna().sum()), "| NaN X:", int(df[df.block >= 0].x_norm_std.isna().sum()))
# window labels
bad_u = 0
for rid, c in cfgs.items():
    g = df[(df.run_id == rid) & (df.block >= 0)].sort_values("block")
    exp = ((g.block >= c.window[0]) & (g.block <= c.window[1])).astype(int) if c.attack != "none" else 0 * g.block
    bad_u += int((g.u_active.values != np.asarray(exp)).sum())
print("u_active mismatches", bad_u)
# twins: identical up to flip start (X, Y) and diverge after
tw = []
for rid in cfgs:
    m = re.match(r"(.+)__(.+)\+flip_s(\d+)_l(\d+)__s(\d+)", rid)
    if not m: continue
    base = rid.replace(f"+flip_s{m.group(3)}_l{m.group(4)}", "")
    s = int(m.group(3))
    a = df[df.run_id == rid].set_index("block"); b = df[df.run_id == base].set_index("block")
    pre = (a.loc[:s - 1, ["x_norm_std", "y_clean_acc", "y_atk"]] - b.loc[:s - 1, ["x_norm_std", "y_clean_acc", "y_atk"]]).abs().max().max() if s > 0 else 0.0
    post = (a.loc[s:, "x_norm_std"] - b.loc[s:, "x_norm_std"]).abs().max()
    c = cfgs[rid]
    tw.append({"twin": rid, "s": s, "pre_maxdiff": pre, "post_maxdiff": post, "pf": c.poison_frac, "attack": c.attack,
               "d_atk_final": a.y_atk.iloc[-1] - b.y_atk.iloc[-1], "d_clean_final": a.y_clean_acc.iloc[-1] - b.y_clean_acc.iloc[-1]})
tw = pd.DataFrame(tw)
print(f"twins {len(tw)}: identical before flip in {int((tw.pre_maxdiff == 0).sum())}, diverge after in {int((tw.post_maxdiff > 0).sum())}")
print("twins with |final backdoor change| > 0.1:", int((tw.d_atk_final.abs() > 0.1).sum()))
# outcome summary
last = df[df.block == df.block.max()].copy()
last["fam"] = last.run_id.str.extract(r"__Krum__([a-z]+)")[0]
last["twin"] = last.run_id.str.contains(r"\+flip")
last["pf"] = np.where(last.attack == "none", "none", last.poison_frac.map({0.125: "1/8", 0.25: "2/8", 1.0: "8/8"}))
m = last[~last.twin]
print("\nmain runs per pf x family"); print(pd.crosstab(m.pf, m.fam, margins=True))
print("\nfinal backdoor success, mean (share killed < 0.1)")
t = m.groupby(["pf", "fam"]).y_atk.agg(lambda s: f"{s.mean():.2f} ({(s < 0.1).mean():.0%})").unstack()
print(t.to_string())
print("\nfinal clean acc mean"); print(m.groupby(["pf", "fam"]).y_clean_acc.mean().unstack().round(3).to_string())
tw.to_csv("results/xp3/twins_check.csv", index=False)
