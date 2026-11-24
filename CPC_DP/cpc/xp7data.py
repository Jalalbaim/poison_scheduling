"""xp7 data for the GRU and CPC models: splits from the xp7 manifests, one dataset per policy level.

- Inputs: the same 80 signals as xp6 (cpc.data.input_columns), passed through a signed log, sign(x) log(1 + |x|),
  before the usual normalisation: SignFlip windows push some signals to ~1e8 (notes/03_xp7_design.md).
- Outcomes [y_clean_acc, y_atk]: y_atk is the targeted attack's success (backdoor / flip). ALIE and SignFlip have no
  targeted outcome (y_atk is NaN in the logs): their harm is entirely in y_clean_acc, so y_atk is set to the
  measured backdoor base rate (y_bd), as in no-attack runs. The utility U = clean - lam * atk then counts their
  damage once, through clean accuracy.
- Policy logs (p_on, decision) are never inputs (input_columns keeps only x_ signals).

Splits (by base setup, identical at every level): train 134, val 22, test 44 setups. Twins: val + test setups only.
Never train on twins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import RunData, input_columns

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results/xp7"
SRC = {p: RES / p for p in ("a", "b", "twins")}
CACHE = RES / "cache"
LEVELS = ["g0", "g2", "g8", "hard"]
PART_OF = {"g0": "a", "g2": "a", "g8": "b", "hard": "b"}
GRADIENT = ("alie_win", "sign_flip_win")


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Signed-log inputs, y_atk for gradient attacks, W as Int64. Idempotent guard via a marker column."""
    if "_prepared" in df:
        return df
    df = df.copy()
    cols = input_columns(df)
    df[cols] = signed_log(df[cols].to_numpy(float))
    grad = df.attack.isin(GRADIENT)
    df.loc[grad, "y_atk"] = df.loc[grad, "y_bd"]
    df["W"] = pd.to_numeric(df.W, errors="coerce").astype("Int64")
    df["_prepared"] = 1
    return df


def _read(part: str) -> pd.DataFrame:
    """One xp7 part, prepared, with its manifest columns; cached in results/xp7/cache (the CSVs are ~190 MB)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{part}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    df = pd.read_csv(SRC[part] / "all_blocks.csv", dtype={"round_start": "Int64", "W": "Int64"}, low_memory=False)
    man = pd.read_csv(SRC[part] / "manifest.csv")
    keep = [c for c in man.columns if c not in df.columns or c == "run_id"]
    df = prepare(df).merge(man[keep], on="run_id")
    df.to_pickle(f)
    return df


def level_splits(level: str) -> dict[str, pd.DataFrame]:
    """{'train', 'val', 'test'}: the main runs of one policy level."""
    df = _read(PART_OF[level])
    df = df[df.level == level]
    return {s: df[df.split == s] for s in ("train", "val", "test")}


def twins() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(blocks, manifest) of the onset twins (val + test setups)."""
    df = _read("twins")
    return df, pd.read_csv(SRC["twins"] / "manifest.csv")


def level_data(level: str) -> tuple[RunData, RunData, RunData]:
    sp = level_splits(level)
    tr = RunData(sp["train"])
    return tr, RunData(sp["val"], ref=tr), RunData(sp["test"], ref=tr)
