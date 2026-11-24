"""Inputs and splits: exactly those of the xp6 OutcomeGRU (gru_experiment.Data, gru_xp5.seen_splits).

Encoder rows (row i = block i-1; row 0 = initial state): [80 signals (normalised, clipped), W, t, init flag]. The
defense one-hot of gru_experiment.Data is dropped (constant under Krum only). Outcomes [y_clean_acc, y_atk] are
targets only, never inputs. See notes/02_inputs.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import _GRU  # noqa: F401  (sys.path)
from analyze import DEFENSES  # noqa: E402
from gru_experiment import Data, x_columns  # noqa: E402

FORBIDDEN_PREFIXES = ("y_", "u_")  # outcomes and attack labels: never an input
N_DEF = len(DEFENSES)


def input_columns(df: pd.DataFrame) -> list[str]:
    """The 80 signal columns (explicit list, checked: no outcome or attack label can slip in)."""
    cols = x_columns(df)
    bad = [c for c in cols if c.startswith(FORBIDDEN_PREFIXES) or c in ("attack", "poison_frac", "run_id")]
    if bad:
        raise ValueError(f"forbidden input columns: {bad}")
    return cols


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    heldout: pd.DataFrame


def load_splits(csv: str) -> Splits:
    """xp6 splits as in gru_xp5 (seen runs split by seed; held-out groups never trained on)."""
    from gru_xp5 import groups_of, seen_splits  # lazy: gru_xp5 pulls in most of GRU_mnist (not bundled on Kaggle)
    df = pd.read_csv(csv, dtype={"round_start": "Int64", "W": "Int64"}, low_memory=False)
    _, train, val, test = seen_splits(df, [])
    return Splits(train, val, test, df[groups_of(df) != "seen"])


class RunData:
    """gru_experiment.Data plus the CPC views. Normalisation from the training runs (pass ref=train RunData)."""

    def __init__(self, df: pd.DataFrame, ref: "RunData | None" = None):
        cols = input_columns(df) if ref is None else ref.x_cols
        self.d = Data(df, None if ref is None else ref.d.x_mean, None if ref is None else ref.d.x_std, cols)
        self.x_cols = cols
        self.enc = self.d.enc[..., :-N_DEF].contiguous()  # (R, n, 80 + 3)
        self.W = torch.tensor(self.d.W, dtype=torch.float32)  # (R, n); row 0 = 0
        self.Y, self.M = self.d.Y, self.d.Ymask  # (R, n, 2)
        n = self.d.n_rows
        self.t = torch.tensor((np.arange(n) - 1) / self.d.n_blocks, dtype=torch.float32)  # (n,)
        self.seed = torch.tensor(self.d.seed.astype(np.int64))
        self.run_ids, self.attack = self.d.run_ids, self.d.attack

    def __len__(self) -> int:
        return len(self.run_ids)

    @property
    def n_rows(self) -> int:
        return self.d.n_rows
