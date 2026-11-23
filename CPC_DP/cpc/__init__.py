"""Causal CPC for 'when to defend' (CPC_DP). Reuses the GRU_mnist testbed and data pipeline unchanged.

Modules: data (inputs, splits, normalisation shared with the xp6 OutcomeGRU), model (CausalCPC), train (two-stage
training), evaluate (factual error, Delta_k against the onset twins, for CPC and the OutcomeGRU alike).
"""

import sys
from pathlib import Path

_GRU = Path(__file__).resolve().parent.parent.parent / "GRU_mnist"
if (_GRU / "gru_experiment.py").exists() and str(_GRU) not in sys.path:  # local layout (Kaggle: files side by side)
    sys.path.insert(0, str(_GRU))
