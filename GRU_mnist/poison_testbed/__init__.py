"""Data-poisoning testbed on top of the ``krum`` library.

One SGD step = one round; the n workers are the n micro-batches of a single
mini-batch, drawn IID from the full training set. See ``run_experiment.py``.
"""

import os

# Required by torch.use_deterministic_algorithms on CUDA; must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
