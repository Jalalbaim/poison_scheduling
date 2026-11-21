# Training-time data-poisoning defenses on `krum`

Two workspaces, one shared virtual environment (`.venv/`, Python 3.12, krum 0.10.0, torch 2.x).

| folder | status | what |
|---|---|---|
| [`GRU_mnist/`](GRU_mnist/README.md) | **archived** (xp1 – xp6, frozen at commit `038a33b`) | the poisoning testbed (`poison_testbed/`), data collection, the encoder–decoder outcome GRU, deployment and the Kaggle pipeline |
| [`CPC_DP/`](CPC_DP/README.md) | **active** | counterfactual regression of "defend vs. not defend" (Causal CPC direction) |

Run everything from inside its folder, with the shared interpreter: `cd GRU_mnist && ../.venv/Scripts/python ...`.
The `.venv` stays at the root on purpose: moving a Windows venv breaks its launchers.
