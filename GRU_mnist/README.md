# GRU_mnist — archived GRU experiments (xp1 – xp6)

**Archived.** Everything produced before the Causal CPC direction, moved here unchanged (git history follows the
files: `git log --follow GRU_mnist/<file>`). New work goes in [`../CPC_DP`](../CPC_DP/README.md); import from
here rather than editing in place, unless a bug fix is needed.

## Layout (all paths are relative to this folder; code resolves them from its own location)

| path | content |
|---|---|
| `poison_testbed/` | the testbed: one SGD step = one round, 16 micro-batches = workers, f = 3 poisoned; attacks, signals v1/v2, runner, `alarm.py` (xp6 alarm schedules, table `alarm_xp5.json`) |
| `run_experiment.py` | run lists and data collection for every experiment (`sanity`, `collect`, `xp2` … `xp6`) |
| `gru_experiment.py`, `gru_xp5.py` | encoder–decoder outcome GRU (inputs: observables X, W, t, defense; targets: clean accuracy, attack success) |
| `deploy*.py`, `check_probe.py`, `detection_xp4.py`, `pilot_xp*.py`, `measure_erasure.py`, `validate_xp*.py`, `analyze.py` | deployment, checks, pilots, analysis |
| `kaggle/` + `run_kaggle.sh` | self-contained Kaggle script (rebuild with `bash kaggle/build.sh` after any code change) |
| `tests/` | `../.venv/Scripts/python -m pytest tests` |
| `results/`, `data/`, `krum/`, `runs_sanity/`, `draft.tex`, `*.log` | git-ignored: results of xp1 – xp5a, datasets, the krum library clone (only its commit hash is logged) |

## Experiment status

| xp | what | where |
|---|---|---|
| xp1 – xp4 | MNIST, data collection, signals v1/v2, deployment pipeline | `results/xp1` … `results/xp4`, `results/deploy*` |
| xp5 | Fashion-MNIST, Krum / Median / TrimmedMean, backdoor / burst / flip / ALIE; GRU never defends on its own (leak through `react` / `probehold` schedules) | `results/xp5` |
| xp5a | leaky schedules removed, `freeze` end rule; GRU defends on its own (2/8 corner onset 176 → 18 blocks) | `results/xp5a_all` |
| xp6 | **built, not run**: Krum only, backdoor + flip, 500 runs, alarm-driven schedules | `bash run_kaggle.sh xp6` → `results/xp6` |

Running from here works exactly as before, e.g. `../.venv/Scripts/python gru_xp5.py --runs results/xp5/runs_xp5/all_blocks.csv --out results/xp5/gru`.
