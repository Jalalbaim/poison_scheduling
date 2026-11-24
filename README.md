# Training-time data-poisoning defenses on `krum`

Two workspaces, one shared virtual environment (`.venv/`, Python 3.12, krum 0.10.0, torch 2.x).

| folder                              | status                                               | what                                                                                                                            |
| ----------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| [`GRU_mnist/`](GRU_mnist/README.md) | **archived** (xp1 – xp6, frozen at commit `038a33b`) | the poisoning testbed (`poison_testbed/`), data collection, the encoder–decoder outcome GRU, deployment and the Kaggle pipeline |
| [`CPC_DP/`](CPC_DP/README.md)       | **active**                                           | counterfactual regression of "defend vs. not defend" (Causal CPC direction)                                                     |

Run everything from inside its folder, with the shared interpreter: `cd GRU_mnist && ../.venv/Scripts/python ...`.
The `.venv` stays at the root on purpose: moving a Windows venv breaks its launchers.

2. xp8: the new data

Point 1: decide at step t using the gradients of step t

The server receives the 16 worker gradients first. It computes the signals from them (norms, Krum scores, loss ratios…), then chooses Krum or plain averaging, then applies the update. No delay at all.

Two things I'll check in the code:

- Signal timing. All 80 signals must be computable before aggregation. Most are already computed from the raw gradients. Any that need the aggregated update will stay one step behind.
- Decision rate. Today a block is 2 SGD steps. I suggest one decision per block, made with the first step's gradients and applied to both steps.

Point 2: paired branches for training

1. Run a normal training run (the "trunk"). Krum is switched by a fair coin, so the histories contain both on and off periods.
2. At a few chosen moments, save the run's state and split it into two copies:
   - A: Krum on for the next 20 blocks;
   - B: Krum off for those 20 blocks.
3. After those 20 blocks, both copies follow the same coin flips to the end. So the only difference between A and B is the decision at that moment.
4. Each pair gives one clean example of "the effect of defending here".

Where to split, for attacked runs:

- at the attack start;
- 5 blocks after the start;
- 15 blocks after the start;
- one quiet moment (before the attack, or after it ends).

For runs with no attack: 4 random moments.

Point 3: cases where defending is the wrong choice

- No attack (about 20% of runs). Krum costs about 8 points of clean accuracy for nothing.
- ALIE. Krum helps the attacker.
- Weak attacks. For example one attacker, or a small poisoned share, where Krum costs more than the attack does.
- More attackers than Krum expects. Krum is set up for 3; with 5 it can fail.
- Attack about to end. Switching on late wastes accuracy.

Point 4: attack variety (all of these already exist in the testbed)

┌─────────────────────────────────┬──────────────────────────────┬───────────────┐
│ family │ what varies │ defending is… │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ backdoor corner (small trigger) │ poisoned share 2/8, 4/8, 8/8 │ usually right │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ backdoor patch (big trigger) │ poisoned share 2/8, 4/8, 8/8 │ right │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ targeted label flip │ poisoned share 4/8, 8/8 │ right │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ ALIE │ strength z: mild / strong │ wrong │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ SignFlip │ scale 2 / 4.6 │ right │
├─────────────────────────────────┼──────────────────────────────┼───────────────┤
│ no attack │ none │ wrong │
└─────────────────────────────────┴──────────────────────────────┴───────────────┘

For every family, I'd also vary:

- Duration: short (15 blocks), long (60 blocks), or persistent (to the end). SignFlip gets no persistent version, because it crashed training.
- Start time: random, early to late.
- Number of attackers: drawn per run from 1, 3 or 5.

Point 5: how many runs

┌─────────────────┬───────────────────────────────┬────────┐
│ family │ configurations │ trunks │
├─────────────────┼───────────────────────────────┼────────┤
│ no attack │ 1 │ 60 │
├─────────────────┼───────────────────────────────┼────────┤
│ backdoor corner │ 3 shares × 3 durations = 9 │ 54 │
├─────────────────┼───────────────────────────────┼────────┤
│ backdoor patch │ 3 shares × 3 durations = 9 │ 54 │
├─────────────────┼───────────────────────────────┼────────┤
│ targeted flip │ 2 shares × 3 durations = 6 │ 36 │
├─────────────────┼───────────────────────────────┼────────┤
│ ALIE │ 2 strengths × 3 durations = 6 │ 36 │
├─────────────────┼───────────────────────────────┼────────┤
│ SignFlip │ 2 scales × 2 durations = 4 │ 24 │
├─────────────────┼───────────────────────────────┼────────┤
│ total │ 35 │ 264 │
└─────────────────┴───────────────────────────────┴────────┘

What this gives:

- Pairs: 264 trunks × 4 split points = 1,056 pairs (2,112 branches).
- Split by trunk: 70% train, 15% val, 15% test. So about 740 training pairs, 160 val and 160 test.
  - The test pairs replace the old "twins": same kind of data, kept apart.
- Bonus: each trunk also has 25 coin-flip decisions, usable as extra data (about 6,600 rows).

Compared with xp7: 134 training runs and no training pairs at all.

Cost, in full-run equivalents (Kaggle does about 80 full runs per hour):

┌─────────────────┬───────────────────────────────────┬──────────────────────────────────────────────┐
│ │ with save/resume of the run state │ without (each branch re-runs from the start) │
├─────────────────┼───────────────────────────────────┼──────────────────────────────────────────────┤
│ run equivalents │ 264 + 2,112 × ~0.55 ≈ 1,430 │ 264 + 2,112 ≈ 2,380 │
├─────────────────┼───────────────────────────────────┼──────────────────────────────────────────────┤
│ Kaggle time │ ~18 h → 2 sessions │ ~30 h → 3 sessions │
└─────────────────┴───────────────────────────────────┴──────────────────────────────────────────────┘

I recommend building save/resume into the testbed. It's a moderate change and it halves the cost. A branch then only simulates from the split point to the end.

If we need it smaller: 3 split points instead of 4, and 2 durations instead of 3, gives about 180 trunks and 540 pairs, around 8 h (one Kaggle session).

What I need from you

1. Is the attack table OK? Anything to add or remove?
2. Full size (2 sessions) or reduced (1 session)?
3. OK to build save/resume into the testbed?
