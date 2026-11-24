"""Alarm-driven defense schedules (xp6): the defense reacts to an OBSERVABLE alarm, never to the true attack window.

Why (xp5a): schedules that switch the defense on at the true window start ('react', 'probehold') make a planned
W = 1 announce an attack, and an outcome model fed with the planned W learns 'W = 1 -> high attack success'. Here
W depends only on what a defender sees (the signals and its own past W), as at deployment.

Alarm score of a block = max of two z-scores, each against no-attack training blocks of the same defense, defense
state W and 25-block time bin: x_norm_std (v1) and x2_outlier_proj_max_max (v2). The z tables are the xp5 alarm
(deploy_xp5.fit_alarm on xp5 training runs, file alarm_xp5.json), fixed before any xp6 run.

Schedule "alarm_t<tau>_d<delay>_h<hold>":
    an alarm is a block with score > tau; the defense switches on `delay` blocks after an alarm seen while off,
    then stays on until `hold` consecutive blocks without an alarm (at least `hold` blocks).
"""

import json
import math
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ALARM_FILE = Path(__file__).with_name("alarm_xp5.json")
ALARM_SCHEDULE = re.compile(r"alarm_t([0-9.]+)_d(\d+)_h(\d+)")
GAMMA_SCHEDULE = re.compile(r"gamma_g([0-9.]+)_c([0-9.]+)_p(\d+)")
V1, V2, BIN = "x_norm_std", "x2_outlier_proj_max_max", 25


@lru_cache(maxsize=None)
def load_alarm(path: str = str(ALARM_FILE)) -> dict:
    """{'table': {'<defense>|<W>|<bin>': [mean_v1, sd_v1, mean_v2, sd_v2]}, 'threshold': float}."""
    return json.loads(Path(path).read_text())


def score(row: dict[str, Any], table: dict) -> float:
    """Alarm score of one observed block row (-inf when it cannot be scored: init row, missing signal or bin)."""
    if row.get("block", -1) < 0:
        return -math.inf
    w = row.get("W")
    if w is None or (isinstance(w, float) and math.isnan(w)):
        return -math.inf
    stats = table.get(f"{row['defense']}|{int(w)}|{int(row['block']) // BIN}")
    v1, v2 = row.get(V1), row.get(V2)
    if stats is None or v1 is None or v2 is None or not (math.isfinite(v1) and math.isfinite(v2)):
        return -math.inf
    m1, s1, m2, s2 = stats
    return max((v1 - m1) / max(s1, 1e-9), (v2 - m2) / max(s2, 1e-9))


class AlarmSchedule:
    """Online policy(history, b) -> (W_b, extra) for simulate(); history = observable rows of blocks -1..b-1."""

    def __init__(self, tau: float, delay: int, hold: int, table: dict):
        self.tau, self.delay, self.hold, self.table = tau, delay, hold, table
        self.on, self.pending, self.quiet = False, None, 0

    def __call__(self, history: list[dict[str, Any]], b: int) -> tuple[int, dict[str, Any]]:
        hit = score(history[-1], self.table) > self.tau
        self.quiet = 0 if hit else self.quiet + 1
        if hit and not self.on and self.pending is None:
            self.pending = b + self.delay
        if self.pending is not None and b >= self.pending:
            self.on, self.pending, self.quiet = True, None, 0
        elif self.on and self.quiet >= self.hold:
            self.on = False
        return int(self.on), {}


class GammaSchedule:
    """xp7: stochastic alarm policy with a known propensity. Every `period` blocks (from block 0) the defense is
    redrawn: P(on) = sigmoid(gamma * (s - center)), s = the highest alarm score seen since the previous decision
    (-inf before any scored block). gamma = 0: a fair coin (no confounding); larger gamma: closer to a threshold
    rule on the alarm. W is held between decisions. Logged on every row: p_on (propensity of the current W's
    decision) and decision (1 on the block where it was drawn)."""

    def __init__(self, gamma: float, center: float, period: int, table: dict, seed: int):
        self.gamma, self.center, self.period, self.table = gamma, center, period, table
        self.rng = random.Random(seed)
        self.w, self.p, self.s = 0, 0.5, -math.inf

    def __call__(self, history: list[dict[str, Any]], b: int) -> tuple[int, dict[str, Any]]:
        self.s = max(self.s, score(history[-1], self.table))
        if b % self.period:
            return self.w, {"p_on": self.p, "decision": 0}
        z = self.gamma * (self.s - self.center) if self.gamma > 0 else 0.0
        self.p = 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0
        self.w, self.s = int(self.rng.random() < self.p), -math.inf
        return self.w, {"p_on": self.p, "decision": 1}


def is_alarm_schedule(schedule: str) -> bool:
    """Schedules decided online by a policy (alarm or gamma)."""
    return ALARM_SCHEDULE.fullmatch(schedule) is not None or GAMMA_SCHEDULE.fullmatch(schedule) is not None


def alarm_policy(schedule: str, seed: int = 0):
    if m := ALARM_SCHEDULE.fullmatch(schedule):
        return AlarmSchedule(float(m.group(1)), int(m.group(2)), int(m.group(3)), load_alarm()["table"])
    if m := GAMMA_SCHEDULE.fullmatch(schedule):
        return GammaSchedule(float(m.group(1)), float(m.group(2)), int(m.group(3)), load_alarm()["table"], seed)
    raise ValueError(f"not an online schedule: {schedule!r}")
