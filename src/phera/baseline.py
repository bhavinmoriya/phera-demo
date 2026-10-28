"""Personal baseline: Bayesian local-level (Kalman) model with a population prior,
plus a persistence rule for flagging meaningful changes."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PersonalBaseline:
    prior_mean: float = 4.2     # population prior for a new user's baseline
    prior_sd: float = 0.40      # between-person spread (hierarchical prior)
    meas_sd: float = 0.25       # per-reading noise (take from the model's error)
    drift_sd: float = 0.03      # day-to-day baseline wander
    z_thresh: float = 2.5
    persist: int = 2            # consecutive outliers needed to call a change
    mean: float = field(init=False)
    var: float = field(init=False)
    _streak: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.mean, self.var = self.prior_mean, self.prior_sd ** 2

    def update(self, y: float) -> dict:
        var_pred = self.var + self.drift_sd ** 2
        s = var_pred + self.meas_sd ** 2
        z = (y - self.mean) / np.sqrt(s)
        changed = False
        if abs(z) > self.z_thresh:
            self._streak.append(y)
            self.var = var_pred
            if len(self._streak) >= self.persist and np.sign(self._streak[-1] - self.mean) == np.sign(z):
                changed = True
                self.mean, self.var = float(np.mean(self._streak)), self.meas_sd ** 2 / len(self._streak)
                self._streak = []
        else:
            self._streak = []
            k = var_pred / s
            self.mean += k * (y - self.mean)
            self.var = (1 - k) * var_pred
        return dict(mean=self.mean, sd=float(np.sqrt(self.var)), z=float(z), change=changed)


def simulate_user(days=40, base=4.1, shift_day=25, shift=1.0, meas_sd=0.25, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.full(days, base) + np.cumsum(rng.normal(0, 0.02, days))
    truth[shift_day:] += shift
    return truth, truth + rng.normal(0, meas_sd, days)
