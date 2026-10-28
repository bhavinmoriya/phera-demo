"""pH regressor on colour features with split-conformal prediction intervals
calibrated on held-out *phones* (so coverage reflects unseen devices)."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .calibrate import features
from .camera import LIGHTINGS, PHONES, capture
from .chart import PH_LEVELS, PH_MAX, PH_MIN, PH_STEP


def build_dataset(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    light_names = list(LIGHTINGS)
    ph = rng.choice(PH_LEVELS, n)
    phone = rng.integers(0, len(PHONES), n)
    light = rng.integers(0, len(light_names), n)
    raw, cal = np.empty((n, 3)), np.empty((n, 3))
    for i in range(n):
        img = capture(ph[i], PHONES[phone[i]], LIGHTINGS[light_names[light[i]]], rng)
        raw[i], cal[i], _ = features(img)
    return dict(ph=ph, phone=phone, light=light, raw=raw, cal=cal, light_names=light_names)


def _reg():
    return make_pipeline(PolynomialFeatures(3), StandardScaler(), Ridge(alpha=1.0))


def quantise(ph):
    return np.clip(np.round((np.asarray(ph) - PH_MIN) / PH_STEP) * PH_STEP + PH_MIN, PH_MIN, PH_MAX)


class PhModel:
    """Regressor + conformal half-width q (coverage `1-alpha`)."""

    def __init__(self, alpha: float = 0.1):
        self.alpha, self.reg, self.q = alpha, _reg(), np.nan

    def fit(self, X, y, groups):
        self.reg.fit(X, y)
        oof = np.empty_like(y, float)
        for g in np.unique(groups):                       # leave-one-phone-out residuals
            m = groups == g
            oof[m] = _reg().fit(X[~m], y[~m]).predict(X[m])
        res = np.abs(y - oof)
        k = int(np.ceil((len(res) + 1) * (1 - self.alpha)))
        self.q = float(np.sort(res)[min(k, len(res)) - 1])
        return self

    def predict(self, X):
        mu = np.clip(self.reg.predict(np.atleast_2d(X)), PH_MIN, PH_MAX)
        return mu, mu - self.q, mu + self.q
