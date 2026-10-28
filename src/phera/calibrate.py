"""Extract chart colours from a photo and colour-correct the pad using the
printed reference patches (per-channel tone curve from the grey ramp, then a
ridge-regularised 3x3 colour matrix)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .chart import N_GREY, REF_SRGB, rois


def extract(image_u8: np.ndarray):
    """Return (pad_rgb (3,), patches (8,3)) in [0,1]; twin patches are averaged."""
    img = image_u8.astype(float) / 255.0
    top, bot, pad = rois()
    t = np.array([np.median(img[ys, xs].reshape(-1, 3), 0) for ys, xs in top])
    b = np.array([np.median(img[ys, xs].reshape(-1, 3), 0) for ys, xs in bot])
    return np.median(img[pad].reshape(-1, 3), 0), (t + b) / 2


@dataclass
class Calibration:
    grey_obs: np.ndarray   # (3, N_GREY+1) sorted observed grey levels per channel (with black anchor)
    grey_tgt: np.ndarray   # (N_GREY+1,)
    matrix: np.ndarray     # (3, 3)

    def tone(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.atleast_2d(rgb)
        out = np.empty_like(rgb)
        for c in range(3):
            out[:, c] = np.interp(rgb[:, c], self.grey_obs[c], self.grey_tgt)
        return out

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        out = np.clip(self.tone(rgb) @ self.matrix.T, 0, 1)
        return out[0] if np.ndim(rgb) == 1 else out


def fit_calibration(patches_obs: np.ndarray, ridge: float = 0.05) -> Calibration:
    tgt = np.concatenate([[0.0], REF_SRGB[:N_GREY, 0]])
    order = np.argsort(tgt)
    tgt = tgt[order]
    obs = np.zeros((3, N_GREY + 1))
    for c in range(3):
        o = np.concatenate([[0.0], patches_obs[:N_GREY, c]])[order]
        obs[c] = np.maximum.accumulate(o + 1e-6 * np.arange(len(o)))  # enforce monotone
    cal = Calibration(obs, tgt, np.eye(3))
    x = cal.tone(patches_obs)
    a_t = np.linalg.solve(x.T @ x + ridge * np.eye(3), x.T @ REF_SRGB + ridge * np.eye(3))
    cal.matrix = a_t.T
    return cal


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.atleast_2d(rgb)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def features(image_u8: np.ndarray):
    """Return (raw_lab (3,), calibrated_lab (3,), calibrated_rgb (3,))."""
    pad, patches = extract(image_u8)
    corrected = fit_calibration(patches).apply(pad)
    return srgb_to_lab(pad)[0], srgb_to_lab(corrected)[0], corrected
