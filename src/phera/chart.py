"""Ideal (camera-free) colour chart: indicator pad + reference patches.

IMPORTANT: the pH -> colour mapping below is a *synthetic* yellow-green-blue
indicator curve, not real chemistry. Replace it with measured strip data.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import shift as nd_shift

PH_MIN, PH_MAX, PH_STEP = 3.5, 7.0, 0.25
PH_LEVELS = np.arange(PH_MIN, PH_MAX + 1e-9, PH_STEP)

_ANCHOR_PH = np.array([3.5, 4.5, 5.5, 6.5, 7.0])
_ANCHOR_RGB = np.array(
    [[240, 200, 60], [190, 205, 70], [110, 185, 100], [60, 140, 150], [50, 110, 170]], float
) / 255.0

W, H = 320, 240
PATCH = 36
X0 = 16
TOP_Y, BOT_Y = 14, 190
PAD_C, PAD_SIZE = (160, 120), 80

# 4 grey levels + 4 colours. Each patch appears twice, point-symmetric about the
# pad centre, so the average of a twin pair cancels linear illumination gradients.
REF_SRGB = np.array(
    [
        [0.92, 0.92, 0.92], [0.66, 0.66, 0.66], [0.40, 0.40, 0.40], [0.16, 0.16, 0.16],
        [0.78, 0.20, 0.17], [0.20, 0.60, 0.25], [0.20, 0.33, 0.72], [0.88, 0.76, 0.22],
    ]
)
N_GREY = 4


def ph_to_srgb(ph: float) -> np.ndarray:
    return np.array([np.interp(ph, _ANCHOR_PH, _ANCHOR_RGB[:, c]) for c in range(3)])


def rois(margin_patch: int = 10, margin_pad: int = 20):
    """Nominal ROI slices (inside each patch / pad, with margin for placement jitter)."""
    top, bot = [], []
    for k in range(8):
        xt = X0 + PATCH * k
        xb = X0 + PATCH * (7 - k)
        top.append((slice(TOP_Y + margin_patch, TOP_Y + PATCH - margin_patch),
                    slice(xt + margin_patch, xt + PATCH - margin_patch)))
        bot.append((slice(BOT_Y + margin_patch, BOT_Y + PATCH - margin_patch),
                    slice(xb + margin_patch, xb + PATCH - margin_patch)))
    h = PAD_SIZE // 2 - margin_pad
    pad = (slice(PAD_C[1] - h, PAD_C[1] + h), slice(PAD_C[0] - h, PAD_C[0] + h))
    return top, bot, pad


def render_ideal(ph: float, rng: np.random.Generator | None = None, jitter: float = 5.0) -> np.ndarray:
    """Float sRGB image (H, W, 3) in [0, 1] of the chart for a given pH."""
    img = np.empty((H, W, 3))
    img[:] = (0.30, 0.22, 0.16)                      # table
    img[6:H - 6, 6:W - 6] = 0.93                     # white card
    for k in range(8):
        xt, xb = X0 + PATCH * k, X0 + PATCH * (7 - k)
        img[TOP_Y:TOP_Y + PATCH, xt:xt + PATCH] = REF_SRGB[k]
        img[BOT_Y:BOT_Y + PATCH, xb:xb + PATCH] = REF_SRGB[k]
    cx, cy = PAD_C
    h = PAD_SIZE // 2
    img[cy - h:cy + h, cx - h:cx + h] = ph_to_srgb(ph)
    if rng is not None and jitter > 0:
        dy, dx = rng.uniform(-jitter, jitter, 2)
        img = np.stack([nd_shift(img[..., c], (dy, dx), order=0, mode="nearest") for c in range(3)], -1)
    return img
