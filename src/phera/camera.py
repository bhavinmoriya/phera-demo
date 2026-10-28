"""Smartphone camera + lighting simulator.

Each phone has its own colour matrix, tone curve (gamma), auto-white-balance
strength, noise and blur. Each shot also draws a lighting condition
(colour temperature, brightness, shading gradient).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from .chart import H, W, render_ideal


def srgb_to_linear(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@dataclass(frozen=True)
class Phone:
    name: str
    matrix: np.ndarray
    gamma: float
    wb: float          # 0 = no auto white balance, 1 = perfect
    noise: float
    blur: float
    seen_in_training: bool


def _make_phones() -> list[Phone]:
    rs = np.random.RandomState(7)
    phones = []
    for i in range(6):
        m = np.eye(3) + 0.13 * rs.randn(3, 3)
        m /= m.sum(1, keepdims=True)                      # keep white neutral
        phones.append(Phone(
            name=f"Phone {'ABCDEF'[i]}", matrix=m, gamma=float(rs.uniform(1.9, 2.5)),
            wb=float(rs.uniform(0.0, 0.7)), noise=float(rs.uniform(0.01, 0.03)),
            blur=float(rs.uniform(0.6, 1.6)), seen_in_training=i < 4))
    return phones


PHONES = _make_phones()


@dataclass(frozen=True)
class Lighting:
    name: str
    gains: tuple       # illuminant RGB gains
    exposure: float
    shading: float     # max relative brightness gradient across the image


LIGHTINGS = {
    "daylight": Lighting("daylight", (1.00, 1.00, 1.00), 1.00, 0.08),
    "warm indoor": Lighting("warm indoor", (1.30, 1.00, 0.62), 0.75, 0.15),
    "cool LED": Lighting("cool LED", (0.78, 1.00, 1.30), 0.85, 0.12),
    "dim + warm": Lighting("dim + warm", (1.25, 1.00, 0.70), 0.35, 0.25),
}


def capture(ph: float, phone: Phone, light: Lighting, rng: np.random.Generator) -> np.ndarray:
    """Return an 8-bit RGB image as this phone would record the chart under this light."""
    lin = srgb_to_linear(render_ideal(ph, rng))
    gains = np.array(light.gains) * np.exp(rng.normal(0, 0.05, 3))
    exposure = light.exposure * np.exp(rng.normal(0, 0.15))

    yy, xx = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    theta = rng.uniform(0, 2 * np.pi)
    field = 1 + rng.uniform(0, light.shading) * (xx * np.cos(theta) + yy * np.sin(theta))

    lin = lin * gains * exposure * field[..., None]
    lin = lin * gains ** (-phone.wb)                                  # partial auto-WB
    lin = np.clip(lin @ phone.matrix.T, 0, 1)
    lin = np.clip(lin + rng.normal(0, 1, lin.shape) * phone.noise * np.sqrt(lin + 0.02), 0, 1)
    enc = lin ** (1 / phone.gamma)
    enc = np.stack([gaussian_filter(enc[..., c], phone.blur) for c in range(3)], -1)
    return np.round(np.clip(enc, 0, 1) * 255).astype(np.uint8)
