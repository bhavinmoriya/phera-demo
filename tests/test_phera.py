import numpy as np
from phera.baseline import PersonalBaseline, simulate_user
from phera.calibrate import extract, features, fit_calibration, srgb_to_lab
from phera.camera import LIGHTINGS, PHONES, capture
from phera.chart import PH_LEVELS, REF_SRGB, ph_to_srgb, render_ideal
from phera.model import quantise


def test_colour_curve_monotone_distinct():
    cols = np.array([ph_to_srgb(p) for p in PH_LEVELS])
    assert np.all(np.linalg.norm(np.diff(cols, axis=0), axis=1) > 0.01)


def test_capture_shape_dtype_and_determinism():
    a = capture(5.0, PHONES[0], LIGHTINGS["daylight"], np.random.default_rng(1))
    b = capture(5.0, PHONES[0], LIGHTINGS["daylight"], np.random.default_rng(1))
    assert a.shape == (240, 320, 3) and a.dtype == np.uint8 and np.array_equal(a, b)


def test_calibration_recovers_reference_patches():
    img = capture(5.0, PHONES[2], LIGHTINGS["warm indoor"], np.random.default_rng(0))
    _, patches = extract(img)
    err = np.abs(fit_calibration(patches).apply(patches) - REF_SRGB).mean()
    assert err < np.abs(patches - REF_SRGB).mean()


def test_calibration_reduces_device_spread():
    """Same pH through different phones/lights: calibrated colours agree better than raw."""
    raw, cal = [], []
    for i, p in enumerate(PHONES):
        for j, l in enumerate(LIGHTINGS.values()):
            r, c, _ = features(capture(5.5, p, l, np.random.default_rng(10 * i + j)))
            raw.append(r); cal.append(c)
    assert np.std(cal, 0).sum() < np.std(raw, 0).sum()


def test_lab_white():
    assert np.allclose(srgb_to_lab(np.array([1, 1, 1.0]))[0], [100, 0, 0], atol=0.1)


def test_quantise():
    assert quantise(4.62) == 4.5 and quantise(9) == 7.0 and quantise(1) == 3.5


def test_baseline_flags_shift_not_noise():
    _, y = simulate_user(shift=1.0, seed=3)
    pb = PersonalBaseline(); flags = [i for i, v in enumerate(y) if pb.update(v)["change"]]
    assert flags and 25 <= flags[0] <= 29
    _, y0 = simulate_user(shift=0.0, seed=3)
    pb = PersonalBaseline(); assert not any(pb.update(v)["change"] for v in y0)
