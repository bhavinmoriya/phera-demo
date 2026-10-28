"""Gradio demo.   uv run python -m phera.app"""
from __future__ import annotations
from pathlib import Path
import gradio as gr
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .baseline import PersonalBaseline, simulate_user
from .calibrate import features
from .camera import LIGHTINGS, PHONES, capture
from .chart import PH_MAX, PH_MIN, PH_STEP, ph_to_srgb
from .model import PhModel, build_dataset, quantise

CACHE = Path(".cache/models.joblib")
DISCLAIMER = ("**Research demo on synthetic data, not a medical device.** The pH-to-colour curve and the cameras are "
              "simulated; real strips and real phone photos are needed before any claim about accuracy.")


def load_models():
    if CACHE.exists():
        return joblib.load(CACHE)
    d = build_dataset(3000, seed=0)
    seen = d["phone"] < 4                                  # phones E and F are never seen in training
    models = {k: PhModel(0.1).fit(d[k][seen], d["ph"][seen], d["phone"][seen]) for k in ("raw", "cal")}
    CACHE.parent.mkdir(exist_ok=True)
    joblib.dump(models, CACHE)
    return models


MODELS = None


def read_strip(ph, phone_name, light_name, seed):
    global MODELS
    MODELS = MODELS or load_models()
    phone = next(p for p in PHONES if p.name == phone_name)
    img = capture(float(ph), phone, LIGHTINGS[light_name], np.random.default_rng(int(seed)))
    raw, cal, rgb = features(img)
    out = {}
    for k, x in (("raw", raw), ("cal", cal)):
        mu, lo, hi = MODELS[k].predict(x)
        out[k] = (float(mu[0]), float(lo[0]), float(hi[0]))
    pad = img[100:140, 140:180].reshape(-1, 3).mean(0) / 255
    fig, ax = plt.subplots(figsize=(4.2, 1.6))
    for i, (lab, c) in enumerate([("true", ph_to_srgb(ph)), ("phone saw", pad), ("corrected", rgb)]):
        ax.add_patch(plt.Rectangle((i, 0), 0.95, 1, color=np.clip(c, 0, 1)))
        ax.text(i + 0.47, -0.18, lab, ha="center", fontsize=9)
    ax.set_xlim(0, 3); ax.set_ylim(-0.3, 1); ax.axis("off"); fig.tight_layout()
    seen = "seen in training" if phone.seen_in_training else "**never seen in training**"
    md = (f"**True pH {ph:.2f}**: {phone_name} ({seen}), {light_name}\n\n"
          f"| pipeline | estimate | rounded to step | 90% interval |\n|---|---|---|---|\n"
          f"| raw colour | {out['raw'][0]:.2f} | {float(quantise(out['raw'][0])):.2f} | {out['raw'][1]:.2f} to {out['raw'][2]:.2f} |\n"
          f"| **calibrated** | **{out['cal'][0]:.2f}** | **{float(quantise(out['cal'][0])):.2f}** | {out['cal'][1]:.2f} to {out['cal'][2]:.2f} |\n\n"
          f"{DISCLAIMER}")
    return img, fig, md


def baseline_demo(shift, meas_sd, seed):
    truth, y = simulate_user(shift=float(shift), meas_sd=float(meas_sd), seed=int(seed))
    pb = PersonalBaseline(meas_sd=float(meas_sd))
    hist = [pb.update(v) for v in y]
    m = np.array([h["mean"] for h in hist]); s = np.array([h["sd"] for h in hist])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(truth, "k--", lw=1, label="true baseline")
    ax.scatter(range(len(y)), y, s=14, c="#888", label="readings")
    ax.plot(m, c="#0066cc", label="learned baseline")
    ax.fill_between(range(len(y)), m - 2 * s, m + 2 * s, color="#0066cc", alpha=0.2)
    for i, h in enumerate(hist):
        if h["change"]:
            ax.axvline(i, c="#c0392b", lw=1.5, label="change flagged")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=8)
    ax.set_xlabel("day"); ax.set_ylabel("pH"); fig.tight_layout()
    flags = [i for i, h in enumerate(hist) if h["change"]]
    return fig, (f"Change flagged on day(s): {flags}" if flags else "No change flagged.")


def build():
    with gr.Blocks(title="pHera demo: colorimetric pH from phone photos") as demo:
        gr.Markdown("# Colorimetric pH from a phone photo (synthetic demo)\n" + DISCLAIMER)
        with gr.Tab("Read a strip"):
            with gr.Row():
                with gr.Column():
                    ph = gr.Slider(PH_MIN, PH_MAX, 4.5, step=PH_STEP, label="True pH")
                    phone = gr.Dropdown([p.name for p in PHONES], value="Phone E", label="Phone (E and F are unseen)")
                    light = gr.Dropdown(list(LIGHTINGS), value="warm indoor", label="Lighting")
                    seed = gr.Number(0, label="Random seed", precision=0)
                    btn = gr.Button("Take photo and read", variant="primary")
                with gr.Column():
                    img = gr.Image(label="Simulated photo (card with 8 reference patches x2 + pad)")
                    sw = gr.Plot(label="Pad colour")
            res = gr.Markdown()
            btn.click(read_strip, [ph, phone, light, seed], [img, sw, res])
        with gr.Tab("Personal baseline"):
            gr.Markdown("A Bayesian local-level model starts from a population prior and learns one person's baseline; "
                        "two consecutive outliers in the same direction flag a meaningful change.")
            with gr.Row():
                shift = gr.Slider(0, 2, 1.0, step=0.1, label="True shift on day 25 (pH)")
                nsd = gr.Slider(0.1, 0.5, 0.25, step=0.05, label="Reading noise (pH sd)")
                sd2 = gr.Number(0, label="Seed", precision=0)
            b2 = gr.Button("Simulate")
            fig2, txt = gr.Plot(), gr.Markdown()
            b2.click(baseline_demo, [shift, nsd, sd2], [fig2, txt])
    return demo


if __name__ == "__main__":
    build().launch(share=True)
