"""Leave-one-phone-out benchmark: raw colour vs reference-calibrated colour.

    uv run python -m phera.evaluate --n 3000
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .camera import PHONES
from .model import PhModel, _reg, build_dataset, quantise


def lopo(X, y, groups):
    pred = np.empty_like(y, float)
    for g in np.unique(groups):
        m = groups == g
        pred[m] = _reg().fit(X[~m], y[~m]).predict(X[m])
    return pred


def metrics(y, p):
    e = np.abs(y - p)
    return dict(mae=float(e.mean()), p90=float(np.quantile(e, 0.9)),
                within_0_25=float((e <= 0.25 + 1e-9).mean()),
                exact_step=float((np.abs(y - quantise(p)) < 1e-9).mean()))


def main(n=3000, out="results", seed=0):
    out = Path(out); out.mkdir(exist_ok=True)
    d = build_dataset(n, seed)
    y, g = d["ph"], d["phone"]
    res = {}
    for key, label in [("raw", "raw colour"), ("cal", "calibrated")]:
        p = lopo(d[key], y, g)
        res[label] = dict(overall=metrics(y, p),
                          by_lighting={ln: metrics(y[d["light"] == i], p[d["light"] == i])
                                       for i, ln in enumerate(d["light_names"])},
                          by_phone={PHONES[i].name: metrics(y[g == i], p[g == i]) for i in range(len(PHONES))})
        res[label]["_pred"] = p

    # conformal coverage on genuinely unseen phones (fit on phones 0-3, test on 4-5)
    seen, unseen = g < 4, g >= 4
    pm = PhModel(alpha=0.1).fit(d["cal"][seen], y[seen], g[seen])
    mu, lo, hi = pm.predict(d["cal"][unseen])
    res["conformal_unseen_phones"] = dict(target=0.9, half_width=pm.q,
                                          coverage=float(((y[unseen] >= lo) & (y[unseen] <= hi)).mean()),
                                          mae=float(np.abs(y[unseen] - mu).mean()))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ln = d["light_names"]
    for j, (label, c) in enumerate([("raw colour", "#c0392b"), ("calibrated", "#0066cc")]):
        ax[0].bar(np.arange(4) + 0.38 * j, [res[label]["by_lighting"][l]["mae"] for l in ln], 0.38, label=label, color=c)
        ax[1].bar(np.arange(6) + 0.38 * j, [res[label]["by_phone"][PHONES[i].name]["mae"] for i in range(6)], 0.38, label=label, color=c)
    ax[0].set_xticks(np.arange(4) + 0.19, ln, fontsize=8); ax[0].set_title("MAE by lighting (unseen phone)")
    ax[1].set_xticks(np.arange(6) + 0.19, [p.name[-1] for p in PHONES]); ax[1].set_title("MAE by held-out phone")
    for a in ax[:2]: a.set_ylabel("MAE (pH units)"); a.legend()
    for label, c in [("raw colour", "#c0392b"), ("calibrated", "#0066cc")]:
        ax[2].scatter(y + np.random.default_rng(1).normal(0, 0.02, len(y)), res[label]["_pred"], s=3, alpha=0.25, c=c, label=label)
    ax[2].plot([3.5, 7], [3.5, 7], "k--", lw=1); ax[2].set_xlabel("true pH"); ax[2].set_ylabel("predicted pH")
    ax[2].set_title("Leave-one-phone-out predictions"); ax[2].legend(markerscale=4)
    fig.tight_layout(); fig.savefig(out / "calibration_effect.png", dpi=140)

    for label in ("raw colour", "calibrated"):
        res[label].pop("_pred")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    o = res
    md = ["| pipeline | MAE | 90th pct error | within 0.25 pH | exact step |", "|---|---|---|---|---|"]
    for label in ("raw colour", "calibrated"):
        m = o[label]["overall"]
        md.append(f"| {label} | {m['mae']:.3f} | {m['p90']:.3f} | {m['within_0_25']:.1%} | {m['exact_step']:.1%} |")
    c = o["conformal_unseen_phones"]
    md += ["", f"Conformal 90% interval on unseen phones: half-width ±{c['half_width']:.2f} pH, empirical coverage {c['coverage']:.1%}."]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="results"); a = ap.parse_args(); main(a.n, a.out)
