#!/usr/bin/env python
"""Paper figure `pp_plot_per_parameter.pdf` (fig:pp_plot).

Simulation-based-calibration coverage PP plot, one panel per hyperparameter:
fraction of the 160 catalogs (20 prior-drawn Lambda x 8 catalogs of 69 events)
whose central credible interval p contains the true value, for PE (met1),
Ensemble (met3_cholfix) and Mixture (met4_cholfix). Dotted line at p = 0.9.

Since the central p interval is the [(1-p)/2, (1+p)/2] posterior-quantile pair,
"truth inside" <=> |2 PIT - 1| <= p, so each curve is the ECDF of |2 PIT - 1|.

Pure plotting from the PIT files written by sbc/sbc_02_extract_pit.py
(sbc_suite/sbc_n10/pit/{met1,met3_cholfix,met4_cholfix}.npz -- the 20-block
files, NOT the later *_n40 ones). Extracted from sbc/sbc_16_coverage_pp.py:
the figure draws the observed curves only; the bootstrap bands and pass/fail
p-values that script also computes go to its CSV, not the figure.

    python fig_pp_plot_per_parameter.py [--pit-dir ...] [--out pp_plot_per_parameter.pdf]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARAMS = ["gamma", "m_min", "m_max", "alpha", "beta", "mu", "sigma", "f1"]
PARAM_TEX = {"gamma": r"$\gamma$", "m_min": r"$m_\mathrm{min}$", "m_max": r"$m_\mathrm{max}$",
             "alpha": r"$\alpha$", "beta": r"$\beta$", "mu": r"$\mu$", "sigma": r"$\sigma$",
             "f1": r"$\lambda$"}
METHODS = {"met1": ("PE", "#0173B2"),
           "met3_cholfix": ("Ensemble", "#DE8F05"),
           "met4_cholfix": ("Mixture", "#029E73")}
GRID = np.linspace(0.0, 1.0, 201)


def coverage_curve(pit, p_grid=GRID):
    """Fraction of runs whose central p interval contains the truth."""
    d = np.sort(np.abs(2.0 * np.asarray(pit, float) - 1.0))
    return np.searchsorted(d, p_grid, side="right") / len(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pit-dir", type=Path, default=Path("sbc_suite/sbc_n10/pit"))
    ap.add_argument("--out", type=Path, default=Path("pp_plot_per_parameter.pdf"))
    args = ap.parse_args()

    matplotlib.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm",
                                "font.family": "serif", "axes.unicode_minus": False})

    pits = {m: np.load(args.pit_dir / f"{m}.npz", allow_pickle=True) for m in METHODS}
    for m, d in pits.items():
        n, k = d["ok"].shape
        print(f"{m}: {n} blocks x {k} catalogs, {int(d['ok'].sum())}/{n * k} usable runs")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for i, (ax, pname) in enumerate(zip(axes.ravel(), PARAMS)):
        ax.plot(GRID, GRID, color="#0b0b0b", lw=1.1, ls="--")
        for m, (label, color) in METHODS.items():
            d = pits[m]
            a = [str(x) for x in d["param_names"]].index(pname)
            cov = coverage_curve(d["pit"][:, :, a][d["ok"]])
            ax.plot(GRID, cov, lw=2.0, color=color, label=label)
        ax.axvline(0.90, color="#999999", lw=0.8, ls=":", zorder=0)
        ax.set_title(PARAM_TEX[pname], fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, ls=":")
        ax.set_xlabel("credible interval $p$", fontsize=11)
        ax.set_ylabel("fraction", fontsize=11)
        if i == 0:
            ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"saved -> {args.out}")

    # coverage of the 90% interval, the number quoted in text
    i90 = int(np.argmin(np.abs(GRID - 0.9)))
    for m, (label, _) in METHODS.items():
        d = pits[m]
        names = [str(x) for x in d["param_names"]]
        cov90 = {p: coverage_curve(d["pit"][:, :, names.index(p)][d["ok"]])[i90] for p in PARAMS}
        print(f"{label:9s} 90% coverage: " + "  ".join(f"{p}={c:.2f}" for p, c in cov90.items()))


if __name__ == "__main__":
    main()
