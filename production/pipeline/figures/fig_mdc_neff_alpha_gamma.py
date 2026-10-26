#!/usr/bin/env python
"""Paper figure `mdc_neff_alpha_gamma.pdf` (fig:Neff_plot).

Effective sample size of the selection-efficiency estimator vs alpha (top) and
gamma (bottom), over 5000 prior draws of Lambda: direct injection reweighting
vs GMM draws at N_samp = 100,000 / 300,000, with the N_eff = 4 N_obs threshold.

Pure plotting from the cache written by
    make_diagnostic_plots.py ... --make-neff-scan-plot --xi-gmm-sizes 100000 300000
        --neff-results-cache sim_cat6_sharp_w8/paper_diagnostics/neff_cache_cat000_100k.npz
(extracted from make_neff_alpha_gamma_plot(); see pipeline/README.md).

    python fig_mdc_neff_alpha_gamma.py [--cache ...] [--out mdc_neff_alpha_gamma.pdf]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

NOBS = 69
SIZES = (100000, 300000)
PARAMS = ("alpha", "gamma")
LATEX = {"alpha": r"$\alpha$", "gamma": r"$\gamma$"}
BOUNDS = {"alpha": (0.0, 6.0), "gamma": (0.0, 10.0)}   # inference prior box


def tex_int(n):
    return f"{n:,}".replace(",", r"{,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=Path("sim_cat6_sharp_w8/paper_diagnostics/neff_cache_cat000_100k.npz"))
    ap.add_argument("--out", type=Path, default=Path("mdc_neff_alpha_gamma.pdf"))
    args = ap.parse_args()

    sns.set_context("talk", font_scale=1.3)
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    matplotlib.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], "axes.unicode_minus": False,
        "axes.labelsize": 20.0, "legend.fontsize": 16.0,
    })

    d = np.load(args.cache, allow_pickle=False)
    neff_inj = d["neff_inj"]
    neff_gmm = {s: d[f"neff_gmm_{s}"] for s in SIZES}

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    kw = dict(s=10, alpha=0.45)
    for ax, p in zip(axes, PARAMS):
        x = d[f"samples_{p}"]
        ax.scatter(x, neff_inj, color="C0", label=r"$\mathrm{reweighting\ injections}$", **kw)
        for color, s in zip(("C1", "C2"), SIZES):
            ax.scatter(x, neff_gmm[s], color=color,
                       label=rf"$\mathrm{{GMM}}\ (N_{{\mathrm{{samp}}}}={tex_int(s)})$", **kw)
        ax.axhline(4 * NOBS, color="black", ls="--", lw=1.2,
                   label=r"$\mathrm{Threshold}$", zorder=5)
        ax.set_yscale("log")
        ax.set_xlim(BOUNDS[p][0] - 0.3, BOUNDS[p][1] + 0.3)
        ax.set_xlabel(LATEX[p])
        ax.set_ylabel(r"$N_{\rm eff}$")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        ax.legend(framealpha=0.8, loc="lower right")
        ax.tick_params(labelsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
