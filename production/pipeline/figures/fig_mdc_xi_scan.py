#!/usr/bin/env python
"""Paper figure `mdc_xi_scan.pdf` (fig:xi_plot).

Selection efficiency xi(Lambda) vs alpha (top) and gamma (bottom) over 5000
prior draws: direct injection reweighting vs GMM draws at N_samp = 100,000 /
300,000.

Pure plotting from the cache written by
    make_diagnostic_plots.py ... --make-xi-scan-plot --xi-gmm-sizes 100000 300000
        --xi-results-cache sim_cat6_sharp_w8/paper_diagnostics/xi_cache_cat000_100k.npz
(extracted from make_xi_scan_plot(); see pipeline/README.md).

    python fig_mdc_xi_scan.py [--cache ...] [--out mdc_xi_scan.pdf]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SIZES = (100000, 300000)
PARAMS = ("alpha", "gamma")
LATEX = {"alpha": r"$\alpha$", "gamma": r"$\gamma$"}


def tex_int(n):
    return f"{n:,}".replace(",", r"{,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path,
                    default=Path("sim_cat6_sharp_w8/paper_diagnostics/xi_cache_cat000_100k.npz"))
    ap.add_argument("--out", type=Path, default=Path("mdc_xi_scan.pdf"))
    args = ap.parse_args()

    sns.set_context("talk", font_scale=1.3)
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    C = sns.color_palette("colorblind")
    matplotlib.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"], "axes.unicode_minus": False,
        "axes.labelsize": 20.0, "legend.fontsize": 16.0,
    })

    d = np.load(args.cache, allow_pickle=False)
    xi_inj = d["xi_inj"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    kw = dict(s=10, alpha=0.4, edgecolors="none")
    for ax, p in zip(axes, PARAMS):
        x = d[f"samples_{p}"]
        ax.scatter(x, xi_inj, color=C[0], label=r"$\mathrm{reweighting\ injections}$", **kw)
        for s, col in zip(SIZES, (C[1], C[2])):
            ax.scatter(x, d[f"xi_gmm_{s}"], color=col,
                       label=rf"$\mathrm{{GMM}}\ (N_{{\mathrm{{samp}}}}={tex_int(s)})$", **kw)
        ax.set_yscale("log")
        ax.set_xlabel(LATEX[p])
        ax.set_ylabel(r"$\xi(\Lambda)$")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
        ax.legend(framealpha=0.8, loc="best")
        ax.tick_params(labelsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
