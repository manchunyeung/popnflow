#!/usr/bin/env python
"""Paper figure `mdc_variance_Nsamp.pdf` (fig:var).

Variance of the catalog log-likelihood sum vs the per-event sample budget
N_samp for PE / Ensemble / Mixture (median and 16-84% over 500 prior draws,
each with a dashed power law fitted to its median), the PE sample ceiling
N_PE = 5000, and the bootstrap fit-variance floor B (black line; grey band =
16-84% of B over Lambda).

Pure plotting from two caches (sim_cat6 cat000):
  --variance-cache  make_diagnostic_plots.py --make-variance-plot --n-pool 1400000
                    --variance-nsamp-max 20000  (variance_cache_3method_cat000_2e4.npz)
  --fitvar-cache    make_diagnostic_plots.py --make-fitvar-plot  (fitvar_cache_cat000.npz)
Extracted from plot_variance_budget_left_numeric.py with the settings the paper
used: --band-mode fit-line --sigma-band none (no hatched jackknife band).

    python fig_mdc_variance_Nsamp.py [--variance-cache ...] [--fitvar-cache ...] [--out ...]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DIAG = Path("sim_cat6_sharp_w8/paper_diagnostics")
MIN_SNR = 3.0   # B must exceed MIN_SNR x its OLS error to be drawn as resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variance-cache", type=Path, default=DIAG / "variance_cache_3method_cat000_2e4.npz")
    ap.add_argument("--fitvar-cache", type=Path, default=DIAG / "fitvar_cache_cat000.npz")
    ap.add_argument("--out", type=Path, default=Path("mdc_variance_Nsamp.pdf"))
    args = ap.parse_args()

    sns.set_context("talk", font_scale=1.3)
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    C = sns.color_palette("colorblind")
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    })

    dv = np.load(args.variance_cache, allow_pickle=False)
    fv = np.load(args.fitvar_cache, allow_pickle=False)

    # fit-variance floor B over the Lambda draws
    B_hat = np.asarray(fv["B_hat"])
    B_err = np.asarray(fv["B_err"])
    n_train = int(np.asarray(fv["n_train"]).item())   # PE samples per event = PE ceiling
    snr = np.nanmedian(B_hat) / max(np.nanmedian(B_err), 1e-300)
    resolved = np.nanmedian(B_hat) > 0 and snr > MIN_SNR
    if not resolved:
        raise SystemExit(f"fit variance not resolved (snr {snr:.2f}); the paper figure assumes it is")
    b_med = float(np.nanmedian(B_hat))
    b_lo, b_hi = (float(np.nanpercentile(B_hat, q)) for q in (16, 84))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5), constrained_layout=True)
    slopes = {}
    for name, color, marker in (("PE", C[0], "s"), ("Ensemble", C[1], "o"), ("Mixture", C[2], "^")):
        x = np.asarray(dv[f"{name}_x"])
        order = np.argsort(x)
        x = x[order]
        med, lo, hi = (np.asarray(dv[f"{name}_{k}"])[order] for k in ("med", "lo", "hi"))

        # dashed measured power law sigma^2 = A N^b, fitted to the median
        b, lnA = np.polyfit(np.log(x), np.log(med), 1)
        slopes[name] = b
        Ns = np.logspace(np.log10(max(x.min(), 10)), np.log10(x.max()), 300)
        ax.plot(Ns, np.exp(lnA) * Ns ** b, color=color, ls="--", lw=1.5, alpha=0.85, zorder=1)

        ax.errorbar(x, med, yerr=np.vstack([med - lo, hi - med]), marker=marker, ls="-",
                    color=color, lw=2, ms=5, capsize=3, elinewidth=1.2,
                    label=rf"$\mathrm{{{name}}}$")

    ax.axhline(b_med, color="k", ls="-", lw=2.0, alpha=0.9, label=r"$\mathrm{fit\ variance}$")
    ax.axhspan(b_lo, b_hi, color="k", alpha=0.12)
    ax.axvline(n_train, color="k", ls=":", lw=2, alpha=0.7, label=fr"$N_{{\rm PE}}={n_train}$")
    ax.axhline(1.0, color="0.5", ls="--", lw=1.0, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N_{\rm samp}$")
    ax.set_ylabel(r"$\sigma^2_{\ln\hat{\mathcal{L}}}$")
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(loc="upper right", fontsize=15.0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved -> {args.out}")
    print(f"fit variance B: median {b_med:.4g}, 16-84% over Lambda [{b_lo:.4g}, {b_hi:.4g}], snr {snr:.2f}")
    print("fitted slopes b (sigma^2 ~ N^b): " + ", ".join(f"{k}={v:+.3f}" for k, v in slopes.items()))


if __name__ == "__main__":
    main()
