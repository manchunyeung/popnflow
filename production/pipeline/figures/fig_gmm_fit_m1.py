#!/usr/bin/env python
"""Paper figure `gmm_fit_m1.pdf` (fig:KS_AD_tests, appendix).

Goodness of fit of the stored per-event GMMs (sim_cat6_sharp_w8/event_gmms.h5)
to their own PE samples, detector-frame primary mass only, across the 276 pool
events: two-sample KS statistic D_GMM (top) and Anderson-Darling A^2 (bottom).
Dashed/dotted lines are the null median / 95th percentile:
  KS  - Kolmogorov distribution, rescaled by sqrt(n m/(n+m)) with n = m = n_test
  AD  - simulated null (null_calibration.npz); its median is negative (-0.31)
In-sample diagnostic: the comparison PE samples were also used to fit the GMMs.

Pure plotting from the per-event CSV written by
    plotting/rerun_ks_ad.py --mode stored       (gmm_ks_ad/ks_ad_stored.csv)
and the null from gmm_ks_ad/calibrate_thresholds.py (null_calibration.npz).
Extracted from replot_ks_ad.py --mode stored (m1_det, null references on).

    python fig_gmm_fit_m1.py [--csv ...] [--null-npz ...] [--out gmm_fit_m1.pdf]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FS = 13.0      # base font size
BINS = 15
KOLMOGOROV_MEDIAN, KOLMOGOROV_95 = 0.828, 1.358


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("gmm_ks_ad/ks_ad_stored.csv"))
    ap.add_argument("--null-npz", type=Path, default=Path("gmm_ks_ad/null_calibration.npz"))
    ap.add_argument("--out", type=Path, default=Path("gmm_fit_m1.pdf"))
    args = ap.parse_args()

    plt.rcParams["text.usetex"] = False   # usetex drops minus signs on this host
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    })
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")
    matplotlib.rcParams.update({
        "axes.labelsize": FS, "axes.titlesize": FS, "legend.fontsize": FS * 0.8,
        "xtick.labelsize": FS * 0.7, "ytick.labelsize": FS * 0.7,
    })

    df = pd.read_csv(args.csv)
    ks = df["ks_D_m1_det"].to_numpy(float)
    ad = df["ad_A2_m1_det"].to_numpy(float)

    # null references
    n_test = float(np.median(df["n_test"]))
    scale = np.sqrt(n_test * n_test / (2.0 * n_test))
    z = np.load(args.null_npz)
    nulls = {"ks": (KOLMOGOROV_MEDIAN / scale, KOLMOGOROV_95 / scale),
             "ad": (float(np.median(z["ad"])), float(np.percentile(z["ad"], 95)))}

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharey="row", squeeze=False)
    for ax, values, color, xlabel, kind in (
            (axes[0, 0], ks, "steelblue", r"$D_{\rm GMM}$", "ks"),
            (axes[1, 0], ad, "darkorange", r"$A^2$", "ad")):
        ax.hist(values[np.isfinite(values)], bins=BINS, color=color,
                edgecolor="white", linewidth=0.5)
        med, p95 = nulls[kind]
        ax.axvline(med, color="k", ls="--", lw=1.5, label="null median")
        ax.axvline(p95, color="k", ls=":", lw=1.5, label="null 95th")
        ax.legend(fontsize=FS * 0.6, frameon=False)
        ax.set_xlabel(xlabel, fontsize=FS * 0.8)
        ax.set_ylabel("events", fontsize=FS * 0.85)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(f"{len(df)} events; KS D median {np.median(ks):.4f} (null {nulls['ks'][0]:.4f}); "
          f"AD A^2 median {np.median(ad):+.4f} (null {nulls['ad'][0]:+.4f})")


if __name__ == "__main__":
    main()
