#!/usr/bin/env python
"""
replot_ks_ad.py
===============
Redraw the KS / Anderson-Darling summary histograms **from the cached CSV**
(`ks_ad_{refit,stored}.csv`) instead of re-running the test.

`rerun_ks_ad.py` reloads 276x5000 PE samples, refits a GMM per event and draws
GMM samples for 64 random slices -- minutes of work -- purely to end up calling
`test_gmm_fit.make_summary_histograms`. Every number that figure plots is
already written to the per-event CSV, so for a styling change there is no reason
to pay for the statistics again. This script reproduces the same 2-row grid
(KS D on top, AD A^2 below) reading only the CSV.

ONE PANEL IS NOT A PIXEL-FOR-PIXEL MATCH, by necessity. The original top-right
panel plots the *raw* sliced statistic `sliced["ks_D_median"]`; the CSV stores
only `excess_sliced = ks_D_median / ks_D_baseline_sliced`, and the baseline is
not written to disk. So the sliced KS panel here is the excess ratio and is
labelled as such. Under exact PE/GMM agreement it sits at 1/sqrt(2) ~ 0.71, not
at 1, because the GMM test is N-vs-N while the baseline split-half is
N/2-vs-N/2 (see README, "READ THE SCALE BEFORE READING THE NUMBERS"). The
per-dimension KS panels are the raw D and are identical to the original.

Typesetting: mathtext + Computer Modern by default. Real LaTeX
(`text.usetex=True`) SILENTLY DROPS MINUS SIGNS on this host and the AD A^2
axis is negative for most events, so `usetex` is offered only with a warning.
The plain-text fallback is also wrong in CM -- cmr10 has no underscore, so a
literal "D_GMM" renders as "D.GMM" (this is the bug visible in the Aug-31
ks_ad_refit.pdf); the labels are therefore always emitted in math form.

Usage:
    python replot_ks_ad.py                        # refit, m1_det only + null refs
    python replot_ks_ad.py --mode stored
    python replot_ks_ad.py --dims m1_det,m2_det,dL --sliced --titles \
        --out ks_ad_refit_alldims
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DIM_LABEL = {"m1_det": r"$m_1^{\rm det}$",
             "m2_det": r"$m_2^{\rm det}$",
             "dL": r"$d_L$"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["refit", "stored"], default="refit",
                    help="which cached CSV to read: ks_ad_<mode>.csv")
    ap.add_argument("--csv", type=Path, default=None,
                    help="explicit CSV path (overrides --mode)")
    ap.add_argument("--dims", default="m1_det",
                    help="comma-separated marginals to show, in order")
    ap.add_argument("--sliced", action="store_true",
                    help="add the sliced (joint) column. Off by default: the "
                         "raw sliced D is not in the CSV, so that column can "
                         "only be shown as the excess ratio (see the module "
                         "docstring).")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--fontsize", type=float, default=13.0)
    ap.add_argument("--titles", action="store_true",
                    help="label each column above the top panel")
    ap.add_argument("--no-null-ref", dest="null_ref", action="store_false",
                    help="overlay the calibrated null on each panel: the "
                         "Kolmogorov median/95%% for the raw KS D, and the "
                         "simulated-null median/95%% from null_calibration.npz "
                         "for the AD A^2 and the sliced excess. See README, "
                         "'Where does excess_threshold = 3.0 come from?'. "
                         "On by default; pass --no-null-ref to suppress.")
    ap.add_argument("--zero-line", action="store_true",
                    help="draw an unlabelled A^2=0 rule on the AD panels. Off "
                         "by default: the original figure drew it and called it "
                         "'baseline', but zero is the null MEAN and the null "
                         "MEDIAN is -0.31.")
    ap.add_argument("--null-npz", type=Path, default=None,
                    help="simulated null (default: null_calibration.npz here)")
    ap.add_argument("--tex-mode", choices=["mathtext", "usetex"],
                    default="mathtext",
                    help="'usetex' drops minus signs on this host -- the AD "
                         "A^2 axis would be corrupted. Use mathtext.")
    ap.add_argument("--out", default=None,
                    help="output stem (default: ks_ad_<mode>); .pdf and .png "
                         "are both written")
    args = ap.parse_args()

    csv = args.csv or HERE / f"ks_ad_{args.mode}.csv"
    if not csv.exists():
        raise SystemExit(f"no cached CSV at {csv} -- run rerun_ks_ad.py --mode "
                         f"{args.mode} first")
    df = pd.read_csv(csv)
    print(f"[cache] {csv.name}: {len(df)} events, K {df['k'].min()}-{df['k'].max()} "
          f"(median {int(df['k'].median())}), flagged={int(df['flagged'].sum())}")

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    for d in dims:
        for c in (f"ks_D_{d}", f"ad_A2_{d}"):
            if c not in df.columns:
                raise SystemExit(f"{csv.name} has no column {c}")

    # (values, xlabel) per panel, top row then bottom row
    ks_panels = [(df[f"ks_D_{d}"].to_numpy(float), r"$D_{\rm GMM}$", "ks_D")
                 for d in dims]
    ad_panels = [(df[f"ad_A2_{d}"].to_numpy(float), r"$A^2$", "ad") for d in dims]
    col_titles = [DIM_LABEL.get(d, d) for d in dims]
    if args.sliced:
        # NB: excess ratio, not raw D -- the raw sliced median is not cached.
        ks_panels.append((df["excess_sliced"].to_numpy(float),
                          r"$D_{\rm GMM}/D_{\rm base}$", "excess"))
        ad_panels.append((df["ad_A2_sliced"].to_numpy(float), r"$A^2$", "ad"))
        col_titles.append("sliced (joint)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if args.tex_mode == "usetex":
        print("[warn] usetex drops minus signs on this host; the AD A^2 axis "
              "will be wrong.")
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    else:
        plt.rcParams["text.usetex"] = False
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")
    fs = args.fontsize
    matplotlib.rcParams["axes.labelsize"] = fs
    matplotlib.rcParams["axes.titlesize"] = fs
    matplotlib.rcParams["legend.fontsize"] = fs * 0.8
    matplotlib.rcParams["xtick.labelsize"] = fs * 0.7
    matplotlib.rcParams["ytick.labelsize"] = fs * 0.7

    # ── calibrated null references ───────────────────────────────────────────
    # ks_D    : Kolmogorov distribution. D* = sqrt(n m /(n+m)) D is asymptotically
    #           Kolmogorov-distributed under H0, median 0.828, 95% 1.358. The GMM
    #           draw count equals n_test, so n = m = n_test and no simulation is
    #           needed -- this is the one panel with a closed-form null.
    # ad      : the normalized Scholz-Stephens statistic has mean 0 / var 1 under
    #           H0 but is right-skewed, so its NULL MEDIAN IS NEGATIVE (-0.31),
    #           not 0. A^2 < 0 is ordinary null behaviour, not a good fit.
    # excess  : denominator is a median of 8 half-splits, so there is no analytic
    #           null; null_calibration.npz simulates it (median 0.710, 95% 1.291).
    null = {}
    if args.null_ref:
        n_test = float(np.median(df["n_test"]))
        scale = np.sqrt(n_test * n_test / (2.0 * n_test))
        null["ks_D"] = (0.828 / scale, 1.358 / scale, "Kolmogorov")
        npz = args.null_npz or HERE / "null_calibration.npz"
        if npz.exists():
            z = np.load(npz)
            null["ad"] = (float(np.median(z["ad"])),
                          float(np.percentile(z["ad"], 95)), "simulated null")
            null["excess"] = (float(np.median(z["excess"])),
                              float(np.percentile(z["excess"], 95)),
                              "simulated null")
        else:
            print(f"[warn] {npz} missing -- AD/excess null reference skipped")
        for k, (med, p95, src) in null.items():
            print(f"[null] {k:7s} median {med:+.4f}  95th {p95:+.4f}  ({src})")

    n = len(ks_panels)
    fig, axes = plt.subplots(2, n, figsize=(max(6.5, 4.2 * n), 7),
                             sharey="row", squeeze=False)

    def add_null(ax, kind):
        ref = null.get(kind)
        if ref is None:
            return False
        med, p95, _src = ref
        ax.axvline(med, color="k", ls="--", lw=1.5, label="null median")
        ax.axvline(p95, color="k", ls=":", lw=1.5, label="null 95th")
        return True

    for j, ((ks_v, ks_lab, ks_kind), (ad_v, ad_lab, ad_kind)) in enumerate(
            zip(ks_panels, ad_panels)):
        ax = axes[0, j]
        ax.hist(ks_v[np.isfinite(ks_v)], bins=args.bins, color="steelblue",
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel(ks_lab, fontsize=fs * 0.8)
        if args.titles:
            ax.set_title(col_titles[j], fontsize=fs * 0.85)
        if j == 0:
            ax.set_ylabel("events", fontsize=fs * 0.85)
        if add_null(ax, ks_kind):
            ax.legend(fontsize=fs * 0.6, frameon=False)

        ax = axes[1, j]
        ax.hist(ad_v[np.isfinite(ad_v)], bins=args.bins, color="darkorange",
                edgecolor="white", linewidth=0.5)
        # The old figure drew A^2=0 here and called it "baseline". It is not:
        # zero is the null MEAN of a right-skewed statistic whose null MEDIAN is
        # -0.31 (the dashed line below). Off by default, and never in the legend.
        if args.zero_line:
            ax.axvline(0.0, color="0.35", ls="-", lw=1.2)
        if add_null(ax, ad_kind):
            ax.legend(fontsize=fs * 0.6, frameon=False)
        ax.set_xlabel(ad_lab, fontsize=fs * 0.8)
        if j == 0:
            ax.set_ylabel("events", fontsize=fs * 0.85)

    fig.tight_layout()
    stem = args.out or f"ks_ad_{args.mode}"
    for ext in ("pdf", "png"):
        p = HERE / f"{stem}.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[out] {p}")
    plt.close(fig)

    for (v, _l, _k), name in zip(ks_panels, dims + (["sliced"] if args.sliced else [])):
        f = v[np.isfinite(v)]
        print(f"[summary] KS {name:8s} median {np.median(f):.4f}  "
              f"90th {np.percentile(f, 90):.4f}  max {f.max():.4f}")
    for (v, _l, _k), name in zip(ad_panels, dims + (["sliced"] if args.sliced else [])):
        f = v[np.isfinite(v)]
        print(f"[summary] AD {name:8s} A^2 median {np.median(f):+.4f}  "
              f"frac<0 {np.mean(f < 0):.2f}")


if __name__ == "__main__":
    main()
