#!/usr/bin/env python
"""Paper figure `pull_boxplot.pdf` (fig:pull_box).

Pull (posterior mean - truth) / posterior sd of the eight hyperparameters across
the 50 fixed-Lambda sim_cat6 catalogs, for PE (met1), Ensemble (met3) and
Mixture (met4). Grey bands mark +-1, 2, 3 posterior sd. Outliers are hidden
(showfliers=False).

Pure plotting from the per-(catalog, method, parameter) CSV written by
    pp_bias_diagnostics.py extract ... --out sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv
(met1 from inference/, met3/met4 from the nlive=1000 boxfix+cholfix runs; see
pipeline/README.md). Extracted from pp_bias_diagnostics.make_pull_boxplot()
with --mathtext (text.usetex drops minus signs on this host).

    python fig_pull_boxplot.py [--csv ...] [--out pull_boxplot.pdf]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PARAMS = ["gamma", "m_min", "m_max", "alpha", "beta", "mu", "sigma", "f1"]
PARAM_LABELS = {
    "gamma": r"$\gamma$", "m_min": r"$m_\mathrm{min}$", "m_max": r"$m_\mathrm{max}$",
    "alpha": r"$\alpha$", "beta": r"$\beta$", "mu": r"$\mu$", "sigma": r"$\sigma$",
    "f1": r"$\lambda$",
}
METHODS = {"met1": ("PE", "#0173B2"), "met3": ("Ensemble", "#DE8F05"), "met4": ("Mixture", "#029E73")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=Path("sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv"))
    ap.add_argument("--out", type=Path, default=Path("pull_boxplot.pdf"))
    args = ap.parse_args()

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm"})
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")
    matplotlib.rcParams.update({"axes.labelsize": 20.0, "legend.fontsize": 16.0})

    df = pd.read_csv(args.csv)
    df = df[df["parameter"].isin(PARAMS)].copy()
    df["method_label"] = df["method"].map(lambda m: METHODS.get(m, (m,))[0])
    df["param_label"] = df["parameter"].map(PARAM_LABELS)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="param_label", y="pull", hue="method_label",
                order=[PARAM_LABELS[p] for p in PARAMS],
                hue_order=[lab for lab, _ in METHODS.values()],
                palette={lab: col for lab, col in METHODS.values()},
                ax=ax, showfliers=False, linewidth=1.2)
    ax.axhline(0, color="black", lw=1.0, ls="--", zorder=0)
    for s, alpha in zip((3, 2, 1), (0.06, 0.10, 0.16)):
        ax.axhspan(-s, s, color="gray", alpha=alpha, lw=0, zorder=-1)
    ax.set_ylabel(r"$(\bar{\theta}_\mathrm{mean} - \theta_\mathrm{true}) \,/\, \sigma_\mathrm{posterior}$")
    ax.set_xlabel("")
    ax.legend(title="", loc="upper right", frameon=False)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"saved -> {args.out}")
    print(df.groupby(["parameter", "method"])["pull"].median().unstack().round(2).loc[PARAMS])


if __name__ == "__main__":
    main()
