#!/usr/bin/env python
"""Paper figure `mdc_Rm1.pdf` (fig:mdc_mass_distributions, left).

Differential merger rate dR/dm1 = R0 p(m1 | Lambda) for the simulated catalog
of mdc_corner_pos.pdf: 90% bands and medians from PE (blue), Ensemble (orange)
and Mixture (pink), 100 Mixture draws as faint lines, and the injected
population (dashed black) passed through the same rate normaliser.

R0 = N_obs / (epsilon V T) with epsilon from the LVK endo3 injections -- see
popfig.py's docstring for what that implies for the absolute scale.
Extracted from make_plots_met4.py --run mdc --seed 0 ({run}_Rm1_posteriors.pdf);
resampling is seeded here (--seed).

    python fig_mdc_Rm1.py [--seed 0] [--met1/--met3/--met4 PATH] [--out mdc_Rm1.pdf]
"""
import argparse
from pathlib import Path

import numpy as np

import popfig

RUN = "mdc"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("mdc_Rm1.pdf"))
    ap.add_argument("--seed", type=int, default=0, help="posterior-resampling seed")
    for m in ("met1", "met3", "met4"):
        ap.add_argument(f"--{m}", type=Path, default=None)
    args = ap.parse_args()

    c = popfig.setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    posts = popfig.load_run(RUN, args.seed, {m: getattr(args, m) for m in ("met1", "met3", "met4")
                                             if getattr(args, m)})
    sel = popfig.Selection()
    draws = popfig.rate_draws(RUN, posts, sel)
    truth = popfig.truth_curves(RUN, sel)

    colors = {"met1": c[0], "met3": c[1], "met4": c[4]}
    m1 = popfig.M1S
    plt.figure(figsize=(10, 6))
    traces = draws["met4"]["Rm1"]
    for i in popfig.trace_index(len(traces)):
        plt.plot(m1, traces[i], color=colors["met4"], alpha=0.1, lw=1)
    for m, col in colors.items():
        p5, p50, p95 = np.percentile(draws[m]["Rm1"], [5, 50, 95], axis=0)
        plt.fill_between(m1, p5, p95, color=col, alpha=0.3)
        plt.plot(m1, p50, color=col, lw=2, label=popfig.METHOD_LABEL[m])
    plt.plot(m1, truth["Rm1"], color="black", ls="--", lw=2, label=r"$\mathrm{True~Distribution}$")

    plt.xlim(5, 80)
    plt.ylim(*popfig.RUNS[RUN]["rm1_ylim"])
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(popfig.log_tex_formatter))
    plt.ylabel(r"$dR/dm_1 \, [\mathrm{Gpc}^{\mbox{-}3} \, \mathrm{yr}^{\mbox{-}1} \, M_\odot^{\mbox{-}1}]$",
               fontsize=25)
    plt.xlabel(r"$m_1 \, [M_\odot]$", fontsize=25)
    plt.legend(fontsize=16)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
