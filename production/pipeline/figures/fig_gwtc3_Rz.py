#!/usr/bin/env python
"""Paper figure `gwtc3_Rz.pdf` (fig:gwtc3_mass_distributions, right).

Merger rate R(z) = R0 (1+z)^gamma inferred from GWTC-3 with the pairing model:
90% bands and medians from PE (blue), Ensemble (orange) and Mixture (pink),
plus 100 Mixture draws as faint lines. R0 from the LVK endo3 injections
(popfig.py). Inputs: the seed-69 Colab checkpoints (fig_gwtc3_corner_pos.py).

--paper-bug reproduces a slip in the script that made the paper figure
(make_plots_met4.py:934): the Ensemble band's upper edge was Mixture's 95th
percentile. Off by default, i.e. the Ensemble band is drawn correctly.
Extracted from make_plots_met4.py --run gwtc3 --seed 69 --model pairing
(gwtc3_Rz_posteriors.pdf); resampling is seeded here (--seed).

    python fig_gwtc3_Rz.py [--seed 0] [--paper-bug] [--met1/--met3/--met4 PATH] [--out gwtc3_Rz.pdf]
"""
import argparse
from pathlib import Path

import numpy as np

import popfig

RUN = "gwtc3"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("gwtc3_Rz.pdf"))
    ap.add_argument("--seed", type=int, default=0, help="posterior-resampling seed")
    ap.add_argument("--paper-bug", action="store_true",
                    help="draw the Ensemble band up to Mixture's 95th percentile, as published")
    for m in ("met1", "met3", "met4"):
        ap.add_argument(f"--{m}", type=Path, default=None)
    args = ap.parse_args()

    c = popfig.setup_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    posts = popfig.load_run(RUN, args.seed, {m: getattr(args, m) for m in ("met1", "met3", "met4")
                                             if getattr(args, m)})
    draws = popfig.rate_draws(RUN, posts, popfig.Selection(), want_m1=False)

    colors = {"met1": c[0], "met3": c[1], "met4": c[4]}
    z = popfig.ZS_PLOT
    pct = {m: np.percentile(draws[m]["Rz"], [5, 50, 95], axis=0) for m in colors}
    if args.paper_bug:
        pct["met3"] = np.array([pct["met3"][0], pct["met3"][1], pct["met4"][2]])

    plt.figure(figsize=(10, 6))
    traces = draws["met4"]["Rz"]
    for i in popfig.trace_index(len(traces)):
        plt.plot(z, traces[i], color=colors["met4"], alpha=0.1, lw=1)
    for m, col in colors.items():
        p5, p50, p95 = pct[m]
        plt.fill_between(z, p5, p95, color=col, alpha=0.3)
        plt.plot(z, p50, color=col, lw=2, label=popfig.METHOD_LABEL[m])

    plt.xlim(0, 1.5)
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(popfig.log_tex_formatter))
    plt.xlabel(r"$z$", fontsize=25)
    plt.ylabel(r"$\mathcal{R}(z) \, [\mathrm{Gpc}^{\mbox{-}3} \, \mathrm{yr}^{\mbox{-}1}]$", fontsize=25)
    plt.legend(fontsize=16)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
