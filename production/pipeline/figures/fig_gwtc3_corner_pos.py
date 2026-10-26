#!/usr/bin/env python
"""Paper figure `gwtc3_corner_pos.pdf` (fig:gwtc3_pos).

Hyperparameter posteriors of the symmetric pairing model (smoothed power law +
peak, 10 parameters) fitted to the 69 GWTC-3 BBHs, from PE (met1, blue),
Ensemble (met3, orange) and Mixture (met4, green).

Inputs: results/data/inference/gwtc3_69met{1,3,4}.pkl -- the seed-69 Colab run
of 2026-05-05 (inference/gwtc3_mdc_69clean.ipynb). It predates the Cholesky,
PE-box, selection-Jacobian and spin p_draw fixes (pipeline/README.md, caveats).
Extracted from make_plots_met4.py --run gwtc3 --seed 69 --model pairing
(its gwtc3_69_corner_all.pdf); resampling is seeded here (--seed).

    python fig_gwtc3_corner_pos.py [--seed 0] [--met1/--met3/--met4 PATH] [--out ...]
"""
import argparse
from pathlib import Path

import numpy as np

import popfig

RUN = "gwtc3"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("gwtc3_corner_pos.pdf"))
    ap.add_argument("--seed", type=int, default=0, help="posterior-resampling seed")
    for m in ("met1", "met3", "met4"):
        ap.add_argument(f"--{m}", type=Path, default=None)
    args = ap.parse_args()

    c = popfig.setup_style()
    import corner
    from matplotlib.lines import Line2D

    cfg = popfig.RUNS[RUN]
    posts = popfig.load_run(RUN, args.seed, {m: getattr(args, m) for m in ("met1", "met3", "met4")
                                             if getattr(args, m)})
    labels = [popfig.LATEX[p] for p in cfg["params"]]

    colors = {"met1": c[0], "met3": c[1], "met4": c[2]}
    fig = None
    for m, col in colors.items():
        kw = dict(label_kwargs={"fontsize": 20}) if fig is None else dict(fig=fig)
        fig = corner.corner(posts[m], labels=labels, hist_kwargs={"density": True},
                            color=col, **kw)
    fig.legend(handles=[Line2D([0], [0], color=col, lw=2, label=popfig.METHOD_LABEL[m])
                        for m, col in colors.items()],
               loc="upper right", bbox_to_anchor=(0.9, 0.9), fontsize=20)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"saved -> {args.out}")
    for m, s in posts.items():
        print(f"{m} medians: " + "  ".join(f"{p}={v:.3g}" for p, v in zip(cfg["params"], np.median(s, 0))))


if __name__ == "__main__":
    main()
