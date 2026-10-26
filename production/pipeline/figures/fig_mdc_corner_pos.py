#!/usr/bin/env python
"""Paper figure `mdc_corner_pos.pdf` (fig:mdc_pos).

Hyperparameter posteriors of the sharp PL+peak model for one simulated catalog
(sim_cat6_sharp_w8 realization 0, 69 events) from PE (met1, blue), Ensemble
(met3, orange) and Mixture (met4, green), with the injected values in black.

Inputs: the three dynesty checkpoints listed in popfig.RUNS["mdc"] -- met1 from
inference/, met3/met4 from the nlive=1000 runs with the PE-bounds box and the
Cholesky proposal fix. Extracted from make_plots_met4.py --run mdc --seed 0
(its {run}_{seed}_corner_all.pdf); resampling is seeded here (--seed).

    python fig_mdc_corner_pos.py [--seed 0] [--met1/--met3/--met4 PATH] [--out ...]
"""
import argparse
from pathlib import Path

import numpy as np

import popfig

RUN = "mdc"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("mdc_corner_pos.pdf"))
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
    truths = [cfg["truths"][p] for p in cfg["params"]]

    colors = {"met1": c[0], "met3": c[1], "met4": c[2]}
    fig = None
    for m, col in colors.items():
        kw = dict(label_kwargs={"fontsize": 20}) if fig is None else dict(fig=fig)
        fig = corner.corner(posts[m], labels=labels, truths=truths, truth_color="black",
                            hist_kwargs={"density": True}, color=col, **kw)
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
