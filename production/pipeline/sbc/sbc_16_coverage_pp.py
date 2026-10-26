#!/usr/bin/env python
"""
sbc_16_coverage_pp.py
=====================
COVERAGE PP plot -- the companion to sbc_03/sbc_04's PIT PP plot, answering the
question those two do NOT answer.

    sbc_03/sbc_04 plot   u  vs  fraction of catalogs with PIT <= u
    this script plots    p  vs  fraction of catalogs whose central p credible
                                interval CONTAINS the truth

The PIT ECDF is the right statistic for detecting a BIASED posterior (a shifted
curve says the truth sits systematically low or high in the posterior), but you
cannot read a coverage number off it: PIT <= u is a one-sided statement about
posterior mass below the truth, not a statement about containment. A posterior
that is shifted low half the time and high the other half has a perfect PIT
ECDF (S-shaped deviations cancel in neither direction here -- they show up, but
as shape, not as level) while its 90% intervals may contain the truth far less
than 90% of the time.

THE CONVERSION IS EXACT AND NEEDS NO NEW RUNS. The central (equal-tailed) p
credible interval runs from the (1-p)/2 to the (1+p)/2 posterior quantile, so

    truth inside  <=>  (1-p)/2 <= PIT <= (1+p)/2  <=>  |2*PIT - 1| <= p

Hence coverage(p) is just the ECDF of |2*PIT - 1|, computed from the SAME
pit/{method}.npz sbc_02 already wrote. This intentionally re-derives from the
PIT arrays rather than re-reading the dynesty checkpoints, so the coverage
curve and the PIT curve can never disagree about which runs were usable.

  (CENTRAL intervals, not HPD. HPD containment is not a function of the PIT --
  it needs the posterior density -- so it cannot be recovered from the cached
  arrays. Central is also what the fixed-Lambda PP machinery quotes.)

NULL: under calibration PIT ~ U(0,1), so |2*PIT - 1| ~ U(0,1) too and the null
coverage curve is the SAME diagonal. A curve BELOW the diagonal is
undercoverage (intervals too narrow / too confident); ABOVE is overcoverage.
That reading is the whole point -- on the PIT plot, "below the diagonal" means
something completely different (bias direction), which is how these two plots
get confused.

BANDS are the identical two-level hierarchical bootstrap as sbc_03_bands.py
(blocks with replacement, then catalogs within), with the pivot centred on the
observed curve, not on the diagonal. Copied deliberately rather than imported:
sbc_03 is a __main__ script whose import would drag in its argparse-era
globals, and the function is 10 lines.

Usage:
    python sbc_16_coverage_pp.py --suite sbc_n10 \
        --methods met1 met3_cholfix met4_cholfix --out coverage_pp_cholfix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbc_prior  # noqa: E402

matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent

# Same mapping as sbc_15_prior_pp.py / pp_bias_diagnostics.py -- f1 is the
# peak fraction, written lambda everywhere in the paper.
PARAM_TEX = {"gamma": r"$\gamma$", "m_min": r"$m_\mathrm{min}$",
             "m_max": r"$m_\mathrm{max}$", "alpha": r"$\alpha$",
             "beta": r"$\beta$", "mu": r"$\mu$", "sigma": r"$\sigma$",
             "f1": r"$\lambda$"}
GRID = np.linspace(0.0, 1.0, 201)
QUOTE_LEVELS = (0.5, 0.68, 0.90)   # levels reported in the CSV / stdout

METHOD_LABELS = {"met1": "PE", "met3": "Ensemble", "met4": "Mixture",
                 "met4_nlive200": "Mixture",
                 "met3_boxfix": "Ensemble (boxfix)",
                 "met4_boxfix": "Mixture (boxfix)",
                 "met3_cholfix": "Ensemble", "met4_cholfix": "Mixture",
                 "met3_cholfix_nlive1000": "Ensemble",
                 "met4_cholfix_nlive1000": "Mixture",
                 # n=40 extension (blocks 0-39), same config as *_cholfix
                 "met1_n40": "PE", "met3_cholfix_n40": "Ensemble",
                 "met4_cholfix_n40": "Mixture"}
SHORT_LABELS = {"met1": "PE", "met3": "Ens", "met4": "Mix",
                "met4_nlive200": "Mix", "met3_boxfix": "Ens-bf",
                "met4_boxfix": "Mix-bf", "met3_cholfix": "Ens",
                "met4_cholfix": "Mix",
                "met3_cholfix_nlive1000": "Ens",
                "met4_cholfix_nlive1000": "Mix",
                "met1_n40": "PE", "met3_cholfix_n40": "Ens",
                "met4_cholfix_n40": "Mix"}
METHOD_COLORS = {"met1": "#0173B2", "met3": "#DE8F05", "met4": "#029E73",
                 "met4_nlive200": "#029E73", "met3_boxfix": "#D55E00",
                 "met4_boxfix": "#CC78BC", "met3_cholfix": "#DE8F05",
                 "met4_cholfix": "#029E73",
                 "met3_cholfix_nlive1000": "#DE8F05",
                 "met4_cholfix_nlive1000": "#029E73",
                 "met1_n40": "#0173B2", "met3_cholfix_n40": "#DE8F05",
                 "met4_cholfix_n40": "#029E73"}


def coverage_curve_ref(pit_values, p_grid=GRID):
    """Reference implementation, written the obvious way: for each level p,
    what fraction of catalogs have the truth inside the central p interval.

    O(len(grid) * n) and only used by the self-test below; `coverage_curve` is
    the vectorised form the bootstrap actually calls (10k replicates x 201
    levels x 8 params x 3 methods is ~20M passes in this form, minutes of
    wall time for a plot that should take seconds).
    """
    v = np.asarray(pit_values, float)
    if v.size == 0:
        return np.full(len(p_grid), np.nan)
    cov = []
    for p in p_grid:
        lo, hi = (1.0 - p) / 2.0, (1.0 + p) / 2.0
        inside = (v >= lo) & (v <= hi)
        cov.append(inside.mean())
    return np.array(cov)


def coverage_curve(pit_values, p_grid=GRID):
    """Fraction of catalogs whose central p credible interval holds the truth.

    Identical to coverage_curve_ref: the central p interval runs from the
    (1-p)/2 to the (1+p)/2 posterior quantile, so

        (1-p)/2 <= PIT <= (1+p)/2   <=>   |2*PIT - 1| <= p

    and coverage(p) is just the ECDF of |2*PIT - 1|, one sort plus a
    searchsorted. `side="right"` keeps the <= of the reference form.
    """
    v = np.asarray(pit_values, float)
    if v.size == 0:
        return np.full(len(p_grid), np.nan)
    d = np.sort(np.abs(2.0 * v - 1.0))
    return np.searchsorted(d, p_grid, side="right") / len(d)


def _selftest_coverage_curve(seed=0):
    """The two forms must agree exactly on random PIT values, including the
    endpoints and exact-tie cases the searchsorted side= argument controls."""
    rng = np.random.default_rng(seed)
    for v in (rng.random(500), np.array([0.0, 0.5, 1.0, 0.25, 0.75]),
              np.array([0.5])):
        a, b = coverage_curve(v), coverage_curve_ref(v)
        assert np.allclose(a, b), (v[:5], np.abs(a - b).max())
    return True


def hierarchical_bootstrap(pit_a, ok, n_boot=10_000, seed=0, grid=GRID):
    """Two-level resample: blocks with replacement, then catalogs within.

    Same design as sbc_03_bands.py -- a flat/binomial band would assume n*m
    independent draws and be far too narrow, since catalogs in a block share
    both Lambda_j and an event pool.
    """
    rng = np.random.default_rng(seed)
    n = ok.shape[0]
    per_block = [pit_a[j][ok[j]] for j in range(n)]
    live = [j for j in range(n) if len(per_block[j])]
    if not live:
        raise SystemExit("no usable runs")
    reps = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        js = rng.choice(live, size=len(live), replace=True)
        draws = [blk[rng.integers(0, len(blk), len(blk))]
                 for blk in (per_block[j] for j in js)]
        reps[b] = coverage_curve(np.concatenate(draws), grid)
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--params", nargs="+", default=None, help="default: all 8")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="coverage_pp")
    ap.add_argument("--csv", type=str, default=None,
                    help="default: <out>_summary.csv next to the suite root")
    args = ap.parse_args()

    _selftest_coverage_curve()   # vectorised form == the explicit one

    suite = args.suite if args.suite.is_absolute() else HERE / args.suite
    figdir = suite / "figures"
    covdir = suite / "coverage"
    figdir.mkdir(parents=True, exist_ok=True)
    covdir.mkdir(parents=True, exist_ok=True)
    params = args.params or sbc_prior.PARAM_NAMES

    loaded = {}
    for method in args.methods:
        f = suite / "pit" / f"{method}.npz"
        if not f.exists():
            print(f"[skip] {f} missing -- run sbc_02 for {method} first")
            continue
        d = np.load(f, allow_pickle=True)
        loaded[method] = d
        n, m = d["ok"].shape
        print(f"[load] {method}: n={n} blocks x m={m} catalogs, "
              f"{int(d['ok'].sum())}/{n*m} usable")
    if not loaded:
        raise SystemExit("no PIT files found for the requested methods")

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    for a_i, pname in enumerate(sbc_prior.PARAM_NAMES):
        ax = axes.ravel()[a_i]
        if pname not in params:
            ax.set_visible(False)
            continue

        curves, bands = {}, {}
        for method, d in loaded.items():
            names = [str(x) for x in d["param_names"]]
            a = names.index(pname)
            pit_a, ok = d["pit"][:, :, a], d["ok"]
            obs = coverage_curve(pit_a[ok], GRID)
            reps = hierarchical_bootstrap(pit_a, ok, args.n_boot,
                                          args.seed + a_i)
            dev = reps - obs[None, :]
            sup = np.abs(dev).max(axis=1)
            d_sim = float(np.percentile(sup, 95))
            d_obs = float(np.abs(obs - GRID).max())
            pval = float((sup >= d_obs).mean())
            curves[method] = obs
            bands[method] = (d_sim, d_obs, pval,
                             GRID + np.percentile(dev, 2.5, axis=0),
                             GRID + np.percentile(dev, 97.5, axis=0))

            # coverage AT the quoted levels, with a bootstrap CI -- the number
            # that actually gets written in a paper ("our 90% intervals contain
            # the truth X% of the time").
            quoted = {}
            for p in QUOTE_LEVELS:
                i = int(np.argmin(np.abs(GRID - p)))
                q_lo, q_hi = np.percentile(reps[:, i], [2.5, 97.5])
                quoted[p] = (float(obs[i]), float(q_lo), float(q_hi))

            np.savez(covdir / f"{method}_{pname}.npz", reps=reps, obs=obs,
                     grid=GRID, sup=sup, d_obs=d_obs, d_sim95=d_sim,
                     pvalue=pval, n_usable=int(ok.sum()),
                     quote_levels=np.array(QUOTE_LEVELS),
                     quoted=np.array([quoted[p] for p in QUOTE_LEVELS]))

            row = dict(method=method, param=pname, n_blocks=ok.shape[0],
                       m_catalogs=ok.shape[1], n_usable=int(ok.sum()),
                       sup_dev=round(d_obs, 4), sim_band_95=round(d_sim, 4),
                       sim_pvalue=round(pval, 4), passes=bool(d_obs <= d_sim))
            for p in QUOTE_LEVELS:
                c, lo, hi = quoted[p]
                tag = f"cov{int(round(p*100)):02d}"
                row[tag] = round(c, 4)
                row[f"{tag}_lo"] = round(lo, 4)
                row[f"{tag}_hi"] = round(hi, 4)
            rows.append(row)

        # Bare curves against the diagonal: the bands and the per-method
        # p-values are still computed above and written to the CSV / npz, they
        # are just not drawn.
        ax.plot(GRID, GRID, color="#0b0b0b", lw=1.1, ls="--")

        for method in args.methods:
            if method not in curves:
                continue
            ax.plot(GRID, curves[method], lw=2.0,
                    color=METHOD_COLORS.get(method),
                    label=METHOD_LABELS.get(method, method))

        ax.axvline(0.90, color="#999999", lw=0.8, ls=":", zorder=0)
        ax.set_title(PARAM_TEX.get(pname, pname), fontsize=13)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, ls=":")
        if a_i == 0:
            ax.legend(fontsize=9, loc="lower right")

    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_xlabel("credible interval $p$", fontsize=11)
            ax.set_ylabel("fraction", fontsize=11)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"{args.out}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n-> {figdir}/{args.out}.pdf")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        csv = suite / (args.csv or f"{args.out}_summary.csv")
        df.to_csv(csv, index=False)
        print(f"[out] {csv}\n")
        for method in args.methods:
            if method not in loaded:
                continue
            print(f"===== {method} =====")
            print(f"  {'param':>7}  " + "  ".join(
                f"{int(round(p*100)):>2d}%" for p in QUOTE_LEVELS)
                + "   sup|C-p|   band     p      ")
            for r in (x for x in rows if x["method"] == method):
                cells = "  ".join(
                    f"{r[f'cov{int(round(p*100)):02d}']*100:4.0f}"
                    for p in QUOTE_LEVELS)
                flag = "PASS" if r["passes"] else "FAIL"
                print(f"  {r['param']:>7}  {cells}     "
                      f"{r['sup_dev']:.3f}   {r['sim_band_95']:.3f}  "
                      f"{r['sim_pvalue']:.3f}  {flag}")
            print()


if __name__ == "__main__":
    main()
