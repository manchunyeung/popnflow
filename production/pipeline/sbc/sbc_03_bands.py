#!/usr/bin/env python
"""
sbc_03_bands.py
===============
Pooled PP curve, hierarchical-bootstrap bands, and the design-effect
diagnostics, from sbc_02's PIT arrays.

WHY NOT A BINOMIAL FUNNEL (spec 6). Each q_jk is marginally U(0,1) by
construction, so the DIAGONAL is a valid null -- but q values sharing a block j
are correlated (they are draws from that Lambda_j's fixed-Lambda coverage
curve), and catalogs within a block additionally share a pool. A binomial band
assumes n*m independent draws and is therefore far too narrow. The bootstrap
below mirrors the two-level design and absorbs both correlations without
having to estimate either.

    for B replicates:
        resample block indices j* with replacement from 1..n
        within each, resample catalog indices k* with replacement from that
            block's m catalogs
        recompute the pooled ECDF

Pointwise bands are quantiles of F*(u) at each u; the SIMULTANEOUS band comes
from quantiles of sup_u |F*(u) - u| and is the PRIMARY criterion -- pointwise
bands invite cherry-picking the worst u.

RHO_LAMBDA (spec 7) is the payoff beyond the pass/fail: it measures how much
fixed-Lambda coverage varies across the prior.
    rho ~ 0  -> fixed-Lambda coverage is already uniform for this parameter, so
                the EXISTING fixed-Lambda PP plot was valid, and extra catalogs
                per block buy full value.
    rho large-> fixed-Lambda coverage varies across the prior; the existing
                fixed-Lambda PP plot for that parameter cannot be read against
                a diagonal, and more catalogs per block are nearly worthless.
Reported band-averaged over u in [0.1, 0.9] with a bootstrap uncertainty, since
it is noisy pointwise at small n.

Mathtext, not usetex: text.usetex drops minus signs on this host.
"""
from __future__ import annotations

import argparse
import json
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

PROD = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent   # suites live here: one big directory
GRID = np.linspace(0.0, 1.0, 201)


def pooled_ecdf(vals, grid=GRID):
    v = np.sort(np.asarray(vals))
    return np.searchsorted(v, grid, side="right") / max(len(v), 1)


def hierarchical_bootstrap(pit_a, ok, n_boot=10_000, seed=0, grid=GRID):
    """Two-level resample: blocks with replacement, then catalogs within."""
    rng = np.random.default_rng(seed)
    n, m = ok.shape
    per_block = [pit_a[j][ok[j]] for j in range(n)]
    live = [j for j in range(n) if len(per_block[j])]
    if not live:
        raise SystemExit("no usable runs")
    reps = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        js = rng.choice(live, size=len(live), replace=True)
        draws = [blk[rng.integers(0, len(blk), len(blk))]
                 for blk in (per_block[j] for j in js)]
        reps[b] = pooled_ecdf(np.concatenate(draws), grid)
    return reps


def rho_lambda(pit_a, ok, m_eff, grid=GRID, lo=0.1, hi=0.9):
    """Spec 7, band-averaged over u in [lo, hi]. Clipped at zero."""
    n = ok.shape[0]
    live = [j for j in range(n) if ok[j].any()]
    if len(live) < 2:
        return np.nan
    p = np.array([[(pit_a[j][ok[j]] <= u).mean() for u in grid] for j in live])
    var_j = p.var(axis=0, ddof=1)
    within = (p * (1 - p)).mean(axis=0) / max(m_eff, 1e-9)
    denom = grid * (1 - grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(denom > 0, (var_j - within) / denom, np.nan)
    sel = (grid >= lo) & (grid <= hi)
    return float(np.clip(np.nanmean(rho[sel]), 0.0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["met1"])
    ap.add_argument("--params", nargs="+", default=None,
                    help="default: all 8")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m-eff", type=float, default=None,
                    help="shared-pool-corrected catalog count; overrides the "
                         "--rho-pool calculation below.")
    ap.add_argument("--rho-pool", type=float, default=0.159,
                    help="latent PIT correlation between catalogs sharing a "
                         "pool; m_eff = m/(1+(m-1)*rho_pool). The default is "
                         "CALIBRATED to the existing fixed-Lambda band "
                         "machinery, which measures N_eff ~ 5.6 of 50 catalogs "
                         "for the 276/69 configuration (8.8x variance "
                         "inflation at p=0.5): 8.8 = 1+49*rho => rho = 0.159. "
                         "NOTE this is NOT the raw subsampling fraction "
                         "N_cat/N_pool = 0.25 -- using that gives m_eff 3.77 "
                         "at m=50 and disagrees with the existing machinery. "
                         "The probit-scale correlation is lower than the event "
                         "overlap. RECOMPUTE if N_pool/N_cat change; the "
                         "implied ceiling is m_eff -> 1/rho_pool = 6.3, which "
                         "is why extra catalogs saturate and extra BLOCKS are "
                         "the better buy.")
    ap.add_argument("--n-pool", type=int, default=276)
    ap.add_argument("--n-cat", type=int, default=69)
    args = ap.parse_args()

    suite = args.suite if args.suite.is_absolute() else HERE / args.suite
    params = args.params or sbc_prior.PARAM_NAMES
    figdir = suite / "figures"
    banddir = suite / "bands"
    figdir.mkdir(parents=True, exist_ok=True)
    banddir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in args.methods:
        f = suite / "pit" / f"{method}.npz"
        if not f.exists():
            print(f"[skip] {f} missing -- run sbc_02 for {method} first")
            continue
        d = np.load(f, allow_pickle=True)
        pit, ok = d["pit"], d["ok"]
        names = list(d["param_names"])
        n, m = ok.shape

        m_eff = args.m_eff
        if m_eff is None:
            m_eff = m / (1.0 + (m - 1) * args.rho_pool)
        print(f"\n===== {method}: n={n} blocks, m={m} catalogs, "
              f"{int(ok.sum())}/{n*m} usable, m_eff={m_eff:.2f} =====")

        fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
        for a_i, pname in enumerate(sbc_prior.PARAM_NAMES):
            ax = axes.ravel()[a_i]
            if pname not in params:
                ax.set_visible(False)
                continue
            a = names.index(pname)
            pit_a = pit[:, :, a]
            reps = hierarchical_bootstrap(pit_a, ok, args.n_boot,
                                          args.seed + a_i)
            obs = pooled_ecdf(pit_a[ok], GRID)

            # THE PIVOT MUST BE CENTRED ON obs, NOT ON THE DIAGONAL.
            # The bootstrap approximates the sampling distribution of F_hat by
            # that of F* about F_hat, so the null quantity is sup|F* - F_hat|.
            # Using sup|F* - u| instead double-counts: those replicates carry
            # the observed deviation AND bootstrap noise, which inflates the
            # band until it never rejects (validated: coverage 1.000 at every
            # rho, i.e. zero power). See test_bands_validation.py.
            dev = reps - obs[None, :]
            lo_pw = GRID + np.percentile(dev, 2.5, axis=0)
            hi_pw = GRID + np.percentile(dev, 97.5, axis=0)
            sup = np.abs(dev).max(axis=1)
            d_sim = float(np.percentile(sup, 95))
            d_obs = float(np.abs(obs - GRID).max())
            pval = float((sup >= d_obs).mean())
            rho = rho_lambda(pit_a, ok, m_eff)
            n_eff = m_eff * n / (1 + (m_eff - 1) * rho) if np.isfinite(rho) else np.nan

            np.savez(banddir / f"{method}_{pname}.npz", reps=reps, obs=obs,
                     grid=GRID, sup=sup, d_obs=d_obs, d_sim95=d_sim,
                     pvalue=pval, rho_lambda=rho, n_eff=n_eff)

            ax.fill_between(GRID, np.clip(GRID - d_sim, 0, 1),
                            np.clip(GRID + d_sim, 0, 1),
                            color="#bfc3c9", alpha=0.45, lw=0,
                            label="95% simultaneous")
            ax.fill_between(GRID, lo_pw, hi_pw, color="#7f8792", alpha=0.35,
                            lw=0, label="95% pointwise")
            ax.plot(GRID, GRID, color="#0b0b0b", lw=1.1, ls="--")
            ax.plot(GRID, obs, color="#2a78d6", lw=2.0)
            ax.set_title(f"{pname}   $p$={pval:.2f}   "
                         r"$\rho_\Lambda$=" + f"{rho:.2f}", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.25, ls=":")
            if a_i == 0:
                ax.legend(fontsize=7, loc="upper left")

            shrink = np.nan
            rows.append(dict(method=method, param=pname, n_blocks=n,
                             m_catalogs=m, n_usable=int(ok.sum()),
                             m_eff=round(m_eff, 3),
                             rho_lambda=round(rho, 4),
                             n_eff=round(float(n_eff), 2),
                             ks_stat=round(d_obs, 4),
                             sim_band_95=round(d_sim, 4),
                             sim_pvalue=round(pval, 4),
                             passes=bool(d_obs <= d_sim),
                             boundary_delta=round(
                                 min(sbc_prior.LAMBDA_MDC[pname]
                                     - sbc_prior.LOWER[a_i],
                                     sbc_prior.UPPER[a_i]
                                     - sbc_prior.LAMBDA_MDC[pname]), 4)))
            flag = "PASS" if d_obs <= d_sim else "FAIL"
            print(f"  {pname:>7}: sup|F-u|={d_obs:.3f} vs band {d_sim:.3f}  "
                  f"p={pval:.3f}  rho={rho:.3f}  N_eff={n_eff:5.1f}   {flag}")

        for ax in axes.ravel():
            if ax.get_visible():
                ax.set_xlabel("$u$", fontsize=11)
                ax.set_ylabel("fraction PIT $\\leq u$", fontsize=10)
        fig.suptitle(f"SBC: {method}, n={n} prior draws x m={m} catalogs "
                     f"(hierarchical bootstrap, {args.n_boot} reps)",
                     fontsize=12, y=1.005)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(figdir / f"pp_{method}.{ext}", dpi=180,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {figdir}/pp_{method}.pdf")

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(suite / "summary.csv", index=False)
        print(f"\n[out] {suite}/summary.csv")
        print("\nINTERPRETING rho_lambda:")
        print("  ~0    fixed-Lambda coverage already uniform -> the existing")
        print("        fixed-Lambda PP plot was valid for this parameter")
        print("  large fixed-Lambda coverage varies across the prior -> that")
        print("        plot cannot be read against a diagonal, and extra")
        print("        catalogs per block buy little")


if __name__ == "__main__":
    main()
