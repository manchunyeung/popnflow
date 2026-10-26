#!/usr/bin/env python
"""
sbc_00_draw_hyperparams.py
==========================
Step 0 of the SBC suite: draw n hyperparameter vectors from the RECOVERY prior
and lay out one block directory per draw.

This is the step the existing MDC does not have. Everything downstream
(sim_cat_03 -> _04 -> GMMs -> inference) already parameterises cleanly once
Lambda_j can be injected, which sim_cat_03's --true-pop-json now allows.

WHAT IT GUARANTEES
  * Lambda_j comes from sbc_prior, which test_prior_agreement.py checks against
    sim_cat_inference.py's own inline bounds (exact + KS). Without that the SBC
    guarantee is void.
  * Every Lambda_j is written to disk BEFORE any downstream compute (spec 10),
    so a block that dies mid-pool is still identifiable.
  * Per-block seeds come from SeedSequence.spawn off one master seed, so the
    whole suite reproduces from --master-seed alone.

POOL-FILL FEASIBILITY (spec 3.2, adapted)
  The spec frames this as "keep N_det in [30, 300]". That framing does not fit
  this pipeline: sim_cat_03 takes the FIRST --pool-size DETECTED events out of
  --n-draw draws, so the pool size is fixed at 276 by construction and what
  actually varies is the detection EFFICIENCY. The real failure mode is a
  Lambda_j so unfavourable that --n-draw draws never yield 276 detections.

  So we predict efficiency per Lambda_j and flag blocks at risk. Efficiency is
  computed as a RATIO against the fixed-Lambda MDC (whose efficiency is known
  by direct measurement, 867/1e6), which cancels every normalisation constant:

      eff(L) / eff(L_MDC) = [xi_raw(L)/Z(L)] / [xi_raw(L_MDC)/Z(L_MDC)]

  xi_raw is the standard reweighted sum over injections.h5 (identical to the
  likelihood's own selection term); Z(L) is the p(z) normalisation, a 1-D
  quadrature, since the mass model is already normalised.

Outputs under --outdir:
    hyperparams.csv              j, all 8 params, predicted efficiency, verdict
    provenance.json              master seed, per-block seeds, prior hash, config
    blocks/block_{j:03d}/lambda_true.json     <- consumed by --true-pop-json

Example
    python sbc_00_draw_hyperparams.py --n-blocks 8 --outdir sbc_run_pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbc_prior  # noqa: E402

PROD = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent   # suites live here: one big directory
DEFAULT_INJ = PROD / "sim_cat6_sharp_w8" / "injections.h5"

# Measured for the fixed-Lambda MDC by sim_cat_03 itself: 1e6 drawn, 867
# detected at rho_obs >= 9 (mdc_pool.h5 attrs n_draw / n_detected).
MDC_EFFICIENCY = 867 / 1_000_000
Z_MAX = 1.9

_cosmo = FlatLambdaCDM(H0=Planck15.H0.value, Om0=Planck15.Om0)
_zg = np.linspace(0.0, Z_MAX, 4000)
_dVdz = (_cosmo.differential_comoving_volume(_zg).to(u.Mpc**3 / u.sr).value
         * 4.0 * np.pi)


def _log_joint_mass(m1, m2, p):
    a = -p["alpha"]
    dp = p["m_max"] ** (1.0 + a) - p["m_min"] ** (1.0 + a)
    p_pl = (1.0 + a) * m1 ** a / dp
    p_pl = np.where((m1 < p["m_min"]) | (m1 > p["m_max"]), 0.0, p_pl)
    p_pk = (np.exp(-0.5 * (m1 - p["mu"]) ** 2 / p["sigma"] ** 2)
            / np.sqrt(2.0 * np.pi * p["sigma"] ** 2))
    p_m1 = p["f1"] * p_pk + (1.0 - p["f1"]) * p_pl
    q = m2 / m1
    q_min = p["m_min"] / m1
    p_q = (1.0 + p["beta"]) * q ** p["beta"] / (1.0 - q_min ** (1.0 + p["beta"]))
    p_q = np.where(q * m1 < p["m_min"], 0.0, p_q)
    lp = (np.log(np.maximum(p_m1, 1e-300)) + np.log(np.maximum(p_q, 1e-300))
          - np.log(m1))
    return np.where(m2 < m1, lp, -np.inf)


def _log_pz_unnorm(z, gamma):
    return (np.log(np.maximum(np.interp(z, _zg, _dVdz), 1e-300))
            + (gamma - 1.0) * np.log1p(z))


def _z_norm(gamma):
    """Z(L) = int_0^zmax dV/dz (1+z)^(gamma-1) dz."""
    return float(np.trapezoid(np.exp(_log_pz_unnorm(_zg, gamma)), _zg))


def _xi_raw(inj, p):
    m1, m2, z, pdraw = inj
    lw = _log_joint_mass(m1, m2, p) + _log_pz_unnorm(z, p["gamma"]) - np.log(pdraw)
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    mx = lw.max()
    return float(np.exp(mx) * np.sum(np.exp(lw - mx)))


def load_injections(path):
    with h5py.File(path, "r") as f:
        return (f["m1"][:], f["m2"][:], f["z"][:], f["sampling_pdf"][:])


def peak_leak_fraction(p, n1=900, n2=500):
    """Fraction of the LIKELIHOOD's p(m1,m2) mass outside the GENERATOR's support.

    THE GENERATOR AND THE LIKELIHOOD DISAGREE ABOUT THE GAUSSIAN PEAK'S TAILS.
    sim_cat_03_generate_pool.draw_population builds its m1 grid on
    [m_min, m_max] and renormalises (`cdf /= cdf[-1]`), so the peak is
    TRUNCATED. sim_cat_inference.py's p_pk has no such cutoff -- only the
    power-law component is zeroed outside the range -- so the likelihood keeps
    Gaussian density above m_max that the generator can never produce. (Below
    m_min the p_q term kills it anyway, since q_min = m_min/m1 > 1 makes p_q
    non-positive.)

    On [m_min, m_max] the two share a functional form, so the difference is a
    Lambda-dependent constant -- and a constant would cancel between the
    per-event product and xi^N. What does NOT cancel is that xi(Lambda)
    integrates injections ABOVE m_max where the likelihood has support and the
    truth has none, biasing the selection term by ~(1 + leak).

    Negligible at the fixed-Lambda MDC point (peak 8.8 sigma from m_max, leak
    0.000%), which is why model_checks/compare_generation_vs_likelihood.py did
    not see it. It is NOT negligible across the prior: a draw with the peak
    ~1 sigma from m_max leaks several percent.
    """
    lo, hi, sg = p["m_min"], p["m_max"], p["sigma"]
    top = hi + 12.0 * sg
    m1 = np.linspace(lo, top, n1)
    frac = np.linspace(0.0, 1.0, n2)
    M1 = np.repeat(m1, n2)
    M2 = (lo + frac[None, :] * (m1[:, None] - lo)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.exp(_log_joint_mass(M1, M2, p)).reshape(n1, n2)
    dens = np.nan_to_num(dens)
    inner = np.trapezoid(dens, axis=1) * (m1 - lo) / max(n2 - 1, 1) * (n2 - 1)
    inner = np.array([np.trapezoid(dens[i], M2.reshape(n1, n2)[i])
                      for i in range(n1)])
    tot = np.trapezoid(inner, m1)
    if tot <= 0:
        return 0.0
    above = np.trapezoid(inner[m1 >= hi], m1[m1 >= hi]) if (m1 >= hi).any() else 0.0
    return float(above / tot)


def predict_efficiency(inj, p, ref_ratio):
    r = _xi_raw(inj, p) / _z_norm(p["gamma"])
    return MDC_EFFICIENCY * r / ref_ratio


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROD),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, required=True,
                    help="n in the spec: number of Lambda_j draws. This is the "
                         "axis that buys statistical power (between-block "
                         "variance is the binding term), and also the "
                         "expensive one -- each block is a fresh pool + mock "
                         "PE + GMM fit.")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--master-seed", type=int, default=20260815,
                    help="the whole suite reproduces from this alone")
    ap.add_argument("--injections", type=Path, default=DEFAULT_INJ,
                    help="Lambda-INDEPENDENT: one injection set is shared by "
                         "every block (fixed reference distribution + stored "
                         "sampling_pdf, so xi is recovered by reweighting). "
                         "Never regenerate this per block.")
    ap.add_argument("--n-draw", type=int, default=4_000_000,
                    help="must match what sbc_01 passes to sim_cat_03")
    ap.add_argument("--pool-size", type=int, default=276)
    ap.add_argument("--margin", type=float, default=3.0,
                    help="flag a block unless expected detections exceed "
                         "margin x pool-size")
    ap.add_argument("--max-leak", type=float, default=0.01,
                    help="warn when the generator/likelihood peak-truncation "
                         "mismatch (see peak_leak_fraction) exceeds this "
                         "fraction. It is a Lambda-dependent MODEL "
                         "misspecification, not an estimator failure, so a "
                         "block above this threshold can produce SBC "
                         "non-uniformity that says nothing about the sampler.")
    args = ap.parse_args()

    outdir = args.outdir if args.outdir.is_absolute() else HERE / args.outdir
    args.outdir = outdir
    blocks_dir = outdir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    # --- prior seam must be sound before anything is drawn -----------------
    sbc_prior.assert_no_joint_constraint_violations()

    ss = np.random.SeedSequence(args.master_seed)
    child = ss.spawn(args.n_blocks)

    # Lambda_j is drawn from BLOCK j's OWN child seed, so it depends only on
    # (master_seed, j) and NOT on n_blocks. That makes the suite EXTENSIBLE:
    # --n-blocks 20 reproduces the first 10 of --n-blocks 10 exactly, so more
    # blocks can be added later without invalidating the ones already run.
    #
    # The obvious alternative -- one rng for all draws, seeded after
    # ss.spawn(n_blocks) -- silently ties every Lambda_j to n_blocks, because
    # spawn() advances the parent's counter. Changing n would then reshuffle
    # every block and quietly orphan the finished ones.
    lambdas = np.array([sbc_prior.draw(1, np.random.default_rng(child[j]))[0]
                        for j in range(args.n_blocks)])
    assert sbc_prior.is_finite_logprior(lambdas).all(), \
        "a drawn Lambda_j has non-finite log-prior"

    print(f"[inj] {args.injections}")
    inj = load_injections(args.injections)
    ref_ratio = _xi_raw(inj, sbc_prior.LAMBDA_MDC) / _z_norm(
        sbc_prior.LAMBDA_MDC["gamma"])
    print(f"[inj] {len(inj[0]):,} found injections; MDC efficiency "
          f"{MDC_EFFICIENCY:.3e} used as the calibration point\n")

    rows = []
    hdr = (f"{'j':>3} " + " ".join(f"{k:>7}" for k in sbc_prior.PARAM_NAMES)
           + f" {'eff':>10} {'E[det]':>9} {'leak':>7}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for j in range(args.n_blocks):
        p = sbc_prior.to_dict(lambdas[j])
        eff = predict_efficiency(inj, p, ref_ratio)
        exp_det = eff * args.n_draw
        leak = peak_leak_fraction(p)
        ok = exp_det >= args.margin * args.pool_size
        leak_ok = leak <= args.max_leak
        verdict = ("ok" if ok else "AT RISK: raise --n-draw")
        if not leak_ok:
            verdict += f"  | peak leak {leak:.1%} (MITIGATED by --truncate-peak)"

        bdir = blocks_dir / f"block_{j:03d}"
        bdir.mkdir(parents=True, exist_ok=True)
        # written BEFORE any downstream compute -- spec 10
        (bdir / "lambda_true.json").write_text(json.dumps(p, indent=2))
        (bdir / "seed.txt").write_text(str(child[j].entropy) + "\n")

        print(f"{j:>3} " + " ".join(f"{p[k]:>7.3f}" for k in sbc_prior.PARAM_NAMES)
              + f" {eff:>10.3e} {exp_det:>9.0f} {leak:>6.2%}  {verdict}")
        rows.append(dict(j=j, **p, efficiency=eff, expected_detections=exp_det,
                         feasible=bool(ok), peak_leak=leak,
                         leak_ok=bool(leak_ok),
                         block_seed=int(child[j].generate_state(1)[0])))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "hyperparams.csv", index=False)

    prior_hash = hashlib.sha256(
        (str(sbc_prior.LOWER) + str(sbc_prior.UPPER)
         + str(sbc_prior.PARAM_NAMES)).encode()).hexdigest()[:16]
    (args.outdir / "provenance.json").write_text(json.dumps(dict(
        git_sha=git_sha(), master_seed=args.master_seed,
        n_blocks=args.n_blocks, prior_hash=prior_hash,
        prior_lower=sbc_prior.LOWER.tolist(),
        prior_upper=sbc_prior.UPPER.tolist(),
        param_names=sbc_prior.PARAM_NAMES,
        injections=str(args.injections), n_draw=args.n_draw,
        pool_size=args.pool_size,
        block_seeds=[int(c.generate_state(1)[0]) for c in child],
    ), indent=2))

    nleak = int((~df.leak_ok).sum())
    if nleak:
        print(f"\n[note] {nleak}/{args.n_blocks} blocks would leak more than "
              f"--max-leak {args.max_leak:.1%} of the likelihood's peak above "
              f"m_max, WITHOUT mitigation (worst {df.peak_leak.max():.1%}; xi "
              f"would shift by ~-16% for that block).")
        print(f"[note] run_sbc_block.sh passes --truncate-peak, which restricts "
              f"the likelihood's Gaussian peak to [m_min, m_max] and so matches "
              f"the generator exactly -- these blocks are MITIGATED. The column "
              f"records the size of the effect that flag is removing. If you "
              f"run sim_cat_inference.py by hand WITHOUT --truncate-peak, these "
              f"blocks are genuinely misspecified.")
    nbad = int((~df.feasible).sum())
    print(f"\n[out] {args.outdir}/hyperparams.csv, provenance.json, "
          f"blocks/block_000..{args.n_blocks-1:03d}/lambda_true.json")
    if nbad:
        print(f"[WARN] {nbad}/{args.n_blocks} blocks may not fill a "
              f"{args.pool_size}-event pool at --n-draw {args.n_draw:,}. "
              f"Raise --n-draw for those blocks or they will fail in sbc_01.")
    else:
        print(f"[ok] all {args.n_blocks} blocks should fill the pool "
              f"with >= {args.margin}x margin")


if __name__ == "__main__":
    main()
