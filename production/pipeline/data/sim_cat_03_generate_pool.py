#!/usr/bin/env python
"""
sim_cat_03_generate_pool.py
============================
Stage 3: draw the population, apply GWMockCat's rho_obs detection, build the
event pool, and generate mock PE posterior samples.

Follows gwmockcat_sample_generation.md:
  * Population drawn directly from the MDC sharp model (md route 2).  Route 1
    (re-running Ray et al.'s script with a loose cut) is not viable: neither
    that script nor its local descendant saved pre-selection draws, and both
    used an H1+L1+V1 network, inconsistent with endo3_v12's H1+L1.
  * Detection thresholded on **rho_obs**, never rho_opt (Essick & Fishbach
    2024).  GWMockCat draws rho_obs ~ N(rho_opt*Theta, 1) and cuts on that,
    letting near-threshold events scatter in and out.
  * Sensitivity: the validated H1+L1 O3-actual grid from stage 1/2.

PRIOR HANDLING -- deliberate deviation, documented per the package's own README
("There is a known issue with the priors returned by the `transforms` module.
We recommend using the 'lalprior' and then computing the priors yourself").
`GWMockCat/transforms.py` lines 380-387 apply the factor

    ddl_dz / (mass_jacobian * distance_jacobian)

TWICE to the lalprior reweighting weights, squaring it.  That is the known bug.
This script therefore performs the reweighting itself, applying the factor once:

    w  =  dL^2 * ddL/dz * p(Theta)  /  (mass_jacobian * distance_jacobian)

Samples are saved in DETECTOR frame -- (m1_det, m2_det, dL) -- matching the
downstream GMM step, which fits the density estimator in that space and
transforms back to source frame afterward.  `z` is saved alongside so that
back-transform (m1_src = m1_det/(1+z), m2_src = m2_det/(1+z)) is always
recoverable with no information loss.  Because the samples are stored in the
SAME coordinates the LVK convention is stated in, the declared PE prior needs
no extra Jacobian at this stage (contrast the source-frame case, which would
need to divide out (1+z)^2):

    p_PE(m1_det, m2_det, dL)  =  dL^2      (unnormalised; uniform in m1_det,
                                            m2_det is a constant and drops out)

Outputs
-------
  sim_cat/mdc_pool.h5     pool true parameters, rho_opt, rho_true, rho_obs, Theta
  sim_cat/pe_samples.h5   per-event mock PE samples (m1_det, m2_det, dL, z, prior)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import h5py
from astropy import units as u
from scipy import stats
from scipy.interpolate import interp1d

sys.path.insert(0, "/home/manchun.yeung/population/mock-PE")
from GWMockCat import posterior_utils as pu           # noqa: E402
from GWMockCat import cosmo_utils                      # noqa: E402
from GWMockCat.vt_utils import interpolate_optimal_snr_grid, draw_thetas  # noqa: E402

# ── config (single place; no magic numbers in functions) ──────────────────
CFG = dict(
    # true hyperparameters -- taken from the existing MDC configuration
    # (make_plots_met4.py / pp_bias_diagnostics.py TRUE_VALUES), not invented
    true_pop=dict(gamma=3.0, m_min=6.0, m_max=70.0, alpha=4.0,
                  beta=1.5, mu=35.0, sigma=4.0, f1=0.04),
    z_max=1.9,                 # matches endo3_v12 max_redshift
    pool_size=276,
    n_draw=4_000_000,          # pre-selection draws (efficiency ~8.5e-4)
    rho_thr=9.0,               # threshold on rho_obs
    n_pe_samples=5000,         # final samples per event
    pe_oversample=8,           # raw samples per event before LVK reweighting
    seed=20260803,
    grid="sim_cat/snr_grid_O3actual_H1L1_Xeff.h5",
    outdir="sim_cat",
    # GWMockCat measurement-uncertainty model
    # `q` is used only by --mass-ratio-sampling q; `eta` only by ...=eta.
    uncert=dict(threshold_snr=9.0, snr=1.0, mc=0.08, Theta=0.21,
                eta=0.022, q=0.22, q_floor=0.176, Xeff=0.1),
    mass_ratio_sampling="q",
)


def draw_population(N, pop, z_max, cosmo_dict, rng):
    """Draw (m1, m2, z) from the sharp power-law + Gaussian-peak MDC model.

    Hard cutoffs at m_min/m_max with NO smoothing filter -- the sharp low-mass
    edge is the feature the study tests and must survive into the mock data.
    Matches logpm1m2_plpeak_massratio / log_p_pop_lvk in
    mdc_inference_multicatalog.py.
    """
    mg = np.linspace(pop["m_min"], pop["m_max"], 20000)
    p_pl = mg ** (-pop["alpha"])
    p_pl /= np.trapezoid(p_pl, mg)
    p_pk = np.exp(-0.5 * ((mg - pop["mu"]) / pop["sigma"]) ** 2) / np.sqrt(
        2 * np.pi * pop["sigma"] ** 2)
    p_m1 = pop["f1"] * p_pk + (1.0 - pop["f1"]) * p_pl
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p_m1[1:] + p_m1[:-1]) * np.diff(mg))])
    cdf /= cdf[-1]
    m1 = np.interp(rng.random(N), cdf, mg)

    # p(q|m1) ~ q^beta on [m_min/m1, 1]; analytic inverse CDF (vectorised)
    q_min = pop["m_min"] / m1
    b1 = 1.0 + pop["beta"]
    q = (q_min ** b1 + rng.random(N) * (1.0 - q_min ** b1)) ** (1.0 / b1)
    m2 = q * m1

    # p(z) ~ dVc/dz (1+z)^(gamma-1)
    zg = np.linspace(1e-4, z_max, 4000)
    p_z = cosmo_dict["diff_comoving_vol"](zg) * (1 + zg) ** (pop["gamma"] - 1.0)
    zc = np.concatenate([[0.0], np.cumsum(0.5 * (p_z[1:] + p_z[:-1]) * np.diff(zg))])
    zc /= zc[-1]
    z = np.interp(rng.random(N), zc, zg)
    return m1, m2, z


def draw_population_smoothed(N, pop, delta_m, z_max, cosmo_dict, rng,
                             n_m=500, n_q=400):
    """Draw (m1, m2, z) from gwpopulation's SinglePeakSmoothedMassDistribution
    -- the same family Ray et al.'s MDC used -- instead of the sharp model.

    WHY THIS EXISTS.  The sharp model renormalises p(q|m1) over the shrinking
    range [m_min/m1, 1], so the marginal p(m1) is NOT penalised for leaving no
    room for m2.  With p(m1) ~ m1^-alpha that piles events at the low-mass edge.
    Measured against real GWTC-3 (PE-median m1, apples-to-apples):

        frac m1 < 10 :  real 0.017   Ray 0.069   sim_cat(sharp) 0.138
        5th pct m1   :  real 10.64   Ray 9.46    sim_cat(sharp)  7.94

    an ~8x excess of low-mass events.  The smoothed model multiplies by a window
    S(m) that -> 0 at m_min, applied to BOTH m1 and m2, which suppresses that
    corner.  Note the *mass-ratio* distribution was NOT the problem: measured
    median q is 0.662 (sim_cat) vs 0.715 (real), with 0% above q=0.9 in both.

    Sampled by Rosenblatt inversion on a 2D (m1, q) grid: m1 from the marginal,
    then q from the conditional row.  delta_m=0 is singular in gwpopulation's
    window (divide-by-zero), so use the sharp path for that.
    """
    from gwpopulation.models.mass import SinglePeakSmoothedMassDistribution
    model = SinglePeakSmoothedMassDistribution()
    kw = dict(alpha=pop["alpha"], beta=pop["beta"], mmin=pop["m_min"],
              mmax=pop["m_max"], lam=pop["f1"], mpp=pop["mu"], sigpp=pop["sigma"])

    m1g = np.linspace(pop["m_min"], pop["m_max"], n_m)
    qg = np.linspace(1e-3, 1.0, n_q)
    M1, Q = np.meshgrid(m1g, qg, indexing="ij")
    with np.errstate(all="ignore"):
        p = model({"mass_1": M1.ravel(), "mass_ratio": Q.ravel()},
                  delta_m=delta_m, **kw)
    p = np.clip(np.nan_to_num(np.asarray(p, float), nan=0, posinf=0, neginf=0),
                0, None).reshape(M1.shape)

    pm1 = np.trapezoid(p, qg, axis=1)
    cdf1 = np.concatenate([[0.0], np.cumsum(0.5 * (pm1[1:] + pm1[:-1]) * np.diff(m1g))])
    cdf1 /= cdf1[-1]
    cq = np.cumsum(0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(qg)[None, :], axis=1)
    cq = np.concatenate([np.zeros((len(m1g), 1)), cq], axis=1)
    cq /= np.clip(cq[:, -1:], 1e-300, None)

    m1 = np.interp(rng.random(N), cdf1, m1g)
    fi = np.interp(m1, m1g, np.arange(len(m1g)))
    lo = np.clip(fi.astype(int), 0, len(m1g) - 2)
    w = (fi - lo)[:, None]
    row = (1 - w) * cq[lo] + w * cq[lo + 1]
    u2 = rng.random(N)
    j = np.clip((row < u2[:, None]).sum(axis=1), 1, len(qg) - 1)
    ar = np.arange(N)
    c_lo, c_hi = row[ar, j - 1], row[ar, j]
    t = np.clip((u2 - c_lo) / np.maximum(c_hi - c_lo, 1e-300), 0, 1)
    q = qg[j - 1] + t * (qg[j] - qg[j - 1])
    m2 = q * m1

    zg = np.linspace(1e-4, z_max, 4000)
    p_z = cosmo_dict["diff_comoving_vol"](zg) * (1 + zg) ** (pop["gamma"] - 1.0)
    zc = np.concatenate([[0.0], np.cumsum(0.5 * (p_z[1:] + p_z[:-1]) * np.diff(zg))])
    zc /= zc[-1]
    z = np.interp(rng.random(N), zc, zg)
    return m1, m2, z


def build_ptheta(rng):
    """Reflected KDE of the Finn & Chernoff Theta distribution (as transforms.py)."""
    th = draw_thetas(100000, rng)
    th = np.sort(np.concatenate((th, [0.0, 1.0])))
    kde = stats.gaussian_kde(th)
    ts = np.linspace(0, 1, 1024)
    ps = kde(ts) + kde(-ts) + kde(2 - ts)
    return interp1d(ts, ps, bounds_error=False, fill_value=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draw", type=int, default=CFG["n_draw"])
    ap.add_argument("--pool-size", type=int, default=CFG["pool_size"])
    ap.add_argument("--n-pe-samples", type=int, default=CFG["n_pe_samples"])
    ap.add_argument("--pe-oversample", type=int, default=CFG["pe_oversample"])
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--outdir", type=Path, default=Path(CFG["outdir"]))
    ap.add_argument("--mass-ratio-sampling", choices=["q", "eta"],
                    default=CFG["mass_ratio_sampling"],
                    help="Coordinate the mock mass-ratio likelihood is generated "
                         "in. 'eta' is GWMockCat's native behaviour, which is "
                         "systematically biased for near-equal-mass systems "
                         "(one-sided truncation at eta=0.25 plus deta/dq->0); "
                         "'q' is the fix, see sim_cat_qsampling.py. Use 'eta' "
                         "to reproduce the original catalogs.")
    ap.add_argument("--mass-model", choices=["sharp", "smoothed"], default="sharp",
                    help="'sharp' = hard cutoffs (the study's model, sim_cat); "
                         "'smoothed' = gwpopulation SinglePeakSmoothedMassDistribution "
                         "as Ray et al. used, which suppresses the low-mass "
                         "pile-up. Requires --delta-m > 0.")
    ap.add_argument("--delta-m", type=float, default=5.0,
                    help="Smoothing scale for --mass-model smoothed. Default 5.0 "
                         "~ the LVK GWTC-3 PL+Peak inferred value (~4.8), and "
                         "calibrated here to reproduce real GWTC-3's low-mass "
                         "fraction (see sim_cat2_calibrate_delta_m.py).")
    ap.add_argument("--width-ref-snr", type=float, default=None,
                    help="Reference SNR normalising the mock measurement widths "
                         "(s_X = ref/rho * scale_X). GWMockCat's scales were "
                         "calibrated at 8, but `threshold_snr` doubles as this "
                         "reference, so thresholding at rho_thr=9 silently "
                         "widened every posterior by 9/8 (RUN_NOTES sec.8). "
                         "Pass 8.0 to restore the calibrated widths while "
                         "keeping the 9.0 detection threshold. Default None "
                         "keeps the old coupled behaviour.")
    ap.add_argument("--true-events-from", type=Path, default=None,
                    help="DIAGNOSTIC. Skip the population draw and take the true "
                         "(m1_src, m2_src, z) from this HDF5 instead, then run "
                         "them through the SAME detection and PE path as normal. "
                         "Lets an external event set (e.g. Ray's MDC truths in "
                         "sim_cat3_qfix/pool_truth.h5) be pushed through the real "
                         "pipeline at the real settings, so 'is it the events or "
                         "the pipeline?' can be answered without reimplementing "
                         "anything. Events are tiled to reach --n-draw, since "
                         "each still has to pass rho_obs >= rho_thr here. "
                         "Reads m1_src/m2_src/z, or m1/m2/z.")
    ap.add_argument("--true-pop-json", type=str, default=None,
                    help="SBC HOOK. Path to a JSON file, or a literal JSON "
                         "string, giving the true hyperparameters for this "
                         "pool: keys gamma, m_min, m_max, alpha, beta, mu, "
                         "sigma, f1. Added for sbc_suite/, which needs one "
                         "pool per prior-drawn Lambda_j instead of the single "
                         "fixed CFG['true_pop']. DEFAULT None reproduces the "
                         "fixed-Lambda behaviour bit-for-bit -- this flag is "
                         "purely additive and the existing MDC path is "
                         "unaffected when it is absent.")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.true_pop_json is None:
        pop = CFG["true_pop"]
    else:
        _raw = args.true_pop_json
        _txt = (Path(_raw).read_text() if Path(_raw).exists() else _raw)
        pop = json.loads(_txt)
        _need = set(CFG["true_pop"])
        _got = set(pop)
        if _got != _need:
            raise SystemExit(
                f"--true-pop-json keys {sorted(_got)} != required "
                f"{sorted(_need)} (missing {sorted(_need - _got)}, "
                f"unexpected {sorted(_got - _need)})")
        pop = {k: float(pop[k]) for k in CFG["true_pop"]}
        print(f"[cfg] true_pop OVERRIDDEN from {_raw if len(_raw) < 120 else 'inline JSON'}")
        print(f"[cfg]   {pop}")
    uncert = dict(CFG["uncert"])
    uncert["threshold_snr"] = CFG["rho_thr"]
    if args.width_ref_snr is not None:
        uncert["width_ref_snr"] = args.width_ref_snr
        print(f"[cfg] width_ref_snr = {args.width_ref_snr} (widths decoupled from "
              f"the rho_thr={CFG['rho_thr']} threshold; GWMockCat calibration = 8)")

    osnr_interp, ref_dist = interpolate_optimal_snr_grid(CFG["grid"])
    cosmo_dict = cosmo_utils.interp_cosmology()

    # ── draw + detect + PE samples: ONE GWMockCat call so the detection and
    #    the samples come from a single consistent generative pass ──────────
    if args.true_events_from is not None:
        with h5py.File(args.true_events_from, "r") as _f:
            _k1 = "m1_src" if "m1_src" in _f else "m1"
            _k2 = "m2_src" if "m2_src" in _f else "m2"
            _m1, _m2, _z = _f[_k1][:], _f[_k2][:], _f["z"][:]
        _R = int(np.ceil(args.n_draw / len(_m1)))
        m1t, m2t, zt = (np.tile(_m1, _R)[:args.n_draw],
                        np.tile(_m2, _R)[:args.n_draw],
                        np.tile(_z, _R)[:args.n_draw])
        print(f"[pool] EXTERNAL true events from {args.true_events_from} "
              f"({len(_m1)} unique, tiled x{_R} -> {len(m1t):,}); "
              f"population draw SKIPPED")
    elif args.mass_model == "smoothed":
        if args.delta_m <= 0:
            raise SystemExit("--mass-model smoothed needs --delta-m > 0 "
                             "(delta_m=0 is singular in gwpopulation's window)")
        print(f"[pool] drawing {args.n_draw:,} pre-selection events "
              f"(SMOOTHED mass model, delta_m={args.delta_m}) ...")
        m1t, m2t, zt = draw_population_smoothed(
            args.n_draw, pop, args.delta_m, CFG["z_max"], cosmo_dict, rng)
    else:
        print(f"[pool] drawing {args.n_draw:,} pre-selection events "
              f"(SHARP mass model) ...")
        m1t, m2t, zt = draw_population(args.n_draw, pop, CFG["z_max"], cosmo_dict, rng)
    Xefft = np.zeros(args.n_draw)          # non-spinning population
    print(f"[pool] m1=[{m1t.min():.2f},{m1t.max():.2f}] z=[{zt.min():.3f},{zt.max():.3f}]")

    n_raw = args.n_pe_samples * args.pe_oversample
    if args.mass_ratio_sampling == "q":
        from sim_cat_qsampling import generate_obs_from_true_list_q as _gen
        print(f"[pool] mass-ratio likelihood in q (scale uncert['q']={uncert['q']}) "
              f"-- corrects GWMockCat's eta bias at near-equal mass")
    else:
        _gen = pu.generate_obs_from_true_list
        print(f"[pool] mass-ratio likelihood in eta (GWMockCat native; "
              f"BIASED for q->1, see sim_cat_qsampling.py)")
    print(f"[pool] detecting (rho_obs >= {CFG['rho_thr']}) and generating "
          f"{n_raw:,} raw samples for the first {args.pool_size} detected ...")
    event_data, det = _gen(
        m1t, m2t, zt, Xefft, osnr_interp, cosmo_dict, rng,
        PEsamps=n_raw, num_evs=args.pool_size, uncert=uncert)

    n_det = len(det["m1"])
    print(f"[pool] detected {n_det:,} / {args.n_draw:,}  "
          f"(efficiency {n_det/args.n_draw:.3e})")
    if n_det < args.pool_size:
        raise SystemExit(f"only {n_det} detected < pool_size {args.pool_size}; "
                         f"increase --n-draw")

    sl = slice(0, args.pool_size)          # first N detected, in draw order (unbiased)
    pool = {k: np.asarray(det[k])[sl] for k in
            ("m1", "m2", "z", "lum_dist", "theta", "rho_obs", "rho_opt", "detected_index")}
    pool["rho_true"] = pool["rho_opt"] * pool["theta"]

    print(f"[pool] pool of {args.pool_size}: "
          f"m1=[{pool['m1'].min():.2f},{pool['m1'].max():.2f}] "
          f"z=[{pool['z'].min():.3f},{pool['z'].max():.3f}]")
    print(f"[pool] up-scatter: N(rho_true < {CFG['rho_thr']}) = "
          f"{int((pool['rho_true'] < CFG['rho_thr']).sum())}, "
          f"N(rho_opt < {CFG['rho_thr']}) = {int((pool['rho_opt'] < CFG['rho_thr']).sum())}")

    # ── transform samples to (m1,m2,dL,z) and reweight to the LVK PE prior ──
    ptheta = build_ptheta(np.random.default_rng(args.seed + 1))
    dH_Gpc = cosmo_dict["dH"] / 1000.0

    out_pe = args.outdir / "pe_samples.h5"
    ess_all, m1_lo, m1_hi, m2_lo, m2_hi = [], np.inf, -np.inf, np.inf, -np.inf
    with h5py.File(out_pe, "w") as fout:
        for i in range(args.pool_size):
            d = event_data[f"event_{i}"].values          # (5, n_raw)
            Mc, eta, rho, Xeff, theta = d[0], d[1], d[2], d[3], d[4]

            Mtot = Mc / eta ** 0.6
            root = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
            m1d = Mtot * (1.0 + root) / 2.0
            m2d = Mtot * (1.0 - root) / 2.0

            A = osnr_interp((m1d, m2d, Xeff))
            dl_Gpc = theta * A / rho
            dl_Mpc = dl_Gpc * 1000.0
            ok = np.isfinite(dl_Mpc) & (dl_Mpc > 0) & (dl_Mpc < 46000.0) & (m2d > 0)
            z = np.full_like(dl_Mpc, np.nan)
            z[ok] = cosmo_dict["z_at_dL"](dl_Mpc[ok])

            # Jacobian of the GWMockCat variable transformation, |d(Mc,X)/d(m1,m2)|,
            # where X is the coordinate the mass-ratio likelihood was SAMPLED in.
            # It must match --mass-ratio-sampling or the reweighting is wrong:
            #   eta:  (m1-m2)*eta^0.6/(m1+m2)^2   -- VANISHES at q=1 (eta is 2-to-1
            #         in q and the map degenerates there), fine only because
            #         eta-sampling's truncation keeps samples strictly off q=1.
            #   q:    eta^0.6*(1+q)/m1            -- finite everywhere, including
            #         q=1 exactly, which is where q-sampling piles samples up.
            # Verified against finite-difference |d(Mc,q)/d(m1,m2)| to 4 dp at
            # q = 0.33, 0.83, 0.97, 0.9996 and exactly 1.0.
            if args.mass_ratio_sampling == "q":
                q_samp = m2d / m1d
                mass_jac = eta ** 0.6 * (1.0 + q_samp) / m1d
            else:
                mass_jac = (m1d - m2d) * eta ** 0.6 / (m1d + m2d) ** 2
            ddl_dz = dl_Gpc / (1 + z) + (1 + z) * dH_Gpc * cosmo_dict["inv_efunc"](
                np.clip(z, 0, 4.99))
            dist_jac = theta * A / dl_Gpc ** 2 * ddl_dz

            # LVK prior: uniform in detector-frame component masses, p(dL) ~ dL^2.
            # Applied ONCE (transforms.py squares this -- known bug, see docstring).
            w = (dl_Gpc ** 2) * ptheta(theta) * ddl_dz / (mass_jac * dist_jac)
            w = np.where(ok & np.isfinite(w) & (w > 0), w, 0.0)
            if w.sum() <= 0:
                raise SystemExit(f"event {i}: all reweighting weights zero")
            w /= w.sum()
            ess = 1.0 / np.sum(w ** 2)
            ess_all.append(ess)

            idx = rng.choice(len(w), size=args.n_pe_samples, p=w, replace=True)
            zs, dls = z[idx], dl_Mpc[idx]
            m1_det, m2_det = m1d[idx], m2d[idx]
            # declared PE prior evaluated directly in the coordinates stored:
            # uniform in detector-frame component masses (constant), p(dL) ~ dL^2.
            # No extra Jacobian needed -- unlike the source-frame output, these
            # ARE the coordinates the LVK convention is stated in.
            prior = dls ** 2

            m1_lo, m1_hi = min(m1_lo, m1_det.min()), max(m1_hi, m1_det.max())
            m2_lo, m2_hi = min(m2_lo, m2_det.min()), max(m2_hi, m2_det.max())

            g = fout.create_group(f"event_{i:03d}")
            for name, arr in (("m1_det", m1_det), ("m2_det", m2_det), ("dL", dls),
                              ("z", zs), ("prior", prior)):
                g.create_dataset(name, data=arr, compression="gzip", shuffle=True)
            g.attrs["pool_index"] = i
            g.attrs["m1_true"] = pool["m1"][i]
            g.attrs["m2_true"] = pool["m2"][i]
            g.attrs["z_true"] = pool["z"][i]
            g.attrs["dL_true"] = pool["lum_dist"][i]
            g.attrs["rho_opt"] = pool["rho_opt"][i]
            g.attrs["rho_true"] = pool["rho_true"][i]
            g.attrs["rho_obs"] = pool["rho_obs"][i]
            g.attrs["Theta_true"] = pool["theta"][i]
            g.attrs["reweight_ESS"] = ess
            if (i + 1) % 50 == 0:
                print(f"[pe]   {i+1}/{args.pool_size} events "
                      f"(median ESS so far {np.median(ess_all):.0f})", flush=True)

        fout.attrs["n_events"] = args.pool_size
        fout.attrs["n_samples"] = args.n_pe_samples
        fout.attrs["n_raw_samples"] = n_raw
        fout.attrs["mass_ratio_sampling"] = args.mass_ratio_sampling
        fout.attrs["mass_model"] = args.mass_model
        fout.attrs["delta_m"] = (args.delta_m if args.mass_model == "smoothed" else 0.0)
        fout.attrs["prior"] = "LVK: uniform in detector-frame component masses, " \
                              "p(dL) propto dL^2; stored directly in " \
                              "(m1_det,m2_det,dL) -- same coords, no extra Jacobian; " \
                              "unnormalised. z stored alongside for source-frame back-transform."

    ess_all = np.array(ess_all)
    print(f"[pe] reweighting ESS: min={ess_all.min():.0f} median={np.median(ess_all):.0f} "
          f"max={ess_all.max():.0f}  (of {n_raw:,} raw)")
    print(f"[pe] sample bounds (detector frame): "
          f"m1_det=[{m1_lo:.3f},{m1_hi:.3f}] m2_det=[{m2_lo:.3f},{m2_hi:.3f}]")
    print(f"[pe] saved -> {out_pe}")

    # ── pool file ──────────────────────────────────────────────────────────
    try:
        commit = subprocess.check_output(
            ["git", "-C", "/home/manchun.yeung/population/mock-PE", "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"

    out_pool = args.outdir / "mdc_pool.h5"
    with h5py.File(out_pool, "w") as f:
        for k in ("m1", "m2", "z", "lum_dist", "theta", "rho_opt", "rho_true", "rho_obs"):
            f.create_dataset(k, data=pool[k], compression="gzip", shuffle=True)
        f.attrs["n_events"] = args.pool_size
        f.attrs["threshold_type"] = "rho_obs"
        f.attrs["rho_thr"] = CFG["rho_thr"]
        f.attrs["psd_name"] = "aligo_O3actual_H1 + aligo_O3actual_L1 (H1L1 network)"
        f.attrs["snr_grid"] = CFG["grid"]
        f.attrs["gwmockcat_commit"] = commit
        f.attrs["seed"] = args.seed
        f.attrs["n_draw"] = args.n_draw
        f.attrs["n_detected"] = n_det
        f.attrs["true_pop"] = json.dumps(pop)
        f.attrs["z_max"] = CFG["z_max"]
        f.attrs["uncert"] = json.dumps(uncert)
        f.attrs["mass_ratio_sampling"] = args.mass_ratio_sampling
        f.attrs["mass_model"] = args.mass_model
        f.attrs["delta_m"] = (args.delta_m if args.mass_model == "smoothed" else 0.0)
    print(f"[pool] saved -> {out_pool}")

    with open(args.outdir / "stage3_stats.json", "w") as f:
        json.dump(dict(n_draw=args.n_draw, n_detected=int(n_det),
                       efficiency=float(n_det / args.n_draw),
                       pool_size=args.pool_size,
                       n_rho_true_below_thr=int((pool["rho_true"] < CFG["rho_thr"]).sum()),
                       n_rho_opt_below_thr=int((pool["rho_opt"] < CFG["rho_thr"]).sum()),
                       ess_min=float(ess_all.min()), ess_median=float(np.median(ess_all)),
                       m1_range=[float(m1_lo), float(m1_hi)],
                       m2_range=[float(m2_lo), float(m2_hi)],
                       gwmockcat_commit=commit, seed=args.seed), f, indent=2)


if __name__ == "__main__":
    main()
