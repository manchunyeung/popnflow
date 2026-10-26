#!/usr/bin/env python
"""
sim_cat_qsampling.py
=====================
Drop-in replacement for `GWMockCat.posterior_utils.generate_obs_from_true_list`
that generates the mock mass-ratio likelihood in **q = m2/m1** instead of in the
symmetric mass ratio **eta = q/(1+q)^2**.

WHY
---
GWMockCat draws `eta_obs = eta + seta * truncnorm(..., upper=(0.25-eta)/seta)`.
For a near-equal-mass binary this is pathological in two compounding ways:

  1. **One-sided truncation.**  eta = 0.25 at q = 1 is the *domain boundary* of
     eta, so the upper truncation limit is (0.25 - 0.25)/seta = 0 and the
     scatter can only ever go DOWNWARD -- a systematic, not a symmetric, error.

  2. **Vanishing Jacobian.**  deta/dq = (1-q)/(1+q)^3 -> 0 as q -> 1, so a tiny
     error in eta maps to an enormous error in q.  Measured on sim_cat
     catalog 0: a typical downward eta scatter of 0.0155 at rho_obs = 10.2 maps
     to q ~ 0.60, dragging m2 from 6.04 down to 3.63.

Measured consequence on the generated catalog (median PE value vs truth):

    q_true bin      N     m1 bias   m2 bias    q bias
    0.00-0.70      19      +0.042    +0.015    -0.021
    0.70-0.85      12      +0.110    -0.070    -0.129
    0.85-0.95      19      +0.245    -0.148    -0.293
    0.95-1.01      19      +0.204    -0.139    -0.287

i.e. a *systematic median bias*, not scatter, switching on exactly as q -> 1.
The MDC population makes this maximally bad: m_min=6 with a steep alpha=4
produces many low-m1 events whose q_min = m_min/m1 is forced near 1, and
beta=1.5 favours q -> 1 anyway; 31 of 69 events in catalog 0 have q_true > 0.9.
That contaminates m_min (via m2 spilling below the cutoff), beta (via the q
distribution) and alpha (via m1).

GWMockCat's own README flags this: "you may find too few effective samples at
high (symmetric) mass ratios ... don't take mass ratio results too seriously."
The authors' planned fix is an "exponential interim prior in symmetric mass
ratio"; this module takes the equivalent route of leaving eta behind entirely.

THE FIX
-------
Sample q directly from a truncated normal on [q_min_support, 1]:

  * q = 1 is a *genuine physical boundary* (m2 <= m1 by definition), not a
    coordinate artifact, so one-sidedness there is correct rather than spurious;
  * dq/dq = 1, so there is no degeneracy amplification.

Residual downward bias at exactly q_true = 1 is physical and expected -- real PE
for an equal-mass system also has a q posterior that rails at 1 and extends
down, so its median is below 1 too.

Everything else (chirp mass, Theta, rho, spin/psi handling, the two-stage
"draw an observed value, then sample the posterior around it" structure) is
copied from GWMockCat unchanged, and the returned arrays are converted back to
eta so that downstream code needs no modification whatsoever.

SNR-SCALING (a second issue this fix exposed)
---------------------------------------------
GWMockCat ties every uncertainty to 1/rho.  For q that is empirically wrong:
the real GWTC-3 90% credible width of q is nearly SNR-INDEPENDENT --

    rho bin :   8-11    11-14   14-20   20+
    real    :   0.584   0.571   0.468   0.493

-- because q is degeneracy-dominated, not SNR-dominated.  A pure 1/rho model
cannot reproduce that: matching at rho~10 leaves it ~3x too narrow by rho~30.

The eta parameterization masked this, because the deta/dq -> 0 amplification
grows toward equal mass and partly offsets the 1/rho shrinkage -- but it paid
for that with the large systematic bias this module exists to remove.  So

    s_q = max( threshold_snr/rho_obs * uncert["q"],  uncert["q_floor"] )

with a floor representing the degeneracy-dominated regime.  Set
`q_floor = 0.0` to recover the pure 1/rho form.

THE q=1 BOUNDARY (added 2026-08-04 -- second, separate defect)
--------------------------------------------------------------
Switching eta -> q removed the Jacobian pathology but left a large residual
bias, measured at -0.179 in PE-median q against -0.071 for real bilby PE on the
SAME events (Ray's MDC; see sim_cat3_from_ray.py). Decomposing the pipeline on
Ray's 276 true q values:

    stage 1  q_true -> qobs        -0.067    <- dominant
    stage 2  qobs   -> sample med  -0.026
    stage 3  reweight to LVK prior -0.022
    TOTAL                          -0.127

and it is concentrated where the boundary bites: -0.145 for q_true in
[0.85,0.95], -0.164 in [0.95,1.0], against -0.05 in [0.70,0.85].

CAUSE. `qobs` was drawn from a normal TRUNCATED at q=1. But `qobs` is a noisy
point estimate, not a physical mass ratio -- it is perfectly allowed to land
above 1. Truncating it discards the upper half of the measurement scatter for
near-equal-mass events, and since 34% of Ray's events have q_true > 0.9, that
drags the whole catalog's mass ratios down.

FIX. Draw `qobs` from an UNtruncated normal, and impose q<=1 only on the
POSTERIOR samples -- the physically correct place, since q>1 has no support.
When qobs > 1 the posterior then piles up AT q=1, which is what real PE does.
Measured against bilby on the same events:

    q_true bin      bilby    old (truncate)   new (prior)
    0.70-0.85      -0.033       -0.066          -0.023
    0.85-0.95      -0.133       -0.143          -0.087
    0.95-1.00      -0.198       -0.211          -0.120
    overall        -0.058       -0.131          -0.074

Overall bias drops from 2.3x bilby's to 1.3x, and the excess over bilby at high
q -- where the beta information lives -- is removed. Note bilby's own bias is
POSITIVE at q_true < 0.7 (+0.06) while the mock stays negative; that residual
is a different effect (real PE pulls low-q events toward equal mass) and is not
addressed here. 90% q width: old 0.497, new 0.457, bilby 0.525.

Reflection about q=1 was also tried and is NOT enough (overall -0.109).

CALIBRATION
-----------
`uncert["q"]` and `uncert["q_floor"]` are set so the mock 90% q width tracks
real GWTC-3 across SNR.  NOTE this is a deliberate calibration of a NEW
parameterization (some scale had to be chosen), NOT tuning results to taste --
gwmockcat_sample_generation.md's "do not auto-tune to match, just report"
applies to the validation figure, which remains a report-only diagnostic.
Verify both bias and width with sim_cat_qsampling_check.py.
"""
import numpy as np
import xarray as xr
from astropy import units as u
from scipy.stats import truncnorm

from GWMockCat import vt_utils as vt
from GWMockCat import spin_utils
from GWMockCat.posterior_utils import (
    generate_detected_from_true_list, mchirp, etafunc,
)

# GWMockCat's uncert_default plus the new q scale (and the Xeff key its own
# generate_obs_from_true_list dereferences but omits from the default dict).
uncert_default_q = {
    "threshold_snr": 8,
    "snr": 1.0,
    "mc": 0.08,
    "Theta": 0.15 * 1.4,
    "eta": 0.022,      # retained only for reference/back-compat; unused here
    "q": 0.22,         # <-- the replacement scale, in q rather than eta
    "q_floor": 0.176,  # see SNR-SCALING note below; 0.0 recovers pure 1/rho
    "Xeff": 0.1,
    # Where the q<=1 boundary is imposed. "prior" (default, correct) puts it on
    # the posterior only; "truncate" also truncates the noise draw, reproducing
    # the pre-2026-08-04 behaviour. See "THE q=1 BOUNDARY" below.
    "q_boundary": "prior",
}

Q_MIN_SUPPORT = 0.02   # floor on q; eta = q/(1+q)^2 -> 0 blows up Mtot = Mc/eta^0.6


def eta_of_q(q):
    """Symmetric mass ratio from mass ratio. Exact inverse of the downstream
    (Mc, eta) -> (m1, m2) transform: m1 = Mtot/(1+q), m2 = Mtot*q/(1+q)."""
    return q / (1.0 + q) ** 2


def _trunc_draw(mean, scale, lo, hi, size, rng):
    """Truncated normal on [lo, hi] with the given mean/scale (vectorised over
    mean/scale when size matches, scalar-mean when generating one event's
    posterior)."""
    a = (lo - mean) / scale
    b = (hi - mean) / scale
    return truncnorm.rvs(a, b, loc=mean, scale=scale, size=size, random_state=rng)


def generate_obs_from_true_list_q(
    m1t, m2t, zt, Xefft, osnr_interp, cosmo_dict, rng,
    osnr_interp_dist=u.Gpc, PEsamps=None, num_evs=np.inf, t=None,
    uncert=uncert_default_q,
):
    """q-sampling counterpart of GWMockCat.posterior_utils.generate_obs_from_true_list.

    Signature, return types and the returned parameter names are identical, so
    this substitutes directly. `eta_samps` in the returned Dataset is derived
    from the sampled q via eta = q/(1+q)^2.
    """
    detected_dict = generate_detected_from_true_list(
        m1t, m2t, zt, Xefft, osnr_interp, cosmo_dict, rng, osnr_interp_dist, t, uncert
    )
    m1, m2, z, Xeff = (detected_dict["m1"], detected_dict["m2"],
                       detected_dict["z"], detected_dict["Xeff"])

    mc = mchirp(m1, m2) * (1 + z)          # detector frame
    eta = etafunc(m1, m2)                  # true eta, still needed for the spin block
    q_true = m2 / m1                       # true mass ratio  (<= 1 by construction)
    rho_obs = detected_dict["rho_obs"]

    # `threshold_snr` does DOUBLE DUTY in GWMockCat: detection threshold AND the
    # reference SNR normalising every measurement width (s_X = ref/rho * scale_X).
    # The md spec forced the threshold to 9, but GWMockCat's width scales were
    # calibrated at 8 -- so thresholding at 9 silently widened all posteriors by
    # 9/8 = 12.5% (RUN_NOTES sec.8, "Not tuned"). `width_ref_snr` decouples the
    # two roles; omit it to keep the old behaviour (ref = threshold).
    width_ref = uncert.get("width_ref_snr", uncert["threshold_snr"])

    # --- chirp mass: unchanged from GWMockCat -----------------------------
    smc = width_ref / rho_obs * uncert["mc"]
    mcobs = rng.lognormal(mean=np.log(mc), sigma=smc)

    # --- mass ratio: THE FIX. truncated normal in q on [Q_MIN_SUPPORT, 1] --
    sq = np.maximum(width_ref / rho_obs * uncert["q"],
                    uncert.get("q_floor", 0.0))
    # `q_boundary` controls WHERE the q<=1 boundary is imposed (see module
    # docstring, "THE q=1 BOUNDARY"). 'prior' is correct and is the default;
    # 'truncate' reproduces the older, biased behaviour.
    if uncert.get("q_boundary", "prior") == "truncate":
        qobs = _trunc_draw(q_true, sq, Q_MIN_SUPPORT, 1.0, len(m1), rng)
    else:
        # qobs is a NOISY POINT ESTIMATE, not a physical mass ratio -- it is
        # allowed to land above 1. Truncating it here discards the upper half
        # of the scatter for near-equal-mass events and drags every posterior
        # down; the boundary belongs on the posterior instead (below).
        qobs = rng.normal(q_true, sq)
    etaobs = eta_of_q(np.clip(qobs, Q_MIN_SUPPORT, 1.0))

    # --- Theta: unchanged --------------------------------------------------
    st = width_ref / rho_obs * uncert["Theta"]
    theta = detected_dict["theta"]
    tobs = _trunc_draw(theta, st, 0.0, 1.0, len(m1), rng)

    # --- spins: unchanged (psi-space, per GWMockCat's own comment) ---------
    uncert_psi = spin_utils.dpsi_from_dXeff_neglect_Xa(uncert["Xeff"], eta)
    psi = spin_utils.psi_from_chi_eff_and_eta_neglect_chi_a(Xeff, eta)
    spsi = width_ref / rho_obs * uncert_psi
    psiobs = spin_utils.truncated_normal([-4.2, -1.2], psi, spsi, size=len(m1))
    Xeffobs = spin_utils.chi_eff_from_psi_and_eta_neglect_chi_a(psiobs, etaobs)

    if PEsamps is None:
        return {
            "Mc_det_obs": mcobs, "eta_obs": etaobs, "rho_obs": rho_obs,
            "Xeff_obs": Xeffobs, "t_obs": tobs,
        }, detected_dict

    num_evs = int(min(len(m1), num_evs))
    param_names = ["Mc_det_samps", "eta_samps", "rho_samps",
                   "Xeff_samps", "theta_samps"]
    event_data = xr.Dataset(
        coords={"parameter": param_names, "PE_sample": np.arange(PEsamps)}
    )
    for i in range(num_evs):
        mc_samps = rng.lognormal(mean=np.log(mcobs[i]), sigma=smc[i], size=PEsamps)

        # Posterior in q around the observed q, same support. THIS is the right
        # place for the q<=1 boundary: q>1 is unphysical so the posterior has no
        # support there, and when qobs > 1 the samples pile up AT q=1 -- which is
        # what real PE does for near-equal-mass systems.
        q_samps = _trunc_draw(qobs[i], sq[i], Q_MIN_SUPPORT, 1.0, PEsamps, rng)
        eta_samps = eta_of_q(q_samps)

        theta_samps = _trunc_draw(tobs[i], st[i], 0.0, 1.0, PEsamps, rng)
        rho_samps = truncnorm(
            (0.0 - rho_obs[i]) / uncert["snr"], np.inf,
            loc=rho_obs[i], scale=uncert["snr"],
        ).rvs(size=PEsamps, random_state=rng)

        psi_samps = spin_utils.truncated_normal(
            [-4.2, -1.2], psiobs[i], spsi[i], size=PEsamps
        )
        Xeff_samps = spin_utils.chi_eff_from_psi_and_eta_neglect_chi_a(
            psi_samps, eta_samps
        )

        event_data = event_data.assign({
            f"event_{i}": xr.DataArray(
                np.vstack([mc_samps, eta_samps, rho_samps, Xeff_samps, theta_samps]),
                dims=["parameter", "PE_sample"],
            )
        })
    return event_data, detected_dict
