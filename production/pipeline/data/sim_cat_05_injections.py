#!/usr/bin/env python
"""
sim_cat_05_injections.py
=========================
Stage 5: matched injection set for the selection function xi(Lambda).

NON-NEGOTIABLE per gwmockcat_sample_generation.md: the injections use the SAME
rho_obs machinery, SAME SNR grid / PSDs, SAME Finn & Chernoff Theta
distribution and SAME threshold as the event pool (stage 3), so the selection
function and the event samples come from one generative process.

Reference (draw) distribution -- deliberately broader than the population
support (m1,m2 in [6,70], z in [0,1.9]) with margin, and matched to
endo3_bbhpop-LIGO-T2100113-v12.hdf5's OWN generating distribution (file attrs
pow_mass1=-2.35, pow_mass2=1.0, pow_z=1.0) and to this repo's already-validated
canonical proposal (make_diagnostic_plots.py::log_pdraw_marginal, which uses
the same alpha=-2.35/1.0 and dV_of_z_normed(z, Om0Planck, gamma=1.0)):

    p(m1) ~ m1^-2.35        on [m_ref_min, m_ref_max] = [2, 100]
    p(m2|m1) ~ m2^1         on [m_ref_min, m1]
    p(z) ~ dVc/dz            on [0, z_ref_max] = [1e-4, 1.9]
             (kappa_ref=0 -- the (1+z)^(gamma-1) factor cancels at gamma=1,
              matching pow_z=1.0's convention exactly)

`sampling_pdf` is written as the MARGINAL density over (m1, m2, z) ONLY --
no spin or angle factors.  The hierarchical estimator needs p_draw(m1,m2,z)
as the denominator; a joint density carrying extra dimensions would leave an
uncancelled constant that biases inferred rates (this is exactly the bug
documented in CLAUDE_CODE_TASK_selection.md for the endo3 set, where
`sampling_pdf` is the full joint including spins).  The covered variables are
recorded explicitly in the file attrs.

Only the DETECTED rows are written to disk -- exactly endo3_v12's own
convention (it stores its n_accepted=284,672 found injections, not the
73,957,576 total_generated draws; the total is kept as a single attribute).
The estimator's denominator only ever needs Ndraw = n_total_drawn as a scalar,
never the missed rows, so storing every draw wastes disk with no benefit
(previously 4.14 GB for 75M rows at n_inj=75e6; now ~7 MB for the ~104k
detected). There is no `detected` column any more since every stored row is,
by construction, detected.

Theta is drawn per injection but is NOT part of the population parameter
space, so it does not enter sampling_pdf -- it is part of the detection model,
exactly as for the events.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import h5py

sys.path.insert(0, "/home/manchun.yeung/population/mock-PE")
from GWMockCat import cosmo_utils                                   # noqa: E402
from GWMockCat.vt_utils import interpolate_optimal_snr_grid, draw_thetas  # noqa: E402

CFG = dict(
    n_inj=75_000_000,      # ~100k detected at current efficiency (~1.4e-3);
                           # endo3_v12 itself has 100,136 found at snr_net>9
    m_ref_min=2.0, m_ref_max=100.0,
    alpha_ref=2.35,        # p(m1) ~ m1^-alpha_ref
    beta_ref=1.0,          # p(m2|m1) ~ m2^beta_ref
    kappa_ref=0.0,         # p(z) ~ dVc/dz (1+z)^kappa_ref; 0 matches endo3's
                           # pow_z=1.0 / this repo's dV_of_z_normed(gamma=1.0)
    z_ref_max=1.9,
    rho_thr=9.0,
    snr_sigma=1.0,
    seed=20260803,
    grid="sim_cat/snr_grid_O3actual_H1L1_Xeff.h5",
    chunk=2_000_000,
)

ap = argparse.ArgumentParser()
ap.add_argument("--n-inj", type=int, default=CFG["n_inj"])
ap.add_argument("--seed", type=int, default=CFG["seed"])
ap.add_argument("--outdir", type=Path, default=Path("sim_cat"))
ap.add_argument("--chunk", type=int, default=CFG["chunk"])
args = ap.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed + 4242)
osnr_interp, _ = interpolate_optimal_snr_grid(CFG["grid"])
cosmo_dict = cosmo_utils.interp_cosmology()

# z grid for the reference redshift distribution and its normalisation
zg = np.linspace(1e-4, CFG["z_ref_max"], 4000)
pz_un = cosmo_dict["diff_comoving_vol"](zg) * (1 + zg) ** CFG["kappa_ref"]
pz_norm = np.trapezoid(pz_un, zg)
zcdf = np.concatenate([[0.0], np.cumsum(0.5 * (pz_un[1:] + pz_un[:-1]) * np.diff(zg))])
zcdf /= zcdf[-1]

mlo, mhi, a_ref, b_ref = (CFG["m_ref_min"], CFG["m_ref_max"],
                          CFG["alpha_ref"], CFG["beta_ref"])
one_ma = 1.0 - a_ref          # p(m1) ~ m1^-a  =>  CDF via m^(1-a)
b1 = 1.0 + b_ref


def draw_chunk(n, rng):
    """Draw (m1, m2, z) from the broad reference distribution + its marginal pdf."""
    u = rng.random(n)
    m1 = (mlo ** one_ma + u * (mhi ** one_ma - mlo ** one_ma)) ** (1.0 / one_ma)
    p_m1 = one_ma * m1 ** (-a_ref) / (mhi ** one_ma - mlo ** one_ma)

    v = rng.random(n)
    m2 = (mlo ** b1 + v * (m1 ** b1 - mlo ** b1)) ** (1.0 / b1)
    p_m2 = b1 * m2 ** b_ref / (m1 ** b1 - mlo ** b1)

    z = np.interp(rng.random(n), zcdf, zg)
    p_z = cosmo_dict["diff_comoving_vol"](z) * (1 + z) ** CFG["kappa_ref"] / pz_norm

    return m1, m2, z, p_m1 * p_m2 * p_z


print(f"[inj] drawing {args.n_inj:,} injections in chunks of {args.chunk:,}")
# Store ONLY the detected rows -- exactly endo3's own convention (n_accepted
# rows on disk, total_generated kept as a single attribute). The estimator
# never touches the missed draws, only Ndraw = n_total_drawn as a scalar, so
# writing all of them wasted 624x the disk (4.14 GB for 75M rows -> ~7 MB for
# the ~104k detected). Filtering per-chunk (not after concatenating all
# n_inj draws) also caps peak memory, independent of how large n_inj gets.
cols = {k: [] for k in ("m1", "m2", "z", "dL", "sampling_pdf",
                        "rho_opt", "rho_obs", "Theta")}
n_done = 0
n_det_running = 0
while n_done < args.n_inj:
    n = min(args.chunk, args.n_inj - n_done)
    m1, m2, z, pdraw = draw_chunk(n, rng)
    dL_Mpc = cosmo_dict["dL"](z)
    m1d, m2d = m1 * (1 + z), m2 * (1 + z)
    Xeff = np.zeros(n)                                   # non-spinning, as the events

    # identical detection model to stage 3 (GWMockCat posterior_utils)
    A = osnr_interp((m1d, m2d, Xeff))
    rho_opt = A / (dL_Mpc / 1000.0)                      # SNR at Theta = 1
    theta = draw_thetas(n, rng)
    rho_obs = rho_opt * theta + CFG["snr_sigma"] * rng.normal(size=n)

    det = rho_obs >= CFG["rho_thr"]
    for k, v in (("m1", m1), ("m2", m2), ("z", z), ("dL", dL_Mpc),
                 ("sampling_pdf", pdraw), ("rho_opt", rho_opt),
                 ("rho_obs", rho_obs), ("Theta", theta)):
        cols[k].append(v[det].astype(np.float64))
    n_det_running += int(det.sum())
    n_done += n
    print(f"[inj]   {n_done:,}/{args.n_inj:,}  (detected so far: {n_det_running:,})",
          flush=True)

data = {k: np.concatenate(v) for k, v in cols.items()}
del cols
n_det = data["m1"].shape[0]
print(f"[inj] detected {n_det:,} / {args.n_inj:,}  "
      f"(efficiency {n_det/args.n_inj:.3e})")
rho_true = data["rho_opt"] * data["Theta"]
print(f"[inj] up-scatter among detected: N(rho_true < {CFG['rho_thr']}) = "
      f"{int((rho_true < CFG['rho_thr']).sum()):,}")

out = args.outdir / "injections.h5"
with h5py.File(out, "w") as f:
    for k, v in data.items():
        f.create_dataset(k, data=v, compression="gzip", shuffle=True)
    f.attrs["n_total_drawn"] = args.n_inj
    f.attrs["n_detected"] = n_det
    f.attrs["rho_thr"] = CFG["rho_thr"]
    f.attrs["threshold_type"] = "rho_obs"
    f.attrs["storage"] = "DETECTED ROWS ONLY (matches endo3's n_accepted " \
                         "convention); n_total_drawn is the selection-integral " \
                         "denominator, stored as a scalar attribute, not a column"
    f.attrs["psd_name"] = "aligo_O3actual_H1 + aligo_O3actual_L1 (H1L1 network)"
    f.attrs["snr_grid"] = CFG["grid"]
    f.attrs["seed"] = args.seed
    f.attrs["sampling_pdf_variables"] = "m1_source, m2_source, z (MARGINAL; " \
                                        "no spin or angle factors)"
    f.attrs["reference_distribution"] = json.dumps(
        dict(p_m1=f"m1^-{a_ref} on [{mlo},{mhi}]",
             p_m2=f"m2^{b_ref} on [{mlo},m1]",
             p_z=f"dVc/dz (1+z)^{CFG['kappa_ref']} on [0,{CFG['z_ref_max']}]"))
print(f"[inj] saved -> {out}  ({out.stat().st_size/1e6:.2f} MB)")
