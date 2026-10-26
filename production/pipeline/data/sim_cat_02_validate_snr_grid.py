#!/usr/bin/env python
"""
sim_cat_02_validate_snr_grid.py
================================
Stage 2: validate the H1+L1 O3-actual SNR grid (stage 1) against
endo3_bbhpop-LIGO-T2100113-v12.hdf5's OWN recorded per-injection SNRs.

This is what turns "use the GWTC-3 sensitivity of endo3_v12" into a checked
statement rather than an assumption.  The pipeline models the network SNR as

    rho_net  =  (A_net(m1det, m2det, Xeff) / dL_Gpc) * Theta,     Theta ~ Finn & Chernoff

where A_net = sqrt(A_H1^2 + A_L1^2) is the face-on/overhead amplitude from the
O3-actual PSDs.  Two things must hold for that to reproduce endo3:

  (A) AMPLITUDE.  Invert endo3's own injections for their implied projection
      factor,  Theta_eff = rho_net_endo3 * dL_Gpc / A_net(m1det, m2det, Xeff).
      A_net assumes BOTH detectors are simultaneously optimally oriented, which
      is geometrically impossible, so Theta_eff must lie strictly below 1.
      A significant excess above 1 means the amplitude model is too quiet.

  (B) PROJECTION DISTRIBUTION.  The empirical Theta_eff distribution should
      track GWMockCat's `vt_utils.draw_thetas` (Finn & Chernoff 1993).
      endo3 is truncated at rho_opt >= 6, which preferentially removes LOW
      Theta, so the subsample must be chosen in a way that does NOT itself
      depend on Theta.  Selecting on the recorded rho_net is exactly such a
      Theta-dependent cut and biases the comparison badly (it keeps only
      high-Theta events at any given amplitude).  Instead select on

          rho_max = A_net / dL_Gpc          (the SNR the event would have at
                                             Theta = 1 -- Theta-independent)

      For rho_max > C, every event with Theta > 6/C survives the endo3
      truncation, so within that Theta range the retained sample is unbiased
      and can be compared against Finn & Chernoff restricted to the same range.

      NOTE ON THE NETWORK: F&C is a single-detector result, but H1 and L1 are
      nearly co-aligned, so the H1+L1 network behaves close to a single
      effective detector.  The check below confirms this empirically against
      endo3 rather than assuming it.

Output: sim_cat/validation_snr_grid.pdf  + printed statistics.
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import RegularGridInterpolator

sns.set_context("talk", font_scale=1.0)
sns.set_style("ticks")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

ap = argparse.ArgumentParser()
ap.add_argument("--grid", type=Path, default=Path("sim_cat/snr_grid_O3actual_H1L1_Xeff.h5"))
ap.add_argument("--injection-file", type=Path,
                default=Path("input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5"))
ap.add_argument("--outdir", type=Path, default=Path("sim_cat"))
ap.add_argument("--rho-max-cut", type=float, default=60.0,
                help="Theta-INDEPENDENT selection: keep injections whose "
                     "rho_max = A_net/dL exceeds this. Every such event with "
                     "Theta > 6/cut survives endo3's rho>=6 truncation, so the "
                     "Theta distribution is unbiased above that value.")
ap.add_argument("--seed", type=int, default=20260803)
args = ap.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)

# ── load grid, build the same interpolator GWMockCat will use ──────────────
with h5py.File(args.grid, "r") as f:
    ms = f["ms"][:]
    xeffs = f["Xeff"][:]
    snr_grid = f["SNR"][:]
    ref_dist = f.attrs["D_L"]
print(f"[grid] {args.grid.name}: ms=[{ms.min():.1f},{ms.max():.1f}] "
      f"Xeff=[{xeffs.min():.2f},{xeffs.max():.2f}] ref D_L={ref_dist}")
osnr_interp = RegularGridInterpolator((ms, ms, xeffs), snr_grid,
                                      bounds_error=False, fill_value=None)

# ── endo3 injections ───────────────────────────────────────────────────────
with h5py.File(args.injection_file, "r") as fh:
    inj = fh["injections"]
    m1det = inj["mass1"][:].astype(np.float64)
    m2det = inj["mass2"][:].astype(np.float64)
    dL_Mpc = inj["distance"][:].astype(np.float64)
    s1z = inj["spin1z"][:].astype(np.float64)
    s2z = inj["spin2z"][:].astype(np.float64)
    snr_h = inj["optimal_snr_h"][:].astype(np.float64)
    snr_l = inj["optimal_snr_l"][:].astype(np.float64)
    snr_net = inj["optimal_snr_net"][:].astype(np.float64)

xeff = (m1det * s1z + m2det * s2z) / (m1det + m2det)
dL_Gpc = dL_Mpc / 1000.0
print(f"[endo3] {len(m1det):,} injections; m1det=[{m1det.min():.1f},{m1det.max():.1f}] "
      f"Xeff=[{xeff.min():.2f},{xeff.max():.2f}]")
print(f"[endo3] recorded rho_net min={snr_net.min():.2f} (file is pre-truncated at 6)")

# ── (A) amplitude check ────────────────────────────────────────────────────
A_net = osnr_interp(np.column_stack([m1det, m2det, xeff]))
theta_eff = snr_net * dL_Gpc / A_net

frac_gt1 = float((theta_eff > 1.0).mean())
print("\n[check A] implied projection factor Theta_eff = rho_net * dL / A_net")
print(f"[check A]   min={theta_eff.min():.4f}  median={np.median(theta_eff):.4f}  "
      f"max={theta_eff.max():.4f}")
print(f"[check A]   fraction > 1 : {frac_gt1:.5f}   ({(theta_eff>1).sum():,} of {len(theta_eff):,})")
print(f"[check A]   99.9th pct   : {np.percentile(theta_eff, 99.9):.4f}")
if frac_gt1 < 0.01:
    print("[check A]   PASS - amplitude model is consistent with endo3 "
          "(Theta_eff essentially bounded by 1)")
else:
    print("[check A]   WARNING - too many Theta_eff > 1; A_net may be too small")

# ── (B) projection distribution, on the loud (truncation-free) subsample ───
import sys
sys.path.insert(0, "/home/manchun.yeung/population/mock-PE")
from GWMockCat.vt_utils import draw_thetas

rho_max = A_net / dL_Gpc                    # Theta-independent amplitude
theta_min = 6.0 / args.rho_max_cut          # below this, truncation still bites
sel_B = (rho_max > args.rho_max_cut) & (theta_eff > theta_min)
theta_loud = theta_eff[sel_B]

theta_fc_all = draw_thetas(500_000, rng)
theta_fc = theta_fc_all[theta_fc_all > theta_min]   # same restricted range

print(f"\n[check B] Theta-independent selection rho_max = A_net/dL > {args.rho_max_cut:g}")
print(f"[check B]   unbiased above Theta > {theta_min:.3f}; {sel_B.sum():,} injections kept")
print(f"[check B]   endo3 Theta_eff : mean={theta_loud.mean():.4f}  "
      f"median={np.median(theta_loud):.4f}")
print(f"[check B]   Finn & Chernoff : mean={theta_fc.mean():.4f}  "
      f"median={np.median(theta_fc):.4f}   (restricted to Theta > {theta_min:.3f})")
from scipy.stats import ks_2samp
ks = ks_2samp(theta_loud, theta_fc)
print(f"[check B]   KS statistic = {ks.statistic:.4f}  "
      f"(mean offset {abs(theta_loud.mean()-theta_fc.mean()):.4f})")
if abs(theta_loud.mean() - theta_fc.mean()) < 0.02:
    print("[check B]   PASS - H1+L1 network projection is consistent with "
          "single-detector Finn & Chernoff (detectors are near co-aligned)")
else:
    print("[check B]   WARNING - network Theta departs from Finn & Chernoff")

# ── figure ────────────────────────────────────────────────────────────────
C = sns.color_palette("colorblind")
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
fig.subplots_adjust(wspace=0.30)

ax = axes[0]
sub = rng.choice(len(theta_eff), size=min(40000, len(theta_eff)), replace=False)
ax.plot(snr_net[sub], theta_eff[sub], ".", ms=1.5, alpha=0.25, color=C[0])
ax.axhline(1.0, color="k", ls="--", lw=1.4)
ax.set_xscale("log")
ax.set_xlabel(r"endo3 recorded $\rho_{\rm net}$")
ax.set_ylabel(r"$\Theta_{\rm eff}=\rho_{\rm net}d_L/A_{\rm net}$")
ax.set_ylim(0, 1.35)
ax.set_title("(A) amplitude: must sit below 1", fontsize=12)
sns.despine(ax=ax)

ax = axes[1]
bins = np.linspace(theta_min, 1, 55)
ax.hist(theta_loud, bins=bins, density=True, histtype="step", lw=2.0,
        color=C[0], label=rf"endo3 ($\rho_{{\max}}>{args.rho_max_cut:g}$)")
ax.hist(theta_fc, bins=bins, density=True, histtype="step", lw=2.0,
        color=C[1], label="Finn \\& Chernoff")
ax.set_xlabel(r"$\Theta$")
ax.set_ylabel("density")
ax.legend(fontsize=10)
ax.set_title("(B) projection-factor distribution", fontsize=12)
sns.despine(ax=ax)

ax = axes[2]
pred = A_net / dL_Gpc * theta_eff  # identity by construction; show A_net vs mass instead
mchirp_det = (m1det * m2det) ** 0.6 / (m1det + m2det) ** 0.2
o = np.argsort(mchirp_det[sub])
ax.plot(mchirp_det[sub][o], (snr_net * dL_Gpc)[sub][o], ".", ms=1.5, alpha=0.25,
        color=C[0], label=r"endo3 $\rho_{\rm net}d_L$")
mm = np.exp(np.linspace(np.log(max(ms.min(), 2.1)), np.log(120), 200))
A_eq = osnr_interp(np.column_stack([mm, mm, np.zeros_like(mm)]))
mc_eq = (mm * mm) ** 0.6 / (2 * mm) ** 0.2
ax.plot(mc_eq, A_eq, "-", lw=2.4, color=C[3], label=r"$A_{\rm net}$ (equal mass, $\chi=0$)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\mathcal{M}_{\rm det}\,[M_\odot]$")
ax.set_ylabel(r"$\rho\,d_L\;[{\rm Gpc}]$")
ax.legend(fontsize=9)
ax.set_title("(C) amplitude vs chirp mass", fontsize=12)
sns.despine(ax=ax)

fig.suptitle("SNR grid (O3-actual H1+L1 PSDs) validated against endo3_v12 recorded SNRs",
             fontsize=12, y=1.02)
out = args.outdir / "validation_snr_grid.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot] saved -> {out}")
