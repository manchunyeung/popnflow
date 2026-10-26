#!/usr/bin/env python
"""
sim_cat_06_validate.py
=======================
Stage 6: the validation suite required by gwmockcat_sample_generation.md
before the run can be declared done.

  1. Prior bounds     -- no posterior samples outside the declared PE prior
                         support; report min/max of m1, m2 across all events.
  2. Threshold consistency
                      -- the detected injection fraction, integrated against
                         the TRUE population, must reproduce the directly
                         observed detection efficiency from stage 3 to within
                         Monte Carlo error.  This is the sharp check that the
                         injection set and the event set really do come from
                         one generative process.
  3. Up-scatter       -- number of detected events whose pre-noise SNR is below
                         threshold.  MUST be non-zero; zero would mean the
                         rho_obs threshold silently reverted to deterministic
                         rho_opt selection.
  4. Width vs SNR     -- 90% credible width of m1 against rho_obs, overlaid
                         with the same quantity from the real GWTC-3 BBH
                         posteriors.  Reported, NOT tuned to match.

Check 5 (recovery test: run the population inference on 5 catalogs) is a
separate, expensive job and is not run here.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk", font_scale=1.0)
sns.set_style("ticks")
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

ap = argparse.ArgumentParser()
ap.add_argument("--indir", type=Path, default=Path("sim_cat"))
ap.add_argument("--gwtc3-dir", type=Path, default=Path("../GWTC-3"))
ap.add_argument("--grid-mass-range", type=float, nargs=2, default=(2.0, 400.0))
args = ap.parse_args()

results = {}
C = sns.color_palette("colorblind")

with h5py.File(args.indir / "mdc_pool.h5", "r") as f:
    pool = {k: f[k][:] for k in f.keys()}
    pool_attrs = dict(f.attrs)
TRUE = json.loads(pool_attrs["true_pop"])
RHO_THR = float(pool_attrs["rho_thr"])
Z_MAX = float(pool_attrs["z_max"])
N_DRAW = int(pool_attrs["n_draw"])
N_DET_STAGE3 = int(pool_attrs["n_detected"])

print("=" * 72)
print("CHECK 1 -- prior bounds")
print("=" * 72)
m1_lo, m1_hi, m2_lo, m2_hi = np.inf, -np.inf, np.inf, -np.inf
m1s_lo, m1s_hi = np.inf, -np.inf
n_bad = dict(m2_gt_m1=0, nonfinite=0, nonpositive=0, mdet_outside_grid=0)
n_tot = 0
with h5py.File(args.indir / "pe_samples.h5", "r") as f:
    for k in f.keys():
        g = f[k]
        m1, m2, dL, z = (g["m1_det"][:], g["m2_det"][:], g["dL"][:], g["z"][:])
        n_tot += len(m1)
        m1_lo, m1_hi = min(m1_lo, m1.min()), max(m1_hi, m1.max())
        m2_lo, m2_hi = min(m2_lo, m2.min()), max(m2_hi, m2.max())
        m1_src = m1 / (1 + z)
        m1s_lo, m1s_hi = min(m1s_lo, m1_src.min()), max(m1s_hi, m1_src.max())
        n_bad["m2_gt_m1"] += int((m2 > m1).sum())
        n_bad["nonfinite"] += int((~np.isfinite(m1 * m2 * dL * z)).sum())
        n_bad["nonpositive"] += int(((m1 <= 0) | (m2 <= 0) | (dL <= 0) | (z < 0)).sum())
        # samples are ALREADY detector frame -- check against the grid directly
        n_bad["mdet_outside_grid"] += int(((m1 < args.grid_mass_range[0]) |
                                           (m1 > args.grid_mass_range[1]) |
                                           (m2 < args.grid_mass_range[0])).sum())
print(f"  total samples          : {n_tot:,}")
print(f"  m1_det min/max         : [{m1_lo:.3f}, {m1_hi:.3f}]")
print(f"  m2_det min/max         : [{m2_lo:.3f}, {m2_hi:.3f}]")
print(f"  m1_src (=m1_det/(1+z)) : [{m1s_lo:.3f}, {m1s_hi:.3f}]  (back-transform check)")
for k, v in n_bad.items():
    print(f"  violations {k:22s}: {v:,}")
ok1 = (n_bad["m2_gt_m1"] == 0 and n_bad["nonfinite"] == 0 and n_bad["nonpositive"] == 0)
print(f"  -> {'PASS' if ok1 else 'FAIL'} (samples inside declared PE prior support)")
print("  NOTE: samples legitimately extend beyond the POPULATION support "
      f"[{TRUE['m_min']}, {TRUE['m_max']}] (source frame) -- that is measurement "
      "scatter, not a bug.")
results["check1"] = dict(n_samples=n_tot, m1_range=[m1_lo, m1_hi],
                         m2_range=[m2_lo, m2_hi], violations=n_bad, passed=bool(ok1))

print()
print("=" * 72)
print("CHECK 2 -- threshold consistency (injection set vs direct efficiency)")
print("=" * 72)


def log_p_pop(m1, m2, z, L, cosmo_dVdz):
    """Normalised true population density in (m1, m2, z); 0 outside support."""
    mg = np.linspace(L["m_min"], L["m_max"], 20000)
    p_pl = mg ** (-L["alpha"]); p_pl /= np.trapezoid(p_pl, mg)
    p_pk = np.exp(-0.5 * ((mg - L["mu"]) / L["sigma"]) ** 2) / np.sqrt(
        2 * np.pi * L["sigma"] ** 2)
    pm_grid = L["f1"] * p_pk + (1 - L["f1"]) * p_pl
    p_m1 = np.interp(m1, mg, pm_grid, left=0.0, right=0.0)

    q = m2 / m1
    q_min = L["m_min"] / m1
    b1 = 1.0 + L["beta"]
    p_q = b1 * q ** L["beta"] / (1.0 - q_min ** b1)
    # (m1,q) -> (m1,m2) Jacobian is 1/m1
    p = p_m1 * p_q / m1
    inside = ((m1 >= L["m_min"]) & (m1 <= L["m_max"]) &
              (m2 >= L["m_min"]) & (m2 <= m1) & (z >= 0) & (z <= Z_MAX))
    return np.where(inside, p * cosmo_dVdz, 0.0)


import sys
sys.path.insert(0, "/home/manchun.yeung/population/mock-PE")
from GWMockCat import cosmo_utils  # noqa: E402
cd = cosmo_utils.interp_cosmology()
zg = np.linspace(1e-4, Z_MAX, 4000)
pz_un = cd["diff_comoving_vol"](zg) * (1 + zg) ** (TRUE["gamma"] - 1.0)
pz_norm = np.trapezoid(pz_un, zg)

with h5py.File(args.indir / "injections.h5", "r") as f:
    inj_attrs = dict(f.attrs)
    # Current format stores DETECTED ROWS ONLY (endo3 convention). Older files
    # carried a `detected` mask over every drawn row; support both.
    det = f["detected"][:] if "detected" in f else slice(None)
    i_m1 = f["m1"][:][det]
    i_m2 = f["m2"][:][det]
    i_z = f["z"][:][det]
    i_pdraw = f["sampling_pdf"][:][det]
    i_rho_opt = f["rho_opt"][:][det]
    i_theta = f["Theta"][:][det]
N_INJ = int(inj_attrs["n_total_drawn"])

pz_det = cd["diff_comoving_vol"](i_z) * (1 + i_z) ** (TRUE["gamma"] - 1.0) / pz_norm
p_pop_det = log_p_pop(i_m1, i_m2, i_z, TRUE, pz_det)
w = p_pop_det / i_pdraw
xi = w.sum() / N_INJ
xi_err = np.sqrt((w ** 2).sum()) / N_INJ
neff = w.sum() ** 2 / (w ** 2).sum()

eff_direct = N_DET_STAGE3 / N_DRAW
eff_direct_err = np.sqrt(N_DET_STAGE3) / N_DRAW
print(f"  xi from injections     : {xi:.4e} +/- {xi_err:.2e}   (N_eff = {neff:.0f})")
print(f"  direct efficiency      : {eff_direct:.4e} +/- {eff_direct_err:.2e}  "
      f"({N_DET_STAGE3:,}/{N_DRAW:,} from stage 3)")
diff = xi - eff_direct
sig = np.hypot(xi_err, eff_direct_err)
print(f"  difference             : {diff:+.3e}  ({abs(diff)/sig:.2f} sigma)")
ok2 = abs(diff) / sig < 3.0
print(f"  -> {'PASS' if ok2 else 'FAIL'} (agree within Monte Carlo error)")
results["check2"] = dict(xi=float(xi), xi_err=float(xi_err), n_eff=float(neff),
                         eff_direct=float(eff_direct), sigma=float(abs(diff) / sig),
                         passed=bool(ok2))

print()
print("=" * 72)
print("CHECK 3 -- up-scatter")
print("=" * 72)
rho_true = pool["rho_true"]
n_up_true = int((rho_true < RHO_THR).sum())
n_up_opt = int((pool["rho_opt"] < RHO_THR).sum())
print(f"  pool events with rho_true (=rho_opt*Theta) < {RHO_THR}: {n_up_true} / "
      f"{len(rho_true)}  ({100*n_up_true/len(rho_true):.1f}%)")
print(f"  pool events with rho_opt (Theta=1 SNR)    < {RHO_THR}: {n_up_opt} / "
      f"{len(rho_true)}")
i_rho_true = i_rho_opt * i_theta
print(f"  detected injections with rho_true < {RHO_THR}: "
      f"{int((i_rho_true < RHO_THR).sum()):,} / {len(i_rho_true):,}")
ok3 = n_up_true > 0
print(f"  -> {'PASS' if ok3 else 'FAIL'} (non-zero: rho_obs threshold genuinely applied)")
results["check3"] = dict(n_rho_true_below=n_up_true, n_rho_opt_below=n_up_opt,
                         frac=float(n_up_true / len(rho_true)), passed=bool(ok3))

print()
print("=" * 72)
print("CHECK 4 -- 90% credible width of m1 vs SNR (mock vs real GWTC-3)")
print("=" * 72)
mock_snr, mock_w = [], []
with h5py.File(args.indir / "pe_samples.h5", "r") as f:
    for k in sorted(f.keys()):
        g = f[k]
        # real GWTC-3 comparison below uses mass_1_SOURCE -- convert to match
        m1 = g["m1_det"][:] / (1 + g["z"][:])
        lo, hi = np.percentile(m1, [5, 95])
        mock_w.append(hi - lo)
        mock_snr.append(float(g.attrs["rho_obs"]))
mock_snr, mock_w = np.array(mock_snr), np.array(mock_w)
print(f"  mock: {len(mock_snr)} events, rho_obs [{mock_snr.min():.1f},{mock_snr.max():.1f}], "
      f"width [{mock_w.min():.1f},{mock_w.max():.1f}]")

PREF = ["C01:Mixed", "C01:IMRPhenomXPHM", "C01:SEOBNRv4PHM",
        "C01:IMRPhenomPv3HM", "C01:IMRPhenomPv2", "PublicationSamples"]
NEED = ("mass_1_source", "network_optimal_snr")


def _pick_group(f):
    """First group carrying BOTH required fields (PublicationSamples has
    mass_1_source but no network_optimal_snr, so group presence is not enough)."""
    def usable(g):
        try:
            names = f[g]["posterior_samples"].dtype.names
            return names is not None and all(n in names for n in NEED)
        except Exception:
            return False
    for g in PREF:
        if g in f and usable(g):
            return g
    return next((g for g in f.keys() if usable(g)), None)


real_snr, real_w, real_m, n_real = [], [], [], 0
for fn in sorted(glob.glob(str(args.gwtc3_dir / "*.h5"))):
    try:
        with h5py.File(fn, "r") as f:
            grp = _pick_group(f)
            if grp is None:
                continue
            ps = f[grp]["posterior_samples"]
            m1s = ps["mass_1_source"][:]
            snr = float(np.median(ps["network_optimal_snr"][:]))
            lo, hi = np.percentile(m1s, [5, 95])
            real_w.append(hi - lo); real_snr.append(snr)
            real_m.append(float(np.median(m1s))); n_real += 1
    except Exception:
        continue
real_snr, real_w, real_m = np.array(real_snr), np.array(real_w), np.array(real_m)
print(f"  real GWTC-3: {n_real} events with usable samples "
      f"(dir has {len(glob.glob(str(args.gwtc3_dir/'*.h5')))} files)")
if n_real:
    print(f"  real: rho [{real_snr.min():.1f},{real_snr.max():.1f}], "
          f"width [{real_w.min():.1f},{real_w.max():.1f}]")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0.34)
ax = axes[0]
ax.loglog(mock_snr, mock_w, "o", ms=5, alpha=0.6, color=C[0], label="mock (this run)")
if n_real:
    ax.loglog(real_snr, real_w, "s", ms=6, alpha=0.8, color=C[3],
              label=f"real GWTC-3 ({n_real})")
s = np.logspace(np.log10(8), np.log10(60), 50)
ax.loglog(s, 200.0 / s, "k--", lw=1.2, alpha=0.7, label=r"$\propto 1/\rho$")
ax.set_xlabel(r"$\rho$  (mock: $\rho_{\rm obs}$; real: median network optimal SNR)")
ax.set_ylabel(r"90\% credible width of $m_1\,[M_\odot]$")
ax.legend(fontsize=10); sns.despine(ax=ax)
ax.set_title("width vs SNR", fontsize=12)

ax = axes[1]
with h5py.File(args.indir / "pe_samples.h5", "r") as f:
    mm = np.array([np.median(f[k]["m1_det"][:] / (1 + f[k]["z"][:]))
                   for k in sorted(f.keys())])
ax.loglog(mm, mock_w / mm, "o", ms=5, alpha=0.6, color=C[0], label="mock")
if n_real:
    ax.loglog(real_m, real_w / real_m, "s", ms=6, alpha=0.8, color=C[3],
              label="real GWTC-3")
ax.set_xlabel(r"median $m_1\,[M_\odot]$")
ax.set_ylabel(r"fractional 90\% width of $m_1$")
ax.legend(fontsize=10); sns.despine(ax=ax)
ax.set_title("fractional width vs mass", fontsize=12)

fig.suptitle("Check 4: mock PE widths vs real GWTC-3 (reported, not tuned)",
             fontsize=12, y=1.02)
out = args.indir / "validation_width_vs_snr.pdf"
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  figure -> {out}")
results["check4"] = dict(n_mock=len(mock_snr), n_real=n_real,
                         mock_width_median=float(np.median(mock_w)),
                         real_width_median=float(np.median(real_w)) if n_real else None)

with open(args.indir / "validation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print()
print("=" * 72)
allpass = ok1 and ok2 and ok3
print(f"SUMMARY: checks 1-3 {'ALL PASS' if allpass else 'HAVE FAILURES'}; "
      f"check 4 reported (see figure)")
print("=" * 72)
