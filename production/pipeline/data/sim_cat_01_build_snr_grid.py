#!/usr/bin/env python
"""
sim_cat_01_build_snr_grid.py
=============================
Stage 1 of the GWMockCat MDC rebuild (see gwmockcat_sample_generation.md).

Builds the `osnr_interp` grid GWMockCat needs: optimal SNR of a face-on,
overhead source at a reference distance of 1 Gpc, as a function of
(m1_det, m2_det, Xeff).

Detector network: **H1 + L1**, matching endo3_bbhpop-LIGO-T2100113-v12.hdf5,
whose `optimal_snr_net` is exactly sqrt(optimal_snr_h^2 + optimal_snr_l^2)
(verified: max deviation 0.0) -- i.e. the endo3_v12 GWTC-3 sensitivity
estimates use H1 and L1 only, no Virgo.  This differs from Ray et al.'s
original MDC script, which injected into H1+L1+V1; the discrepancy is
recorded rather than silently reconciled (see RUN_NOTES.md).

Sensitivity curves: `aligo_O3actual_H1.txt` / `aligo_O3actual_L1.txt`,
shipped with GWMockCat (sensitivity/), downloaded from LIGO-T2000012-v2 --
the O3-actual (GWTC-3 era) amplitude spectral densities.  Stage 2
(sim_cat_02_validate_snr_grid.py) validates the resulting grid directly
against endo3_v12's own recorded per-injection SNRs.

Output format matches GWMockCat's shipped grids exactly so it is a drop-in
for `vt_utils.interpolate_optimal_snr_grid` (which takes the
RegularGridInterpolator branch whenever an "Xeff" dataset is present):

    ms   (nm,)            log-spaced detector-frame masses, Msun
    Xeff (nx,)            effective spin axis
    SNR  (nm, nm, nx)     network optimal SNR at D_L = 1 Gpc
    attrs: D_L = '1.000000 Gpc'
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import h5py

SENS_DIR = "/home/manchun.yeung/population/mock-PE/sensitivity"
sys.path.insert(0, SENS_DIR)
import create_snr_grid as csg  # noqa: E402  (needs SENS_DIR on path)

# ── config ────────────────────────────────────────────────────────────────
CFG = dict(
    m_min_grid=2.0,        # matches GWMockCat's shipped grid convention
    m_max_grid=400.0,
    n_mass=100,
    n_xeff=11,             # population is non-spinning (Xeff=0); axis exists
                           # only because posterior_utils indexes (m1,m2,Xeff)
    xeff_min=-1.0,
    xeff_max=1.0,
    ref_dist_Mpc=1000.0,   # 1 Gpc
    psd_H1=os.path.join(SENS_DIR, "aligo_O3actual_H1.txt"),
    psd_L1=os.path.join(SENS_DIR, "aligo_O3actual_L1.txt"),
    outfile="sim_cat/snr_grid_O3actual_H1L1_Xeff.h5",
    n_proc=24,
)


def _net_snr(task):
    """Network optimal SNR sqrt(H^2 + L^2) for one (m1, m2, Xeff) point."""
    i, j, k, m1, m2, xeff = task
    kw = dict(dL=CFG["ref_dist_Mpc"], a1z=xeff, a2z=xeff)
    h = csg.optimal_snr_detframe(m1, m2, psd_file=CFG["psd_H1"], **kw)
    l = csg.optimal_snr_detframe(m1, m2, psd_file=CFG["psd_L1"], **kw)
    return i, j, k, float(np.hypot(h, l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=CFG["n_proc"])
    ap.add_argument("--outfile", default=CFG["outfile"])
    args = ap.parse_args()

    ms = np.exp(np.linspace(np.log(CFG["m_min_grid"]),
                            np.log(CFG["m_max_grid"]), CFG["n_mass"]))
    xeffs = np.linspace(CFG["xeff_min"], CFG["xeff_max"], CFG["n_xeff"])

    # only the lower triangle in (i,j); grid is symmetric under m1<->m2
    tasks = [(i, j, k, ms[i], ms[j], xeffs[k])
             for k in range(len(xeffs))
             for i in range(len(ms))
             for j in range(i + 1)]
    print(f"[grid] {len(ms)} masses x {len(xeffs)} Xeff -> {len(tasks):,} unique "
          f"(m1,m2,Xeff) points, 2 detectors each")
    print(f"[grid] H1 PSD: {CFG['psd_H1']}")
    print(f"[grid] L1 PSD: {CFG['psd_L1']}")

    snr = np.zeros((len(ms), len(ms), len(xeffs)))
    t0 = time.time()
    with Pool(args.n_proc) as pool:
        for n, (i, j, k, v) in enumerate(
                pool.imap_unordered(_net_snr, tasks, chunksize=64), 1):
            snr[i, j, k] = v
            snr[j, i, k] = v
            if n % 5000 == 0:
                el = time.time() - t0
                print(f"[grid]   {n:,}/{len(tasks):,}  ({el:.0f}s elapsed, "
                      f"eta {el*(len(tasks)-n)/n:.0f}s)", flush=True)
    print(f"[grid] done in {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with h5py.File(args.outfile, "w") as out:
        out.create_dataset("ms", data=ms, compression="gzip", shuffle=True)
        out.create_dataset("Xeff", data=xeffs, compression="gzip", shuffle=True)
        out.create_dataset("SNR", data=snr, compression="gzip", shuffle=True)
        out.attrs["D_L"] = "%.6f Gpc" % (CFG["ref_dist_Mpc"] / 1000.0)
        out.attrs["detectors"] = "H1,L1"
        out.attrs["psd_H1"] = CFG["psd_H1"]
        out.attrs["psd_L1"] = CFG["psd_L1"]
        out.attrs["network_convention"] = "sqrt(snr_H1^2 + snr_L1^2), face-on overhead"
        out.attrs["waveform"] = "IMRPhenomPv2 (via GWMockCat create_snr_grid)"
    print(f"[grid] saved -> {args.outfile}")
    print(f"[grid] SNR range at 1 Gpc: [{snr.min():.3f}, {snr.max():.3f}]")


if __name__ == "__main__":
    main()
