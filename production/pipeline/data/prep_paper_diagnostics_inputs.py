#!/usr/bin/env python
"""Build the inputs make_diagnostic_plots.py needs for paper Figs 2-4 (sim_cat6).

RECONSTRUCTED 2026-10-26. The original 2026-08-18 conversion was a one-off that
was never saved (paper_figs/SOURCES.md, Fig 3 note). This script reproduces its
outputs bit-for-bit -- checked against the files on disk with --check.

Writes, into --outdir (default sim_cat6_sharp_w8/paper_diagnostics):

  m1det.txt, m2det.txt, dL.txt
      catalog --catalog's events (event_000..event_068, in key order)
      concatenated into one flat column, np.savetxt default format.

  injections_endo3schema.h5
      GWMockCat's own injection campaign (injections.h5: rho_obs > 9 selected,
      DETECTED ROWS ONLY) re-keyed into the official-LVK-endo3 schema that
      make_diagnostic_plots.py hard-codes:
        injections/mass{1,2}         = m{1,2} * (1 + z)   (detector frame)
        injections/mass{1,2}_source  = m{1,2}
        injections/distance, redshift, sampling_pdf  copied
        injections/ifar_*            = 10.0 (dummy: every row is already found)
        attrs total_generated = injections.h5 n_total_drawn, analysis_time_s = 1.0
      analysis_time_s is dead downstream; it is a placeholder.

Usage (from production/):
  python pipeline/data/prep_paper_diagnostics_inputs.py            # write
  python pipeline/data/prep_paper_diagnostics_inputs.py --check    # compare only
"""
import argparse
from pathlib import Path

import h5py
import numpy as np

SOURCE_NOTE = (
    "Schema-adapted from sim_cat6_sharp_w8/injections.h5 (GWMockCat own "
    "rho_obs-threshold injection campaign, DETECTED ROWS ONLY) into the "
    "official-LVK-endo3-shaped schema make_diagnostic_plots.py expects. ifar_* "
    "columns are dummy placeholders (all >1) since every stored row is already "
    "a found injection by construction."
)
IFAR_KEYS = ("ifar_gstlal", "ifar_pycbc_bbh", "ifar_pycbc_hyperbank")


def flat_pe(catalog_file):
    with h5py.File(catalog_file, "r") as f:
        events = sorted(f.keys())
        return {name: np.concatenate([f[e][key][()] for e in events])
                for key, name in (("m1_det", "m1det"), ("m2_det", "m2det"), ("dL", "dL"))}


def adapted_injections(inj_file):
    with h5py.File(inj_file, "r") as f:
        m1, m2, z = f["m1"][()], f["m2"][()], f["z"][()]
        cols = {
            "mass1": m1 * (1 + z),
            "mass2": m2 * (1 + z),
            "mass1_source": m1,
            "mass2_source": m2,
            "distance": f["dL"][()],
            "redshift": z,
            "sampling_pdf": f["sampling_pdf"][()],
        }
        n_total = int(f.attrs["n_total_drawn"])
    for k in IFAR_KEYS:
        cols[k] = np.full(m1.shape, 10.0)
    return cols, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simdir", type=Path, default=Path("sim_cat6_sharp_w8"))
    ap.add_argument("--catalog", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=None,
                    help="default: <simdir>/paper_diagnostics")
    ap.add_argument("--check", action="store_true",
                    help="compare against existing files in --outdir, write nothing")
    args = ap.parse_args()
    outdir = args.outdir or args.simdir / "paper_diagnostics"

    pe = flat_pe(args.simdir / f"mdc_cat{args.catalog:03d}.h5")
    cols, n_total = adapted_injections(args.simdir / "injections.h5")
    adapter = outdir / "injections_endo3schema.h5"

    if args.check:
        ok = True
        for name, arr in pe.items():
            same = np.array_equal(np.loadtxt(outdir / f"{name}.txt"), arr)
            ok &= same
            print(f"{name}.txt: {'match' if same else 'DIFFERS'}")
        with h5py.File(adapter, "r") as f:
            for k, v in cols.items():
                same = np.array_equal(f["injections"][k][()], v)
                ok &= same
                print(f"injections/{k}: {'match' if same else 'DIFFERS'}")
            same = int(f.attrs["total_generated"]) == n_total
            ok &= same
            print(f"total_generated: {'match' if same else 'DIFFERS'}")
        raise SystemExit(0 if ok else 1)

    outdir.mkdir(parents=True, exist_ok=True)
    for name, arr in pe.items():
        np.savetxt(outdir / f"{name}.txt", arr)
    with h5py.File(adapter, "w") as f:
        g = f.create_group("injections")
        for k, v in cols.items():
            g.create_dataset(k, data=v)
        f.attrs["total_generated"] = np.int64(n_total)
        f.attrs["analysis_time_s"] = 1.0
        f.attrs["source_note"] = SOURCE_NOTE
    print(f"wrote {outdir}/{{m1det,m2det,dL}}.txt and {adapter}")


if __name__ == "__main__":
    main()
