#!/usr/bin/env python
"""
sim_cat_04_build_catalogs.py
=============================
Stage 4: assemble the catalogs from the event pool.

Draws `n_events` per catalog WITHOUT replacement from the shared pool,
independently per catalog, and writes one HDF5 per catalog in the format
specified by gwmockcat_sample_generation.md.

DEVIATION FROM THE MD SPEC: the md's per-event schema names the mass columns
`m1`/`m2` as SOURCE-frame. This project's GMM step fits in DETECTOR frame
(m1_det, m2_det, dL) and transforms back to source frame afterward, so
stage 3 stores `m1_det`/`m2_det` instead (with `z` alongside for the
back-transform), and this stage just carries that naming through unchanged.
See sim_cat/RUN_NOTES.md.

The membership array (n_catalogs x n_events indices into the pool) is saved to
its own file -- it is required downstream for correlated-null PP-plot band
construction and must not be regenerated later.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("--indir", type=Path, default=Path("sim_cat"))
ap.add_argument("--n-catalogs", type=int, default=50)
ap.add_argument("--n-events", type=int, default=69)
ap.add_argument("--seed", type=int, default=20260803)
args = ap.parse_args()

rng = np.random.default_rng(args.seed + 777)

with h5py.File(args.indir / "mdc_pool.h5", "r") as f:
    pool_attrs = dict(f.attrs)
    pool_size = int(f.attrs["n_events"])

pe_path = args.indir / "pe_samples.h5"
with h5py.File(pe_path, "r") as f:
    n_samples = int(f.attrs["n_samples"])
    prior_desc = f.attrs["prior"]

if args.n_events > pool_size:
    raise SystemExit(f"n_events {args.n_events} > pool size {pool_size}")

membership = np.array([rng.choice(pool_size, size=args.n_events, replace=False)
                       for _ in range(args.n_catalogs)])
np.save(args.indir / "mdc_membership.npy", membership)
print(f"[cat] membership {membership.shape} -> {args.indir/'mdc_membership.npy'}")
print(f"[cat] pool coverage: {len(np.unique(membership))}/{pool_size} pool events used; "
      f"mean occurrences per pool event = "
      f"{args.n_catalogs*args.n_events/pool_size:.2f}")

FIELDS = ("m1_det", "m2_det", "dL", "z", "prior")
ATTRS = ("pool_index", "m1_true", "m2_true", "z_true", "dL_true",
         "rho_opt", "rho_true", "rho_obs", "Theta_true", "reweight_ESS")

with h5py.File(pe_path, "r") as fpe:
    for c in range(args.n_catalogs):
        out = args.indir / f"mdc_cat{c:03d}.h5"
        with h5py.File(out, "w") as fo:
            for k, pidx in enumerate(membership[c]):
                src = fpe[f"event_{pidx:03d}"]
                g = fo.create_group(f"event_{k:03d}")
                for name in FIELDS:
                    g.create_dataset(name, data=src[name][:], compression="gzip",
                                     shuffle=True)
                for a in ATTRS:
                    g.attrs[a] = src.attrs[a]
            fo.attrs["seed"] = args.seed
            fo.attrs["catalog_id"] = c
            fo.attrs["n_events"] = args.n_events
            fo.attrs["n_samples"] = n_samples
            fo.attrs["threshold_type"] = "rho_obs"
            fo.attrs["rho_thr"] = pool_attrs["rho_thr"]
            fo.attrs["psd_name"] = pool_attrs["psd_name"]
            fo.attrs["gwmockcat_version"] = "0.0.1"
            fo.attrs["git_commit"] = pool_attrs["gwmockcat_commit"]
            fo.attrs["true_pop"] = pool_attrs["true_pop"]
            fo.attrs["prior"] = prior_desc
        if (c + 1) % 10 == 0:
            print(f"[cat]   wrote {c+1}/{args.n_catalogs} catalogs", flush=True)

print(f"[cat] done: {args.n_catalogs} catalogs x {args.n_events} events -> {args.indir}")
