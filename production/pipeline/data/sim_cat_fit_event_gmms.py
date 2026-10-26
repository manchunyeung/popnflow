#!/usr/bin/env python3
"""
sim_cat_fit_event_gmms.py
==========================================================================
Fit the per-event GMM ONCE, at a BIC-selected K, for every event in a
sim_cat pool -- so sim_cat_inference.py can LOAD it instead of fitting it
inline on every inference run.

WHY. sim_cat_inference.py's Method 3/4 currently fit a fresh
GaussianMixture(K=args.kfixed, ...) per event on every run (K=7 for all
events when no --kfile is given -- see logs/smoke_met3.log for
sim_cat6_sharp_w8, which has none). That both (a) uses a fixed K instead of
one chosen by BIC, and (b) refits from scratch, with a catalog-dependent
random_state (EFFECTIVE_SEED = seed + 1000*catalog_id), on every single job --
so nominally identical pool events can get very slightly different GMMs
across catalogs purely from n_init RNG jitter, and the BIC scan itself
(K=1..20, n_init=5-10 each) is far too expensive to repeat inside every
inference job.

This script fits ONCE per POOL event (not per catalog) -- matching the
existing --kfile convention already in sim_cat_inference.py, whose docstring
notes "a pool event keeps the same optimal K no matter which catalog it
lands in" -- using select_k_per_event.py's exact conventions (full
covariance, reg_covar=1e-6, BIC argmin/plateau selection, DELTA_BIC=10) but
reading sim_cat's HDF5 pool format instead of Ray's bbh_result.json files.

Because every catalog built by sim_cat_04_build_catalogs.py copies a pool
event's (m1_det, m2_det, dL) arrays VERBATIM (see that script), a pool event
fit here is identical to what per-catalog inline fitting would produce for
the same K and random_state -- so this changes ONLY the source of K (BIC
instead of a fixed guess) and removes the catalog-to-catalog RNG jitter, not
the model.

OUTPUTS (companions, keep together):
    <out-dir>/per_event_best_K.csv   event(=pool index),n_samp,k_argmin,
                                      k_plateau,bic_at_argmin,bic_at_plateau
                                      -- SAME schema as select_k_per_event.py,
                                      directly usable as --kfile on its own.
    <out-dir>/event_gmms.h5          per-event weights_/means_/covariances_/
                                      precisions_/precisions_cholesky_ at
                                      BOTH k_argmin and k_plateau, keyed by
                                      pool index. Load with sim_cat_inference
                                      .py's --gmm-file.
    <out-dir>/bic_curve_<event>.csv, bic_curves.png  (best-effort, matches
                                      select_k_per_event.py's diagnostics)

USAGE
    python sim_cat_fit_event_gmms.py --indir sim_cat6_sharp_w8

Then:
    python sim_cat_inference.py --indir sim_cat6_sharp_w8 \\
        --gmm-file sim_cat6_sharp_w8/event_gmms.h5 --kcol k_argmin \\
        --catalog-id 0 --method method3 --live-points 200
==========================================================================
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import h5py
from sklearn.mixture import GaussianMixture

# Match select_k_per_event.py / Section 7.2 exactly.
FEATURE_NAMES = ("m1_det", "m2_det", "dL")   # detector frame
K_MIN, K_MAX = 1, 20
N_INIT = 5
REG_COVAR = 1e-6
RANDOM_STATE = 0
DELTA_BIC = 10.0


def select_k(ks, bics, delta_bic=DELTA_BIC):
    """(K_argmin, K_plateau) -- see select_k_per_event.py for the definition."""
    i_min = int(np.argmin(bics))
    k_argmin = int(ks[i_min])
    within = np.where(bics - bics[i_min] <= delta_bic)[0]
    k_plateau = int(ks[within[0]])
    return k_argmin, k_plateau


def bic_scan_and_cache(X, k_min, k_max, n_init, reg_covar, random_state):
    """Fit K=k_min..k_max ONCE, return (ks, bics, {K: fitted_model})."""
    n = X.shape[0]
    k_hi = min(k_max, n - 1)
    ks, bics, cache = [], [], {}
    for k in range(k_min, k_hi + 1):
        g = GaussianMixture(n_components=k, covariance_type="full",
                            reg_covar=reg_covar, n_init=n_init,
                            random_state=random_state).fit(X)
        ks.append(k)
        bics.append(float(g.bic(X)))
        cache[k] = g
    return np.asarray(ks), np.asarray(bics), cache


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", type=Path, required=True,
                    help="Pool directory holding pe_samples.h5 (e.g. sim_cat6_sharp_w8).")
    ap.add_argument("--pe-file", type=Path, default=None,
                    help="Override; default {indir}/pe_samples.h5")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: --indir itself, so --gmm-file/--kfile "
                         "point straight at the pool directory.")
    ap.add_argument("--kmin", type=int, default=K_MIN)
    ap.add_argument("--kmax", type=int, default=K_MAX)
    ap.add_argument("--n-init", type=int, default=N_INIT)
    ap.add_argument("--reg-covar", type=float, default=REG_COVAR)
    ap.add_argument("--random-state", type=int, default=RANDOM_STATE)
    ap.add_argument("--delta-bic", type=float, default=DELTA_BIC)
    ap.add_argument("--max-samp", type=int, default=None,
                    help="Subsample each event to at most this many PE rows "
                         "before the BIC scan (None = all 5000).")
    ap.add_argument("--example", default=None,
                    help="Pool index to use for the single-event BIC-curve plot.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers (joblib). One event's BIC scan "
                         "(K=1..20, n_init=5) took ~25s in testing -> ~2hr "
                         "serial for a 276-event pool; each event is fully "
                         "independent, so this parallelises linearly.")
    args = ap.parse_args()

    pe_file = args.pe_file or (args.indir / "pe_samples.h5")
    out_dir = args.out_dir or args.indir
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(pe_file, "r") as fpe:
        ev_names = sorted(k for k in fpe if k.startswith("event_"))
        pool_idx_of = {name: (int(fpe[name].attrs["pool_index"])
                              if "pool_index" in fpe[name].attrs else i)
                       for i, name in enumerate(ev_names)}
    print(f"[fit] {len(ev_names)} pool events from {pe_file}")

    def process(name):
        # Re-opens the file per call: each joblib worker (loky/subprocess
        # backend) needs its own handle -- an open h5py.File does not pickle.
        rng = np.random.default_rng(0)
        with h5py.File(pe_file, "r") as fpe:
            g = fpe[name]
            X = np.column_stack([np.asarray(g[k][:]) for k in FEATURE_NAMES])
        if args.max_samp is not None and X.shape[0] > args.max_samp:
            X = X[rng.choice(X.shape[0], args.max_samp, replace=False)]

        ks, bics, cache = bic_scan_and_cache(
            X, args.kmin, args.kmax, args.n_init, args.reg_covar, args.random_state)
        k_argmin, k_plateau = select_k(ks, bics, args.delta_bic)

        # Pull out only the two WINNING models' arrays -- cheap to ship back
        # over IPC, unlike the full K=1..20 cache or the sklearn objects.
        fits = {}
        for tag, k in (("argmin", k_argmin), ("plateau", k_plateau)):
            m = cache[k]
            fits[tag] = dict(weights=m.weights_, means=m.means_,
                             covariances=m.covariances_, precisions=m.precisions_,
                             precisions_cholesky=m.precisions_cholesky_)
        return dict(event=pool_idx_of[name], event_name=name, n_samp=int(X.shape[0]),
                   k_argmin=k_argmin, k_plateau=k_plateau,
                   bic_at_argmin=float(bics[ks == k_argmin][0]),
                   bic_at_plateau=float(bics[ks == k_plateau][0]),
                   ks=ks, bics=bics, fits=fits)

    if args.n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(process)(name) for name in ev_names)
    else:
        results = []
        for i, name in enumerate(ev_names):
            r = process(name)
            results.append(r)
            print(f"[{i+1:>3}/{len(ev_names)}] pool_{r['event']:03d}  "
                  f"N={r['n_samp']:>5}  K_argmin={r['k_argmin']:>2}  "
                  f"K_plateau={r['k_plateau']:>2}")

    rows = []
    gmm_out = out_dir / "event_gmms.h5"
    with h5py.File(gmm_out, "w") as fout:
        fout.attrs["source_pe_file"] = str(pe_file)
        fout.attrs["feature_names"] = list(FEATURE_NAMES)
        fout.attrs["reg_covar"] = args.reg_covar
        fout.attrs["random_state"] = args.random_state
        fout.attrs["n_init"] = args.n_init
        fout.attrs["delta_bic"] = args.delta_bic

        for r in results:
            grp = fout.create_group(f"pool_{r['event']:03d}")
            grp.attrs["pool_index"] = r["event"]
            grp.attrs["event_name"] = r["event_name"]
            grp.attrs["n_samp"] = r["n_samp"]
            grp.attrs["k_argmin"] = r["k_argmin"]
            grp.attrs["k_plateau"] = r["k_plateau"]
            for tag in ("argmin", "plateau"):
                sub = grp.create_group(tag)
                for name, arr in r["fits"][tag].items():
                    sub.create_dataset(name, data=arr)
            rows.append({k: v for k, v in r.items() if k != "fits"})

    print(f"[fit] saved GMMs -> {gmm_out}")

    csv_path = out_dir / "per_event_best_K.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "n_samp", "k_argmin", "k_plateau",
                    "bic_at_argmin", "bic_at_plateau"])
        for r in rows:
            w.writerow([r["event"], r["n_samp"], r["k_argmin"], r["k_plateau"],
                        f"{r['bic_at_argmin']:.3f}", f"{r['bic_at_plateau']:.3f}"])
    print(f"[fit] saved K table -> {csv_path}")

    ka = np.array([r["k_argmin"] for r in rows])
    kp = np.array([r["k_plateau"] for r in rows])
    print(f"[fit] K_argmin : min={ka.min()} median={int(np.median(ka))} max={ka.max()}")
    print(f"[fit] K_plateau: min={kp.min()} median={int(np.median(kp))} max={kp.max()}")

    # ---- best-effort diagnostics, matches select_k_per_event.py -----------
    try:
        ex = rows[0] if args.example is None else next(
            (r for r in rows if str(r["event"]) == str(args.example)), rows[0])
        curve = out_dir / f"bic_curve_pool{ex['event']:03d}.csv"
        with open(curve, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["K", "BIC", "dBIC_from_min"])
            bmin = ex["bics"].min()
            for k, b in zip(ex["ks"], ex["bics"]):
                w.writerow([int(k), f"{b:.3f}", f"{b - bmin:.3f}"])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ex["ks"], ex["bics"], "o-")
        ax.axvline(ex["k_plateau"], color="C2", ls="--", label=f"K_plateau={ex['k_plateau']}")
        ax.axvline(ex["k_argmin"], color="C3", ls=":", label=f"K_argmin={ex['k_argmin']}")
        ax.set_xlabel("K (GMM components)"); ax.set_ylabel("BIC")
        ax.set_title(f"pool_{ex['event']:03d}  (N={ex['n_samp']})")
        ax.legend(); fig.tight_layout()
        fig.savefig(out_dir / f"bic_curve_pool{ex['event']:03d}.png", dpi=130)
        plt.close(fig)

        sub = rows[:12]
        ncol = 4; nrow = int(np.ceil(len(sub) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for ax, r in zip(axes.ravel(), sub):
            ax.plot(r["ks"], r["bics"], "o-", ms=3)
            ax.axvline(r["k_plateau"], color="C2", ls="--")
            ax.set_title(f"pool_{r['event']:03d}\nK*={r['k_plateau']}", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(sub):]:
            ax.axis("off")
        fig.supxlabel("K"); fig.supylabel("BIC"); fig.tight_layout()
        fig.savefig(out_dir / "bic_curves.png", dpi=130)
        plt.close(fig)
        print(f"[fit] wrote {curve.name}, bic_curve_pool{ex['event']:03d}.png, bic_curves.png")
    except Exception as e:
        print(f"[fit] (diagnostics skipped: {e})")


if __name__ == "__main__":
    main()
