#!/usr/bin/env python
"""
rerun_ks_ad.py
==============
Re-run `test_gmm_fit.py`'s KS / Anderson-Darling goodness-of-fit test against
the CURRENT inputs: the GWMockCat mock-PE samples and the per-event GMMs fitted
from them.

WHY A DRIVER AND NOT THE SCRIPT'S OWN CLI. `test_gmm_fit.py` reads flat text
files (`input_data/mdc_*_69rand.txt`) -- the pre-GWMockCat PE -- and only its
`--json-glob` branch wires `--k-csv` into the per-event fit. Our PE now lives in
`sim_cat6_sharp_w8/pe_samples.h5` and the production GMMs use a BIC-chosen K per
event (3-10, median 5), so running the CLI as-is would test the old data with a
fixed K=7. This driver reuses the script's statistics unchanged -- it imports
`run_events`, `print_summary` and `make_summary_histograms` -- and only replaces
the loading.

TWO MODES, because "redo the test" means two different things:

  --mode stored   the exact GMMs production reads from event_gmms.h5, scored
                  against ALL of that event's PE samples. This is the object in
                  use; nothing is refitted.
  --mode refit    refit per event with the production recipe (K = k_argmin,
                  reg_covar=1e-6, n_init=5, raw coords) on 80% of the samples
                  and score the held-out 20% -- the generalization protocol the
                  original test used, so the excess factors are comparable to
                  the old numbers in gmm_fit_qa_notes.md.

The headline statistic is the EXCESS FACTOR D_GMM / D_baseline, where the
baseline is a PE-vs-PE split-half of the same event: with 5000 samples a raw KS
p-value is ~0 for any fit, so the meaningful question is how the discrepancy
compares to what perfect agreement already produces at this N.

Usage:
    python rerun_ks_ad.py --mode stored
    python rerun_ks_ad.py --mode refit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
PROD = HERE.parent
sys.path.insert(0, str(PROD))

import test_gmm_fit as T  # noqa: E402  (defs only at import; argv is parsed in main)

FEATURES = ["m1_det", "m2_det", "dL"]     # column order of the IS cache and the GMMs


def load_pe(pe_file: Path, n_events=None):
    """(list of (N,3) arrays, labels) from the GWMockCat PE file."""
    out, labels = [], []
    with h5py.File(pe_file, "r") as h:
        names = sorted(h.keys())
        if n_events:
            names = names[:n_events]
        for nm in names:
            g = h[nm]
            X = np.column_stack([np.asarray(g[f], float) for f in FEATURES])
            out.append(np.ascontiguousarray(X[np.isfinite(X).all(axis=1)]))
            labels.append(nm)
    return out, labels


def load_gmms(gmm_file: Path, which="argmin", n_events=None):
    """Rebuild sklearn GaussianMixture objects from the stored fits.

    Only the fitted attributes are set -- `.sample()` needs weights_, means_ and
    covariances_ plus a precisions_cholesky_, all of which are in the file, so
    nothing is re-estimated here.
    """
    from sklearn.mixture import GaussianMixture
    models, ks = [], []
    with h5py.File(gmm_file, "r") as h:
        names = sorted(h.keys())
        if n_events:
            names = names[:n_events]
        for nm in names:
            g = h[nm][which]
            w = np.asarray(g["weights"], float)
            m = GaussianMixture(n_components=len(w), covariance_type="full")
            m.weights_ = w
            m.means_ = np.asarray(g["means"], float)
            m.covariances_ = np.asarray(g["covariances"], float)
            m.precisions_cholesky_ = np.asarray(g["precisions_cholesky"], float)
            m.precisions_ = np.asarray(g["precisions"], float)
            m.converged_ = True
            m.n_features_in_ = m.means_.shape[1]
            models.append(m)
            ks.append(len(w))
    return models, np.array(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stored", "refit"], default="stored")
    ap.add_argument("--pe-file", type=Path,
                    default=PROD / "sim_cat6_sharp_w8/pe_samples.h5")
    ap.add_argument("--gmm-file", type=Path,
                    default=PROD / "sim_cat6_sharp_w8/event_gmms.h5")
    ap.add_argument("--which-k", default="argmin", choices=["argmin", "plateau"],
                    help="which stored fit -- production passes --kcol k_argmin")
    ap.add_argument("--n-events", type=int, default=None,
                    help="limit to the first N events (default: all 276)")
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--baseline-repeats", type=int, default=8)
    ap.add_argument("--excess-threshold", type=float, default=3.0)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tex-mode", choices=["off", "mathtext", "usetex"],
                    default="off",
                    help="Typesetting. 'mathtext' = Computer Modern via matplotlib's "
                         "own mathtext -- LaTeX appearance, and the ONLY safe choice "
                         "here. 'usetex' = real LaTeX, which on this host SILENTLY "
                         "DROPS THE MINUS SIGN from tick labels (verified: a -2..1 "
                         "axis renders as '2 1 0 1' with text.usetex=True under both "
                         "settings of axes.unicode_minus, while mathtext renders "
                         "'-2 -1 0 1' correctly). The AD A^2 axis is negative for most "
                         "events, so usetex corrupts this figure. 'off' = defaults.")
    ap.add_argument("--fontsize", type=float, default=13.0,
                    help="Axis-label/title size; legend is 0.8x this.")
    args = ap.parse_args()

    tag = args.out or f"ks_ad_{args.mode}"
    events, labels = load_pe(args.pe_file, args.n_events)
    print(f"[pe]  {len(events)} events from {args.pe_file.name}, "
          f"{events[0].shape[0]} samples in event 0, cols={FEATURES}")

    gmms, ks = load_gmms(args.gmm_file, args.which_k, args.n_events)
    if len(gmms) != len(events):
        raise SystemExit(f"{len(gmms)} GMMs vs {len(events)} PE events")
    print(f"[gmm] K from '{args.which_k}': min={ks.min()} median={int(np.median(ks))} "
          f"max={ks.max()}")

    if args.mode == "stored":
        # the production objects themselves; no refit, so the whole PE set is
        # the test set and train_frac is irrelevant.
        kw = dict(gmms=gmms, train_frac=1.0)
        print("[mode] stored GMMs, scored against all PE samples")
    else:
        kw = dict(fit_fns=[T.make_k_fit_fn(int(k)) for k in ks],
                  train_frac=args.train_frac)
        print(f"[mode] refit per event at production K, "
              f"train_frac={args.train_frac} (held-out test)")

    results = T.run_events(
        events, labels=labels, n_slices=args.n_slices,
        baseline_repeats=args.baseline_repeats,
        excess_threshold=args.excess_threshold, seed=args.seed, **kw)

    T.print_summary(results, cols=(0,))

    rows = []
    for r in results:
        row = dict(event=r.name, n_pe=r.n_pe, n_train=r.n_train, n_test=r.n_test,
                   k=r.k_components, flagged=bool(r.flagged))
        for c, f in enumerate(FEATURES):
            d = r.per_dim.get(c, {})
            row[f"ks_D_{f}"] = d.get("ks_D")
            row[f"ad_A2_{f}"] = d.get("ad_A2")
            row[f"ad_p_{f}"] = d.get("ad_p")
            row[f"excess_{f}"] = r.excess.get(f"ks_{c}")
        row["excess_sliced"] = r.excess.get("ks_sliced")
        row["ad_A2_sliced"] = r.sliced.get("ad_A2_median")
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    csv = HERE / f"{tag}.csv"
    df.to_csv(csv, index=False)
    print(f"\n[out] {csv}")

    exc = df["excess_m1_det"].to_numpy(float)
    fin = exc[np.isfinite(exc)]
    print(f"[summary] KS excess (m1_det): median {np.median(fin):.2f}  "
          f"90th pct {np.percentile(fin, 90):.2f}  max {fin.max():.2f}")
    print(f"[summary] events over threshold {args.excess_threshold:g}: "
          f"{int((fin > args.excess_threshold).sum())}/{len(fin)}")

    if args.tex_mode == "mathtext":
        import matplotlib
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["Computer Modern Roman",
                                             "DejaVu Serif"]
        matplotlib.rcParams["mathtext.fontset"] = "cm"
        matplotlib.rcParams["axes.formatter.use_mathtext"] = True
    elif args.tex_mode == "usetex":
        print("[warn] --tex-mode usetex drops minus signs on this host; the AD A^2 "
              "axis will be wrong. Use 'mathtext' unless you have checked the output.")
    T.make_summary_histograms(results, cols=(0,), out_path=str(HERE / f"{tag}.pdf"),
                              use_tex=(args.tex_mode == "usetex"),
                              math_labels=(args.tex_mode != "off"),
                              fontsize=args.fontsize, include_sliced=True)
    print(f"[out] {HERE/tag}.pdf")


if __name__ == "__main__":
    main()
