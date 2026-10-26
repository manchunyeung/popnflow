#!/usr/bin/env python
"""
sbc_02_extract_pit.py
=====================
Compute the SBC PIT values q_jk from the finished inference runs.

    q_jk^(a) = sum_s w_s * 1[Lambda_s,a < Lambda_j,a] / sum_s w_s

WEIGHTS ARE MANDATORY (spec 5) and are applied DIRECTLY here, rather than via
dynesty's resample_equal. Both are correct estimators of the same quantity, but
resample_equal draws a finite equal-weight set and so injects resampling noise
into every PIT -- noise that then propagates into the ECDF the whole test is
built on. The direct sum has no such term.

  (For the record: the existing fixed-Lambda path, pp_bias_diagnostics.py:126,
  does NOT drop weights -- it uses resample_equal correctly. Spec 5 asks that a
  dropped-weight bug be reported if found; there is none. This is a variance
  refinement, not a bug fix.)

ESS is recorded per run, (sum w)^2 / sum w^2, and runs below --min-ess are
EXCLUDED with a logged reason. Nothing in the existing pipeline records this.

TIES: the indicator is strictly `<`. With continuous posteriors ties have
measure zero, but a run that collapsed to a point would otherwise yield q=0
silently; --min-ess catches that case first.

Output: pit/{method}.npz with
    pit        (n_blocks, n_catalogs, n_params)  NaN where excluded
    lambda_true(n_blocks, n_params)
    ess        (n_blocks, n_catalogs)
    ok         (n_blocks, n_catalogs)  bool
    param_names, method, min_ess
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbc_prior  # noqa: E402

PROD = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent   # suites live here: one big directory

# dynesty.NestedSampler.restore() unpickles the checkpoint's stored
# prior_transform/loglikelihood by (module, name) lookup in __main__ -- it
# never calls them, only needs the names resolvable. Real definitions live in
# ../sim_cat_inference.py / ../make_plots_met4.py with heavy module-level
# closure state (PE samples, injections, ...) this script never loads, so
# stubs suffice here.
def prior_transform(theta): raise RuntimeError("stub: not meant to be called")
def loglike_method_1(coord): raise RuntimeError("stub: not meant to be called")
def loglike_method_3(coord): raise RuntimeError("stub: not meant to be called")
def loglike_method_4(coord): raise RuntimeError("stub: not meant to be called")


def load_weighted_samples(h5_path):
    """Return (samples (N,D), normalised weights (N,)) from a dynesty checkpoint."""
    from dynesty import NestedSampler
    sampler = NestedSampler.restore(str(h5_path))
    res = sampler.results
    logwt = np.asarray(res["logwt"], float)
    logz = float(np.asarray(res["logz"], float)[-1])
    w = np.exp(logwt - logz)
    s = np.asarray(res["samples"], float)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("weights underflowed or non-finite")
    return s, w / tot


def pit_and_ess(samples, w, truth):
    """Weighted PIT per parameter, plus the run's effective sample size."""
    ess = float(1.0 / np.sum(w ** 2))          # w already normalised
    q = np.array([float(w[samples[:, a] < truth[a]].sum())
                  for a in range(samples.shape[1])])
    return q, ess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True)
    ap.add_argument("--method", default="met1", choices=["met1", "met3", "met4"])
    ap.add_argument("--pattern", default=None,
                    help="filename template inside <block>/inference/. Default "
                         "follows the fixed-Lambda convention: simcat_ for "
                         "met1/met4 and simcatnm_ for met3.")
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--n-catalogs", type=int, required=True)
    ap.add_argument("--subdir", default=None,
                    help="subdirectory of each block holding the runs. Default "
                         "'inference' for met1/met3; met4 lives in "
                         "'inference_met4_nlive<N>/' because its filename "
                         "carries no nlive and the two settings would collide. "
                         "Pass e.g. --subdir inference_met4_nlive200.")
    ap.add_argument("--min-ess", type=float, default=200.0,
                    help="spec 5: flag and exclude runs below this")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    suite = args.suite if args.suite.is_absolute() else HERE / args.suite
    pattern = args.pattern or (
        "simcatnm_cat{i:03d}_{method}.h5" if args.method == "met3"
        else "simcat_cat{i:03d}_{method}.h5")

    subdir = args.subdir or ("inference_met4_nlive200"
                             if args.method == "met4" else "inference")
    print(f"[cfg] reading <block>/{subdir}/{pattern}")
    blocks = sorted((suite / "blocks").glob("block_*"))
    if args.n_blocks:
        blocks = blocks[:args.n_blocks]
    if not blocks:
        raise SystemExit(f"no blocks under {suite/'blocks'}")

    n, m, D = len(blocks), args.n_catalogs, sbc_prior.NDIM
    pit = np.full((n, m, D), np.nan)
    ess = np.full((n, m), np.nan)
    ok = np.zeros((n, m), bool)
    lam = np.full((n, D), np.nan)

    n_missing = n_lowess = n_err = 0
    for j, bdir in enumerate(blocks):
        p = json.loads((bdir / "lambda_true.json").read_text())
        truth = np.array([p[k] for k in sbc_prior.PARAM_NAMES])
        lam[j] = truth
        for k in range(m):
            f = bdir / subdir / pattern.format(i=k, method=args.method)
            if not f.exists():
                n_missing += 1
                continue
            try:
                s, w = load_weighted_samples(f)
            except Exception as e:
                print(f"[err] {f.name} in {bdir.name}: {e}")
                n_err += 1
                continue
            # RUNTIME PRIOR GUARD. SBC is void unless the recovery prior is the
            # one Lambda_j was drawn from. The static check
            # (test_prior_agreement.py) compares source; this catches a
            # divergence introduced at RUN time -- e.g. a --mass-model smoothed
            # job (9 dims, delta_m appended) or edited bounds. Nested-sampling
            # samples come from the unit cube through prior_transform, so they
            # can never leave [LOWER, UPPER] unless the prior itself differed.
            if s.shape[1] == D:
                lo_v, hi_v = s.min(axis=0), s.max(axis=0)
                tol = 1e-6 * (sbc_prior.UPPER - sbc_prior.LOWER)
                bad = ((lo_v < sbc_prior.LOWER - tol)
                       | (hi_v > sbc_prior.UPPER + tol))
                if bad.any():
                    names = [sbc_prior.PARAM_NAMES[i]
                             for i in np.where(bad)[0]]
                    print(f"[err] {f.name} in {bdir.name}: samples outside the "
                          f"SBC prior box for {names} -- the recovery prior does "
                          f"NOT match sbc_prior. SBC is invalid for this run.")
                    n_err += 1
                    continue
            if s.shape[1] != D:
                print(f"[err] {f.name}: {s.shape[1]} dims, expected {D} "
                      f"(9 means --mass-model smoothed; SBC assumes sharp)")
                n_err += 1
                continue
            q, e = pit_and_ess(s, w, truth)
            ess[j, k] = e
            if e < args.min_ess:
                print(f"[excl] {bdir.name}/{f.name}: ESS {e:.0f} < {args.min_ess:.0f}")
                n_lowess += 1
                continue
            pit[j, k] = q
            ok[j, k] = True
        done = int(ok[j].sum())
        print(f"[{bdir.name}] {done}/{m} usable"
              + ("" if done == m else "   <-- incomplete"))

    tag = args.method if subdir == "inference" else f"{args.method}_{subdir.split('_')[-1]}"
    out = args.out or (suite / "pit" / f"{tag}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, pit=pit, lambda_true=lam, ess=ess, ok=ok,
             param_names=np.array(sbc_prior.PARAM_NAMES),
             method=args.method, min_ess=args.min_ess)

    tot = n * m
    print(f"\n[summary] {int(ok.sum())}/{tot} runs usable "
          f"({n_missing} missing, {n_lowess} low-ESS, {n_err} errored)")
    if np.isfinite(ess).any():
        print(f"[summary] ESS median {np.nanmedian(ess):.0f}, "
              f"min {np.nanmin(ess):.0f}")
    print(f"[out] {out}")

    if ok.sum() and ok.sum() == tot:
        # cheap sanity read-out; the real test is sbc_03's bands
        for a, nm in enumerate(sbc_prior.PARAM_NAMES):
            v = pit[:, :, a][ok]
            print(f"    {nm:>7}: mean PIT {v.mean():.3f}  (0.5 if calibrated)")


if __name__ == "__main__":
    main()
