#!/usr/bin/env python
"""
sbc_01_run_block.py
===================
Build one SBC block: pool -> catalogs -> per-event GMMs, at that block's
prior-drawn Lambda_j.

This is a DRIVER, not a reimplementation. It shells out to the existing,
already-validated stages so the SBC arm and the fixed-Lambda MDC run identical
generation code (spec 0: "reuse existing modules; do not fork-and-edit"):

    sim_cat_03_generate_pool.py  --true-pop-json blocks/block_j/lambda_true.json
    sim_cat_04_build_catalogs.py --indir blocks/block_j
    sim_cat_fit_event_gmms.py    --indir blocks/block_j

The ONLY change any of them needed was --true-pop-json on stage 3, which
defaults to the fixed CFG['true_pop'] and so leaves the MDC path bit-identical.

Stage 3 also generates the mock PE internally, so spec 4's steps 2 and 3 are
one call here.

WHAT IS *NOT* PER-BLOCK: injections.h5. It is Lambda-independent -- a fixed
reference distribution with a stored sampling_pdf, from which xi(Lambda) is
recovered by reweighting -- so one injection set is shared by every block and
by the fixed-Lambda MDC. Regenerating it per block would be wasted compute and
would silently break comparability.

Idempotent: --skip-existing lets a partially-failed suite be resumed without
redoing finished stages.

Example
    python sbc_01_run_block.py --suite sbc_run_pilot --block 0 \\
        --n-catalogs 8 --n-events 69 --n-jobs 32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROD = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent   # suites live here: one big directory
PY = sys.executable


def run(cmd, log_path, label):
    print(f"\n=== {label} ===\n$ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run([str(c) for c in cmd], stdout=fh,
                              stderr=subprocess.STDOUT, cwd=str(PROD))
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join(Path(log_path).read_text().splitlines()[-25:])
        raise SystemExit(f"[FAIL] {label} exited {proc.returncode} "
                         f"after {dt/60:.1f} min\n--- last 25 lines "
                         f"of {log_path} ---\n{tail}")
    print(f"[ok] {label} in {dt/60:.1f} min -> {log_path}")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True,
                    help="the --outdir given to sbc_00")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--n-catalogs", type=int, default=15,
                    help="m in the spec: catalogs subsampled from this "
                         "block's pool. Nearly free relative to the pool "
                         "itself, but see the design effect -- correlated "
                         "within a block, so m buys less than n.")
    ap.add_argument("--n-events", type=int, default=69)
    ap.add_argument("--pool-size", type=int, default=276)
    ap.add_argument("--n-draw", type=int, default=4_000_000)
    ap.add_argument("--width-ref-snr", type=float, default=8.0,
                    help="matches sim_cat6_sharp_w8; see the width-calibration "
                         "fix in RUN_NOTES sec.8")
    ap.add_argument("--mass-ratio-sampling", default="q")
    ap.add_argument("--mass-model", default="sharp")
    ap.add_argument("--n-jobs", type=int, default=32,
                    help="joblib workers for the GMM BIC scan")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    suite = args.suite if args.suite.is_absolute() else HERE / args.suite
    bdir = suite / "blocks" / f"block_{args.block:03d}"
    lam_file = bdir / "lambda_true.json"
    if not lam_file.exists():
        raise SystemExit(f"{lam_file} missing -- run sbc_00_draw_hyperparams.py first")

    seed_file = bdir / "seed.txt"
    seed = int(seed_file.read_text().strip()) % (2**31 - 1) if seed_file.exists() \
        else 20260815 + args.block
    logs = bdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    lam = json.loads(lam_file.read_text())
    print(f"[block {args.block}] Lambda_j = "
          + "  ".join(f"{k}={v:.3f}" for k, v in lam.items()))
    print(f"[block {args.block}] seed = {seed}   dir = {bdir}")

    timings = {}

    # ---- stage 3: pool + mock PE ----
    if args.skip_existing and (bdir / "pe_samples.h5").exists():
        print("[skip] pe_samples.h5 exists")
    else:
        timings["pool"] = run([
            PY, PROD / "sim_cat_03_generate_pool.py",
            "--true-pop-json", lam_file,
            "--outdir", bdir,
            "--seed", seed,
            "--pool-size", args.pool_size,
            "--n-draw", args.n_draw,
            "--mass-ratio-sampling", args.mass_ratio_sampling,
            "--mass-model", args.mass_model,
            "--width-ref-snr", args.width_ref_snr,
        ], logs / "stage03_pool.log", "stage 3: pool + mock PE")

    # ---- stage 4: catalogs ----
    if args.skip_existing and (bdir / "mdc_membership.npy").exists():
        print("[skip] mdc_membership.npy exists")
    else:
        timings["catalogs"] = run([
            PY, PROD / "sim_cat_04_build_catalogs.py",
            "--indir", bdir,
            "--n-catalogs", args.n_catalogs,
            "--n-events", args.n_events,
            "--seed", seed,
        ], logs / "stage04_catalogs.log", "stage 4: catalogs")

    # ---- per-event GMMs ----
    if args.skip_existing and (bdir / "event_gmms.h5").exists():
        print("[skip] event_gmms.h5 exists")
    else:
        timings["gmms"] = run([
            PY, PROD / "sim_cat_fit_event_gmms.py",
            "--indir", bdir,
            "--n-jobs", args.n_jobs,
        ], logs / "fit_gmms.log", "per-event GMMs (BIC K=1..20)")

    (bdir / "block_timings.json").write_text(json.dumps(
        {k: round(v, 1) for k, v in timings.items()}, indent=2))
    print(f"\n[block {args.block}] COMPLETE -> {bdir}")
    for k, v in timings.items():
        print(f"    {k:>9}: {v/60:6.1f} min")


if __name__ == "__main__":
    main()
