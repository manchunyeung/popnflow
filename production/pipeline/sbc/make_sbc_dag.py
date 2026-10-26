#!/usr/bin/env python
"""
make_sbc_dag.py
===============
Emit the HTCondor DAG for an SBC suite (spec 10: "blocks are embarrassingly
parallel; emit a DAG rather than a serial driver").

DEPENDENCY STRUCTURE

    GEN_j                                  one per block: pool -> catalogs -> GMMs
      |-> INF_j_k_met1                     n*m
      |     \\-> INF_j_k_met4              met4 needs met1's posterior median
      |                                    as its mixture reference Lambda_0
      \\-> INF_j_k_met3                    n*m, independent of met1

The met1 -> met4 edge is REAL, not cosmetic: sim_cat_inference.py's Method 4
reads `{run}_cat{k}_met1_lambda0.npy`, so a met4 job that starts before its
met1 sibling finishes will fail. Any hand-rolled parallel launcher has to
encode this too; the DAG does it for free.

Every job is idempotent (`--skip-existing` / `--skip-if-exists`), so a
`condor_submit_dag` on a partially-complete suite resumes rather than redoes,
and a rescue DAG works normally.

Usage
    python make_sbc_dag.py --suite sbc_run_pilot --n-blocks 8 --n-catalogs 8 \\
        --methods met1 --nlive 200 > sbc_run_pilot/sbc.dag
    condor_submit_dag sbc_run_pilot/sbc.dag        # NOT run automatically
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--first-block", type=int, default=0,
                    help="Emit jobs for blocks [first-block, n-blocks) only. "
                         "For EXTENDING a running suite: sbc_00 derives "
                         "Lambda_j from block j's own child seed, so "
                         "--n-blocks 20 reproduces blocks 0-9 byte-identically "
                         "and only adds 10-19. Generate the new lambda_true "
                         "files with sbc_00 --n-blocks 20, then emit a DAG with "
                         "--first-block 10 --n-blocks 20 and submit that "
                         "alongside the one already running. Do NOT re-emit "
                         "0-9: their jobs are idempotent but resubmitting them "
                         "would contend for slots with the run in flight.")
    ap.add_argument("--n-catalogs", type=int, required=True)
    ap.add_argument("--n-events", type=int, default=69)
    ap.add_argument("--methods", nargs="+", default=["met1"],
                    choices=["met1", "met3", "met4"])
    ap.add_argument("--nlive", type=int, default=1000,
                    help="MATCHES THE FIXED-LAMBDA MDC. run_sim_cat.sub sets no "
                         "`environment` line, so run_sim_cat_catalog.sh's "
                         "NLIVE=${NLIVE:-1000} applies -- the fixed-Lambda "
                         "met1/met3 sweep ran at nlive=1000 (raised from 200 on "
                         "2026-08-05). Keep these equal: rho_lambda's whole "
                         "purpose is to pass verdict on the fixed-Lambda PP "
                         "plot, and that comparison is confounded if the two "
                         "arms use different sampler settings. met4 is the "
                         "exception -- the fixed-Lambda arm ran it at BOTH 200 "
                         "and 1000 (run_sim_cat_met4_nlive{200,1000}.sub), so "
                         "match whichever one you intend to compare against.")
    ap.add_argument("--combined", action="store_true",
                    help="Emit ONE `--method all` job per (block, catalog) "
                         "instead of one job per method. Strongly preferred "
                         "for multi-method suites: sim_cat_inference.py has NO "
                         "early exit, so the per-event GMM section (:885) and "
                         "the guided-IS cache (:1075-1130, 200k proposal "
                         "samples + per-event GMM evaluation) run "
                         "UNCONDITIONALLY even for --method method1. Three "
                         "separate jobs therefore pay that shared setup three "
                         "times. It also removes the met4->met1 edge, since "
                         "Method 4 picks up Lambda_0 in memory in 'all' mode.")
    ap.add_argument("--nlive-met4", type=int, nargs="+", default=[200],
                    help="nlive for met4 SPECIFICALLY -- accepts several, e.g. "
                         "`--nlive-met4 200 1000`, which emits one met4 job per "
                         "value. Default 200, because that is what the reported "
                         "fixed-Lambda PP plot used: "
                         "pit_bias_cat6_met124_nlive200.csv takes met1/met3 from "
                         "inference/ (nlive=1000) and met4 from "
                         "inference_met4_nlive200/. The existing figure is "
                         "MIXED-nlive, so matching it means met1/met3 at 1000 "
                         "and met4 at 200. Each value writes to its own "
                         "inference_met4_nlive<N>/ directory, since the output "
                         "filename carries no nlive and they would otherwise "
                         "overwrite each other.")
    ap.add_argument("--mem-met1", default="4000M")
    ap.add_argument("--mem-met3", default="6000M")
    ap.add_argument("--mem-met4", default="7000M",
                    help="Per-method request_memory. Defaults carry ~40%% "
                         "headroom over the measured peak across ~500 real jobs "
                         "(met1 max 2.9 GB, met3 4.0 GB, met4 5.1 GB) rather "
                         "than the uniform 15 GB production/*.sub uses. On a "
                         "512 GB / 128-core node the uniform value is the "
                         "binding constraint and idles ~3/4 of the cores.")
    ap.add_argument("--max-idle", type=int, default=200,
                    help="DAGMAN_MAX_JOBS_IDLE. Batching by hand is unnecessary "
                         "-- this is the throttle. On a single 128-core node, "
                         "~150 is plenty.")
    ap.add_argument("--max-submitted", type=int, default=500,
                    help="DAGMAN_MAX_JOBS_SUBMITTED")
    ap.add_argument("--gen-jobs", type=int, default=8,
                    help="joblib workers for the GMM BIC scan AND request_cpus "
                         "for the gen job -- one knob, they cannot drift. "
                         "Default lowered 32 -> 8 after measuring the live "
                         "pool: a 32-core request matches only 90 of 20,234 "
                         "slots (0.4%%) and sits idle for hours, while 8 cores "
                         "matches 1,076. Only the GMM stage uses them; pool "
                         "generation is single-threaded, so a large request "
                         "mostly holds cores idle. 8 workers takes the BIC "
                         "scan from ~12 min to ~48 min -- negligible against "
                         "the inference that follows.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.combined and set(args.methods) != {"met1", "met3", "met4"}:
        # sim_cat_inference.py's --method all runs met1 AND met3 AND met4
        # unconditionally; there is no way to ask "all" for a subset. Refusing
        # here is better than silently producing methods the user did not ask
        # for (and paying for them).
        sys.exit(f"--combined runs sim_cat_inference.py --method all, which "
                 f"always does met1+met3+met4. You asked for "
                 f"{sorted(args.methods)}. Either pass "
                 f"--methods met1 met3 met4 --combined, or drop --combined "
                 f"and accept one job per method.")
    if not args.combined and "met4" in args.methods and "met1" not in args.methods:
        sys.exit("met4 requires met1 in --methods: it reads met1's "
                 "*_lambda0.npy as its mixture reference. (Or pass --combined, "
                 "where 'all' runs met1 in the same process and hands Lambda_0 "
                 "over in memory.)")

    # ABSOLUTE PATHS THROUGHOUT. condor_submit_dag resolves `CONFIG` against the
    # CURRENT WORKING DIRECTORY, not the .dag file's directory, so a relative
    # `CONFIG sbc.dagconfig` fails with "Can't read DAGMan config file" whenever
    # the DAG is submitted from anywhere but its own directory. JOB submit-file
    # paths have the same ambiguity (they follow the cwd unless -usedagdir is
    # given). Emitting absolute paths removes the question entirely.
    suite_dir_abs = (HERE / args.suite) if not Path(args.suite).is_absolute() \
        else Path(args.suite)
    cfg_abs = suite_dir_abs / "sbc.dagconfig"
    # NOTE: an extension DAG reuses the SAME sbc.dagconfig, which is fine --
    # the throttles are per-DAGMan-instance, so two concurrent DAGs each get
    # their own MAX_JOBS_IDLE budget. Halve --max-idle if that is too much.
    sub_gen = HERE / "submit_sbc_gen.sub"
    sub_inf = HERE / "submit_sbc_infer.sub"

    L = []
    A = L.append
    A(f"# SBC DAG -- suite={args.suite}  n={args.n_blocks} blocks  "
      f"m={args.n_catalogs} catalogs  methods={','.join(args.methods)}  "
      f"nlive={args.nlive}")
    _perj = (1 if args.combined else
             sum(len(args.nlive_met4) if m == "met4" else 1 for m in args.methods))
    A(f"# total: {args.n_blocks} gen + "
      f"{args.n_blocks*args.n_catalogs*_perj} inference jobs"
      + ("  (--combined: one --method all job per catalog)" if args.combined else ""))
    A("")
    A(f"CONFIG {cfg_abs}")
    A("")

    for j in range(args.first_block, args.n_blocks):
        g = f"GEN_{j:03d}"
        A(f"JOB {g} {sub_gen}")
        A(f'VARS {g} suite="{args.suite}" block="{j}" '
          f'ncat="{args.n_catalogs}" nev="{args.n_events}" '
          f'njobs="{args.gen_jobs}" tag="{g}"')
        A(f"RETRY {g} 2")

        for k in range(args.n_catalogs):
            if args.combined:
                n = f"INF_{j:03d}_{k:03d}_all"
                A(f"JOB {n} {sub_inf}")
                A(f'VARS {n} suite="{args.suite}" block="{j}" cat="{k}" '
                  f'method="all" nlive="{args.nlive}" tag="{n}" '
                  f'mem="{args.mem_met4}"')
                A(f"RETRY {n} 2")
                A(f"PARENT {g} CHILD {n}")
                continue
            for meth in args.methods:
                nlives = args.nlive_met4 if meth == "met4" else [args.nlive]
                for nl in nlives:
                    suffix = f"_nlive{nl}" if meth == "met4" else ""
                    n = f"INF_{j:03d}_{k:03d}_{meth}{suffix}"
                    A(f"JOB {n} {sub_inf}")
                    mem = {"met1": args.mem_met1, "met3": args.mem_met3,
                           "met4": args.mem_met4}[meth]
                    A(f'VARS {n} suite="{args.suite}" block="{j}" cat="{k}" '
                      f'method="{meth}" nlive="{nl}" tag="{n}" mem="{mem}"')
                    A(f"RETRY {n} 2")
                    A(f"PARENT {g} CHILD {n}")
                    if meth == "met4":
                        # met4 reads met1's *_lambda0.npy as its mixture
                        # reference Lambda_0 -- a real ordering constraint, not
                        # cosmetic. Applies at EVERY met4 nlive.
                        A(f"PARENT INF_{j:03d}_{k:03d}_met1 CHILD {n}")
        A("")

    text = "\n".join(L) + "\n"
    suite_dir = HERE / args.suite
    out = args.out or (suite_dir / ("sbc.dag" if args.first_block == 0
                                    else f"sbc_from{args.first_block:03d}.dag"))
    suite_dir.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)

    # DAG throttle: inference jobs are long and memory-hungry; without a cap a
    # full suite floods the negotiator with thousands of idle jobs at once.
    (suite_dir / "sbc.dagconfig").write_text(
        f"DAGMAN_MAX_JOBS_IDLE = {args.max_idle}\n"
        f"DAGMAN_MAX_JOBS_SUBMITTED = {args.max_submitted}\n")

    if args.combined:
        _per = 1
    else:
        _per = sum(len(args.nlive_met4) if m == "met4" else 1
                   for m in args.methods)
    _nb = args.n_blocks - args.first_block
    n_inf = _nb * args.n_catalogs * _per
    print(f"[dag] {out}", file=sys.stderr)
    print(f"[dag] {_nb} gen (blocks {args.first_block}..{args.n_blocks-1}) + "
          f"{n_inf} inference jobs", file=sys.stderr)
    print(f"[dag] submit with:  condor_submit_dag {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
