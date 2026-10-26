#!/usr/bin/env bash
# ============================================================================
# Condor executable wrapper for one SBC stage.
#
# WHY A WRAPPER, not `executable = .../python`: under
# should_transfer_files=YES, HTCondor copies the interpreter BINARY into the
# job sandbox and runs that copy, severed from its site-packages/encodings
# tree, so Python dies at startup before any of our code runs. A bash script
# needs nothing but itself to start, and invoking the real interpreter by
# absolute path from inside runs it IN PLACE. Same reason as
# bilby_pe_cat6/run_pe_event.sh; matches the convention every other
# production/*.sub already uses.
#
# Usage (from the .sub `arguments`):
#   run_sbc_block.sh gen   <suite> <block> <n_catalogs> <n_events> <n_jobs>
#   run_sbc_block.sh infer <suite> <block> <catalog> <method> <nlive>
# ============================================================================
set -euo pipefail

# JAX MUST NOT PROBE FOR A GPU. sim_cat_inference.py imports jax, and jaxlib's
# CUDA plugin aborts the process on any execute node without a visible GPU:
#   XlaRuntimeError: FAILED_PRECONDITION: No visible GPU devices.
#   RuntimeError: Unable to initialize backend 'cuda'
# raised at sim_cat_inference.py:389, BEFORE the --method branch, so it kills
# met1/met3/met4 alike. It is node-dependent, so failures look random: jobs
# that land on a GPU node succeed and the rest die within seconds. 132 jobs
# were lost to this. The fixed-Lambda sweep never hit it only because of where
# its jobs happened to land -- run_sim_cat.sub does not set this either.
#
# Set here in the WRAPPER rather than the .sub `environment` line on purpose:
# the wrapper is read from the shared filesystem at job start, so this repairs
# jobs ALREADY QUEUED, whereas a submit-file change only affects future
# submissions.
export JAX_PLATFORMS=cpu

PYTHON=/home/manchun.yeung/.conda/envs/pop/bin/python
PROD=/home/manchun.yeung/population/simon/popnflow/production
HERE="${PROD}/sbc_suite"

MODE="$1"; shift

case "${MODE}" in
  gen)
    SUITE="$1"; BLOCK="$2"; NCAT="$3"; NEV="$4"; NJOBS="$5"
    exec "${PYTHON}" "${HERE}/sbc_01_run_block.py" \
        --suite "${SUITE}" --block "${BLOCK}" \
        --n-catalogs "${NCAT}" --n-events "${NEV}" \
        --n-jobs "${NJOBS}" --skip-existing
    ;;
  infer)
    SUITE="$1"; BLOCK="$2"; CAT="$3"; METHOD="$4"; NLIVE="$5"
    # OPTIONAL args 6-8, added 2026-08-21 for the PE-bounds box fix. Defaults
    # reproduce the previous behaviour EXACTLY, so the existing sbc.dag (which
    # passes only 5 args) is bit-for-bit unchanged.
    #   $6,$7 = --pe-bounds-pct LO HI. Default 1.5/98.5 = the old hardcoded box.
    #           Measured: that box sits INSIDE the data (PE samples fall outside
    #           it 8.02% of the time) and biases beta low by ~0.66. Use
    #           0.02 99.98 -- the PE sample EDGE -- which recovers +0.646 while
    #           still bounding the GMM's unbounded tails.
    #   $8    = OUTTAG. Appended to the output directory so a rerun CANNOT
    #           overwrite the baseline: met3 -> inference_<tag>,
    #           met4 -> inference_met4_nlive<N>_<tag>. Empty = old paths.
    PEB_LO="${6:-1.5}"; PEB_HI="${7:-98.5}"; OUTTAG="${8:-}"
    # HERE, not PROD: suites live under sbc_suite/, so a suite named "sbc_n10"
    # is at sbc_suite/sbc_n10, NOT production/sbc_n10.  Using PROD here made
    # every inference job die instantly with
    #   FileNotFoundError: .../production/sbc_n10/blocks/block_NNN/event_gmms.h5
    # while the gen jobs succeeded, because gen delegates path resolution to
    # sbc_01_run_block.py (which resolves against HERE) and only this branch
    # built the path itself.
    BDIR="${HERE}/${SUITE}/blocks/$(printf 'block_%03d' "${BLOCK}")"
    # met3 uses the simcatnm_ prefix, met1/met4 use simcat_ -- this mirrors the
    # fixed-Lambda naming so sbc_02's default --pattern finds them.
    #
    # met4 GETS ITS OWN nlive-TAGGED OUTPUT DIR. sim_cat_inference.py names its
    # output `{run}_cat{k}_met4.h5` with NO nlive in the filename, so two met4
    # runs at different nlive would silently overwrite each other in a shared
    # directory. The fixed-Lambda arm solved this the same way --
    # inference_met4_nlive200/ vs inference_met4_nlive1000/ -- and this mirrors
    # it so both can coexist and be extracted separately.
    CATPAD=$(printf '%03d' "${CAT}")
    if [ "${METHOD}" = "met4" ]; then
        OUTDIR="${BDIR}/inference_met4_nlive${NLIVE}${OUTTAG:+_${OUTTAG}}"
        # BUG FIXED HERE (2026-08-15): sim_cat_inference.py computes
        # OUT_MED1 = args.outdir / "{run}_cat{k}_met1_lambda0.npy" -- i.e.
        # ALWAYS relative to THIS run's own --outdir. Since met4 gets its own
        # nlive-tagged outdir (above), met4 was looking for met1's lambda0 in
        # inference_met4_nlive200/, where it never gets written -- met1 always
        # writes it into inference/. Every met4 job failed deterministically
        # with "neither simcat_cat000_met1_lambda0.npy nor
        # simcat_cat000_met1.h5 exists", on every retry, because the file can
        # never appear where it was looking. Not a race: the DAG's
        # PARENT met1 CHILD met4 edge was already correctly ordering things
        # (verified: met1 had completed and its file existed in inference/ by
        # the time met4 ran) -- met4 was just checking the wrong directory.
        #
        # Fix: copy met1's Lambda_0 into met4's own outdir under the same
        # filename, so met4's OUT_MED1.exists() check finds it. Fails loudly
        # if met1's file is missing rather than silently falling through to
        # sim_cat_inference.py's OUT_MET1.exists() fallback, which would
        # re-derive Lambda_0 by re-loading and re-medianing met1's full
        # posterior -- correct, but means an ordering bug here would go
        # unnoticed instead of erroring.
        mkdir -p "${OUTDIR}"
        LAMBDA0_SRC="${BDIR}/inference/simcat_cat${CATPAD}_met1_lambda0.npy"
        if [ -f "${LAMBDA0_SRC}" ]; then
            cp -n "${LAMBDA0_SRC}" "${OUTDIR}/simcat_cat${CATPAD}_met1_lambda0.npy"
        else
            echo "FATAL: ${LAMBDA0_SRC} missing -- met1 must complete for this" \
                 "catalog before met4 can run (DAG ordering should guarantee" \
                 "this; if you see this, the PARENT/CHILD edge did not hold)" >&2
            exit 1
        fi
    else
        OUTDIR="${BDIR}/inference${OUTTAG:+_${OUTTAG}}"
    fi
    exec "${PYTHON}" "${PROD}/sim_cat_inference.py" \
        --indir "${BDIR}" \
        --outdir "${OUTDIR}" \
        --catalog-id "${CAT}" \
        --method "$( [ "${METHOD}" = "all" ] && echo all || echo "method${METHOD#met}" )" \
        --gmm-file "${BDIR}/event_gmms.h5" \
        --kcol k_argmin \
        --injection-file "${PROD}/sim_cat6_sharp_w8/injections.h5" \
        --run simcat \
        --truncate-peak \
        --live-points "${NLIVE}" \
        --pe-bounds-pct "${PEB_LO}" "${PEB_HI}" \
        --skip-if-exists
    # --truncate-peak makes the likelihood's Gaussian peak share the
    # GENERATOR's support, [m_min, m_max]. Without it the likelihood keeps peak
    # density above m_max that sim_cat_03 cannot produce, biasing xi(Lambda).
    # 0.000% at the fixed-Lambda MDC point, but ~4% for a prior draw whose peak
    # sits ~1 sigma from m_max -- which SBC's Lambda_j routinely do. It is OFF
    # by default in sim_cat_inference.py, so the fixed-Lambda arm is unchanged.
    # --run simcat is sim_cat_inference.py's own default, passed explicitly so
    # the output names stay simcat_cat{k}_met1.h5 / simcatnm_cat{k}_met3.h5,
    # which is what sbc_02_extract_pit.py's default --pattern looks for.
    ;;
  *)
    echo "unknown mode '${MODE}' (expected gen|infer)" >&2
    exit 2
    ;;
esac
