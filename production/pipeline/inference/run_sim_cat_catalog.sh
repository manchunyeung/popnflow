#!/usr/bin/env bash
# ============================================================================
# Wrapper invoked by HTCondor for each sim_cat6_sharp_w8 catalog job.
#
# Called by Condor as:
#   NLIVE=<n> run_sim_cat_catalog.sh <catalog_id> <method> [<catalog_file>]
#
# - <catalog_id>:   0, 1, 2, ... Used for output filenames and, when
#                   <catalog_file> is empty, to resolve the default
#                   {indir}/mdc_cat{catalog_id:03d}.h5 path.
# - <method>:       method1 | method3 | method4 | all
#                   method1 and method3 have been smoke-tested against this
#                   catalog and run to full dlogz<0.1 convergence at nlive=25
#                   (logs/smoke_cat6.log, logs/smoke_gmmload2.log -- 381
#                   iterations, ~17.8min at nlive=25; nlive=200 below will
#                   take substantially longer, budget accordingly).
#                   method4's Lambda0 dependency is now wired up: it reads
#                   Method 1's posterior median for this catalog from the
#                   finished met1 sweep (see the LAMBDA0 block below), and
#                   all 50 of those files exist as of 2026-08-06. method4
#                   itself is still UNSMOKE-TESTED on this catalog set --
#                   canary a single catalog before queueing all 50.
# - $NLIVE (env):   OPTIONAL dynesty live points, default 1000. Method 4 uses
#                   it to pick its output directory too (see OUTDIR below),
#                   which is what keeps the nlive=200 "quick" sweep and the
#                   nlive=1000 "detailed" sweep from overwriting each other.
#                   Set by the submit file's `environment = "NLIVE=..."`.
# - <catalog_file>: OPTIONAL explicit override. Needed for the peak-SELECTED
#                   realizations (mdc_cat2476_peak45.h5, ...), whose names
#                   don't follow the zero-padded {ID:03d} convention that
#                   sim_cat_inference.py resolves catalog_id against. Pass ""
#                   to use the default path.
#
# Unlike run_one_catalog.sh (which drives mdc_inference_multicatalog.py and
# draws its OWN catalog membership from --catalog-seed at run time),
# sim_cat_inference.py reads catalogs already built by
# sim_cat_04_build_catalogs.py / sim_cat_04b_select_catalogs.py, so there is
# no --catalog-seed here.
#
# Method 3/4's per-event GMMs are LOADED, not fit inline: sim_cat_fit_event_
# gmms.py (2026-08-05) fit every one of the 276 pool events ONCE via BIC
# (K=1..20, matching select_k_per_event.py's conventions), producing
# sim_cat6_sharp_w8/event_gmms.h5. That replaces the old inline-fit fallback
# of a FIXED K=7 for every event with a genuine per-event BIC choice (this
# pool: K_argmin median 5, range 3-13), and removes the catalog-dependent
# random_state jitter the inline path had (EFFECTIVE_SEED varies with
# catalog_id, so the same pool event previously got very slightly different
# GMMs in different catalogs; the loaded fit is one fixed random_state=0 fit
# per pool event, shared identically across every catalog that uses it).
# --kcol picks WHICH BIC choice to load (k_argmin vs k_plateau -- they came
# out identical for every event in this pool, so either works here).
# ============================================================================

set -euo pipefail

CATALOG_ID="${1:?missing catalog_id}"
METHOD="${2:?missing method}"
CATALOG_FILE="${3:-}"
# NLIVE comes from the ENVIRONMENT (Condor's `environment = "NLIVE=200"`),
# not a 4th positional arg: catalog_file is an optional MIDDLE argument, so a
# positional nlive would have to be preceded by an empty placeholder, and if
# Condor's argument tokenizer collapsed that placeholder instead of emitting
# an empty argv slot, nlive would land in $3 and be silently used as the
# catalog file -- a wrong-nlive sweep that still looks like it ran fine.
# PYTHON BY ABSOLUTE PATH. The bare `python` this script used relied on the
# execute node happening to have one on PATH; HTCondor does not guarantee that,
# and the 2026-08-21 boxfix sweep died on every job with
#   run_sim_cat_catalog.sh: line 176: python: command not found
# 71 jobs "completed" in ~4 minutes having produced nothing. sbc_suite/
# run_sbc_block.sh already resolves the interpreter absolutely for exactly this
# reason. Overridable via the environment if a different env is ever wanted.
# JAX MUST NOT PROBE FOR A GPU. jaxlib's CUDA plugin aborts the process on any
# execute node without a visible GPU:
#   XlaRuntimeError: FAILED_PRECONDITION: No visible GPU devices.
#   RuntimeError: Unable to initialize backend 'cuda'
# raised at sim_cat_inference.py:401, before any --method branch, so it kills
# every method alike. Node-dependent, so failures look RANDOM: jobs landing on
# a GPU node succeed, the rest die in seconds. sbc_suite/run_sbc_block.sh has
# carried this line since 132 jobs were lost to it; this runner never got it,
# and it killed 12/100 of the 2026-08-21 boxfix sweep the same way.
#
# Set in the WRAPPER, not the .sub `environment`: the wrapper is read from the
# shared filesystem at job start, so this repairs jobs ALREADY QUEUED, whereas
# a submit-file change only affects future submissions.
export JAX_PLATFORMS=cpu

PYTHON="${PYTHON:-/home/manchun.yeung/.conda/envs/pop/bin/python}"
NLIVE="${NLIVE:-1000}"
# PE-BOUNDS BOX + OUTPUT TAG, added 2026-08-21. Both from the ENVIRONMENT for
# the same reason NLIVE is (see the note above): catalog_file is an optional
# MIDDLE positional, so adding trailing positionals risks silent misparsing.
#   PEB_LO/PEB_HI -> --pe-bounds-pct. Defaults 1.5/98.5 reproduce the old
#     hardcoded box EXACTLY, so existing submit files are unchanged. Measured
#     2026-08-21: that box sits INSIDE the data (PE samples fall outside it
#     8.02% of the time), preferentially clips the high-q corner, and biases
#     beta LOW by ~0.66. Use 0.02 99.98 -- the PE sample EDGE -- which recovers
#     +0.646 while still bounding the GMM's unbounded tails.
#   OUTTAG -> appended to the output dir so a rerun CANNOT overwrite the
#     baseline sweep. Empty = old paths.
PEB_LO="${PEB_LO:-1.5}"
PEB_HI="${PEB_HI:-98.5}"
OUTTAG="${OUTTAG:-}"

cd /home/manchun.yeung/population/simon/popnflow/production

# ── Paths ────────────────────────────────────────────────────────────────
# sim_cat6_sharp_w8: sharp PL+Peak, q-boundary fix, width_ref_snr=8 (decouples
# GWMockCat's PE-width calibration from the rho_thr=9 detection threshold --
# see RUN_NOTES sec.8 and sim_cat_qsampling.py's module docstring).
INDIR="sim_cat6_sharp_w8"
MET1_DIR="${INDIR}/inference"       # where the finished met1/met3 sweep lives

# Method 4 writes to its OWN per-nlive directory; method1/method3 keep the
# original shared one. Two reasons this is not optional:
#   1. OUT_MET4 is {run}_cat{ID:03d}_met4.h5 with NO nlive in the name, so an
#      nlive=200 and an nlive=1000 sweep would collide file-for-file (and
#      --skip-if-exists would silently turn the second sweep into 50 no-ops).
#   2. Every job rewrites the shared per-catalog sidecar
#      {run}_cat{ID:03d}_info.json, so a met4 job landing in inference/ would
#      stamp ITS nlive over the record of the nlive=1000 met1/met3 runs
#      already there.
if [[ "${METHOD}" == "method4" ]]; then
    OUTDIR="${INDIR}/inference_met4_nlive${NLIVE}${OUTTAG:+_${OUTTAG}}"
else
    OUTDIR="${MET1_DIR}${OUTTAG:+_${OUTTAG}}"
fi

# ── Fixed across the whole sweep -- do not vary per catalog ────────────────
# NLIVE now comes from the environment (default 1000) so the met4 quick
# (nlive=200) and detailed (nlive=1000) sweeps can share this wrapper.
# 2026-08-05: raised from 200 to 1000 at Simon's request for met1/met3.
# Real cluster data at nlive=200 (39 finished jobs, cluster 4929634 before it
# was removed for an unrelated reason): Method 1 35-69min, Method 3 28-71min.
# That was MUCH faster than a linear extrapolation from a tiny local nlive=25
# smoke test predicted -- so do NOT trust a naive 5x linear scaling for
# nlive=1000 either. Measured nlive=1000 (2026-08-06, the finished cat6
# sweep): met1 and met3 both ran multi-hour, and the 7 met3 jobs that hit
# 11-16h were killed before reaching dlogz<0.1. Method 4's cost per
# likelihood call is NOT met1's -- canary one catalog before committing 50.
MASS_MODEL="sharp"          # matches this catalog's generation truth
BASE_SEED=1234
GMM_FILE="${INDIR}/event_gmms.h5"   # pre-fit per-event GMMs (method3/4 only)
KCOL="k_argmin"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "Catalog ${CATALOG_ID} | method ${METHOD} | nlive ${NLIVE} | file ${CATALOG_FILE:-<default>}"
echo "Outdir: ${OUTDIR}"
echo "Host: $(hostname)   Date: $(date)"
echo "============================================================"

ARGS=(
    --indir "${INDIR}"
    --outdir "${OUTDIR}"
    --catalog-id "${CATALOG_ID}"
    --method "${METHOD}"
    --live-points "${NLIVE}"
    --pe-bounds-pct "${PEB_LO}" "${PEB_HI}"
    --mass-model "${MASS_MODEL}"
    --seed "${BASE_SEED}"
    --skip-if-exists
)
if [[ -n "${CATALOG_FILE}" ]]; then
    ARGS+=(--catalog-file "${CATALOG_FILE}")
fi

# ── Method 4's reference Lambda0 comes from the FINISHED Method-1 sweep ────
# sim_cat_inference.py would resolve Lambda0 itself from OUT_MED1/OUT_MET1 --
# but only inside --outdir, and met4 no longer shares a directory with met1
# (see above). So pass it explicitly from the met1 sweep's saved posterior
# median. This also puts the exact Lambda0 in the job log and the sidecar's
# "mixture_lambda0" field, instead of leaving it implicit.
if [[ "${METHOD}" == "method4" ]]; then
    MED1="${MET1_DIR}/simcat_cat$(printf '%03d' "${CATALOG_ID}")_met1_lambda0.npy"
    if [[ ! -f "${MED1}" ]]; then
        echo "ERROR: Method 4 needs Method 1's posterior median for catalog" \
             "${CATALOG_ID}, but ${MED1} does not exist." >&2
        echo "Run --method method1 for this catalog first." >&2
        exit 1
    fi
    LAMBDA0=$("${PYTHON}" -c "
import numpy as np, sys
a = np.load(sys.argv[1])
if a.shape != (8,):
    sys.exit(f'expected 8 values in {sys.argv[1]}, got {a.shape}')
print(','.join('%.17g' % v for v in a))
" "${MED1}")
    echo "Method-4 Lambda0 <- ${MED1##*/}: ${LAMBDA0}"
    ARGS+=(--mixture-lambda0 "${LAMBDA0}")
fi
# sim_cat_inference.py builds the per-event GMMs UNCONDITIONALLY, before
# branching on --method (confirmed: "Skipping Method 1" prints, then the GMM
# fit/load runs anyway) -- so a method1-only job pays for an inline GMM fit
# it never uses. Always passing --gmm-file replaces that wasted fit with a
# fast load, for every method, at no cost.
ARGS+=(--gmm-file "${GMM_FILE}" --kcol "${KCOL}")

# ── INJFILE (env, OPTIONAL): the selection set ───────────────────────────────
# Unset -> sim_cat_inference.py uses ${INDIR}/injections.h5, the exact
# reweighted injection campaign. This is the production path and is unchanged.
# Set   -> any file in the injections.h5 schema, which is how the selection-GMM
# comparison swaps in a GMM-derived selection set without touching the
# inference script (see selection_gmm_check/build_gmm_injections.py).
if [[ -n "${INJFILE:-}" ]]; then
    if [[ ! -f "${INJFILE}" ]]; then
        echo "ERROR: INJFILE=${INJFILE} does not exist" >&2
        exit 1
    fi
    echo "Injection file (override): ${INJFILE}"
    ARGS+=(--injection-file "${INJFILE}")
fi

"${PYTHON}" sim_cat_inference.py "${ARGS[@]}"

echo "============================================================"
echo "Catalog ${CATALOG_ID} | method ${METHOD} | nlive ${NLIVE} | DONE at $(date)"
echo "============================================================"
