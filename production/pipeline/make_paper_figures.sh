#!/usr/bin/env bash
# Regenerate every figure in popGMM_paper_arXiv/Figures/ from the existing
# dynesty checkpoints and NPZ/CSV caches (stage 4 of pipeline/README.md).
#
#   bash pipeline/make_paper_figures.sh [--sanity] [--compare <arXiv Figures dir>] [fig ...]
#
#   fig       any of: sanity neff_xi variance pull pp mdc gwtc3 ksad  (default: all but sanity)
#   --sanity  also run the toy model (sanity_test_methods.py RUNS dynesty and is
#             stochastic, so it will not be byte-identical to the paper's)
#   --compare render each regenerated PDF and the paper's to PNG and report
#             the fraction of differing pixels
#
# Scripts run from this pipeline/ copy; the working directory is the DATA root,
# because every data path in the scripts is relative to production/. The data
# root is the parent of pipeline/ when it holds sim_cat6_sharp_w8/, else
# $POPNFLOW_PROD. Writes ONLY under pipeline/output/ -- result directories are
# read, never written. Paper-named PDFs land in pipeline/output/Figures/.
set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${PIPE}/../sim_cat6_sharp_w8" ]]; then
    PROD="$(cd "${PIPE}/.." && pwd)"
else
    PROD="${POPNFLOW_PROD:-/home/manchun.yeung/population/simon/popnflow/production}"
fi
cd "${PROD}"
PY="${PYTHON:-/home/manchun.yeung/.conda/envs/pop/bin/python}"   # needs jax, dynesty, corner, fitz
S="${PIPE}/plotting"
OUT="${PIPE}/output"
FIG="${OUT}/Figures"
mkdir -p "${FIG}" "${OUT}/logs"
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
echo "data root: ${PROD}"

DIAG=sim_cat6_sharp_w8/paper_diagnostics
COMPARE=""
FIGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sanity)  FIGS+=(sanity); shift ;;
        --compare) COMPARE="$2"; shift 2 ;;
        *)         FIGS+=("$1"); shift ;;
    esac
done
[[ ${#FIGS[@]} -eq 0 || ( ${#FIGS[@]} -eq 1 && ${FIGS[0]} == sanity ) ]] && \
    FIGS+=(neff_xi variance pull pp mdc gwtc3 ksad)

run() {  # run <name> <cmd...>: log to output/logs/<name>.log
    local name="$1"; shift
    echo "[${name}] running (log: output/logs/${name}.log)"
    "$@" > "${OUT}/logs/${name}.log" 2>&1 || { echo "[${name}] FAILED, see ${OUT}/logs/${name}.log"; return 1; }
}

fig_sanity() {   # corner_sanity.pdf
    run sanity "${PY}" "${PIPE}/inference/sanity_test_methods.py" --outdir "${OUT}/sanity" --tag sanity
    cp "${OUT}/sanity/corner_sanity.pdf" "${FIG}/corner_sanity.pdf"
}

fig_neff_xi() {  # N_eff + xi scans: one invocation so both share one GMM fit
    run neff_xi "${PY}" "${S}/make_diagnostic_plots.py" --run mdc --catalog 0 \
        --indir sim_cat6_sharp_w8/inference \
        --sampler-pattern "simcat_cat{i:03d}_{method}.h5" \
        --method-patterns "met3=simcatnm_cat{i:03d}_{method}.h5" \
        --pe-m1det ${DIAG}/m1det.txt --pe-m2det ${DIAG}/m2det.txt --pe-dL ${DIAG}/dL.txt \
        --injection-file ${DIAG}/injections_endo3schema.h5 \
        --xi-gmm-sizes 100000 300000 \
        --neff-results-cache ${DIAG}/neff_cache_cat000_100k.npz \
        --xi-results-cache   ${DIAG}/xi_cache_cat000_100k.npz \
        --outdir "${OUT}/neff_xi" --make-neff-scan-plot --make-xi-scan-plot
    cp "${OUT}/neff_xi/mdc_neff_alpha_gamma.pdf" "${FIG}/mdc_neff_alpha_gamma.pdf"
    cp "${OUT}/neff_xi/mdc_xi_scan.pdf"          "${FIG}/mdc_xi_scan.pdf"
}

fig_variance() { # variance vs N_samp + fit-variance floor: post-processing of two NPZ caches
    run variance "${PY}" "${S}/plot_variance_budget_left_numeric.py" --run mdc \
        --fitvar-cache   ${DIAG}/fitvar_cache_cat000.npz \
        --variance-cache ${DIAG}/variance_cache_3method_cat000_2e4.npz \
        --outdir "${OUT}/variance"
    cp "${OUT}/variance/mdc_variance_budget_left_numeric.pdf" "${FIG}/mdc_variance_Nsamp.pdf"
}

fig_pull() {     # pull box plot, 50 fixed-Lambda catalogs, nlive=1000 + box/Cholesky fixes
    run pull "${PY}" "${S}/pp_bias_diagnostics.py" plot \
        --csv sim_cat6_sharp_w8/pit_bias_cat6_met124_nlive1000.csv \
        --outdir "${OUT}/pull" --methods met1 met3 met4 --mathtext \
        --membership sim_cat6_sharp_w8/mdc_membership.npy
    cp "${OUT}/pull/pull_boxplot.pdf" "${FIG}/pull_boxplot.pdf"
}

fig_pp() {       # PP plot = SBC coverage PP, n=20 prior draws x 8 catalogs
    # sbc_16 writes figures/ and coverage/ INSIDE the suite dir, so give it a
    # staging suite whose only real content is a link to the PIT files.
    local st="${OUT}/sbc_stage"
    mkdir -p "${st}"
    ln -sfn "${PROD}/sbc_suite/sbc_n10/pit" "${st}/pit"
    run pp "${PY}" "${PIPE}/sbc/sbc_16_coverage_pp.py" --suite "${st}" \
        --methods met1 met3_cholfix met4_cholfix --out coverage_pp_cholfix
    cp "${st}/figures/coverage_pp_cholfix.pdf" "${FIG}/pp_plot_per_parameter.pdf"
}

fig_mdc() {      # sim_cat6 cat000 corner + R(m1), R(z): met1 inference/, met3/met4 nlive=1000 boxfix+cholfix
    local st="${OUT}/stage_mdc"
    mkdir -p "${st}"
    # make_plots_met4.py only resolves {run}_{seed}met{N}.pkl; restore() is pickle-based regardless
    ln -sfn "${PROD}/sim_cat6_sharp_w8/inference/simcat_cat000_met1.h5"                               "${st}/mdc_0met1.pkl"
    ln -sfn "${PROD}/sim_cat6_sharp_w8/inference_boxfix_cholfix_nlive1000/simcatnm_cat000_met3.h5"    "${st}/mdc_0met3.pkl"
    ln -sfn "${PROD}/sim_cat6_sharp_w8/inference_met4_nlive1000_boxfix_cholfix/simcat_cat000_met4.h5" "${st}/mdc_0met4.pkl"
    run mdc "${PY}" "${S}/make_plots_met4.py" --run mdc --seed 0 --indir "${st}" --outdir "${OUT}/mdc"
    cp "${OUT}/mdc/mdc_0_corner_all.pdf"   "${FIG}/mdc_corner_pos.pdf"
    cp "${OUT}/mdc/mdc_Rm1_posteriors.pdf" "${FIG}/mdc_Rm1.pdf"
    cp "${OUT}/mdc/mdc_Rz_posteriors.pdf"  "${FIG}/mdc_Rz.pdf"
}

fig_gwtc3() {    # GWTC-3, pairing model, the 2026-05-05 seed-69 Colab run
    run gwtc3 "${PY}" "${S}/make_plots_met4.py" --run gwtc3 --seed 69 --model pairing \
        --indir results/data/inference --outdir "${OUT}/gwtc3"
    cp "${OUT}/gwtc3/gwtc3_69_corner_all.pdf"  "${FIG}/gwtc3_corner_pos.pdf"
    cp "${OUT}/gwtc3/gwtc3_Rm1_posteriors.pdf" "${FIG}/gwtc3_Rm1.pdf"
    cp "${OUT}/gwtc3/gwtc3_Rz_posteriors.pdf"  "${FIG}/gwtc3_Rz.pdf"
}

fig_ksad() {     # KS/AD of the STORED per-event GMMs, m1_det only (from the per-event CSV)
    mkdir -p "${OUT}/ksad"
    run ksad "${PY}" "${S}/replot_ks_ad.py" --mode stored \
        --csv gmm_ks_ad/ks_ad_stored.csv --null-npz gmm_ks_ad/null_calibration.npz \
        --out "${OUT}/ksad/ks_ad_stored_m1"
    cp "${OUT}/ksad/ks_ad_stored_m1.pdf" "${FIG}/gmm_fit_m1.pdf"
}

FAILED=()
for f in "${FIGS[@]}"; do "fig_${f}" || FAILED+=("$f"); done

if [[ -n "${COMPARE}" ]]; then
    "${PY}" - "${FIG}" "${COMPARE}" <<'EOF'
import sys, pathlib, numpy as np, fitz
new, ref = map(pathlib.Path, sys.argv[1:])
def px(p):
    pm = fitz.open(p)[0].get_pixmap(dpi=100)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)
for r in sorted(ref.glob("*.pdf")):
    n = new / r.name
    if not n.exists():
        print(f"{r.name:28s} not regenerated"); continue
    a, b = px(n), px(r)
    if a.shape != b.shape:
        print(f"{r.name:28s} page size differs {a.shape[:2]} vs {b.shape[:2]}"); continue
    d = (np.abs(a.astype(int) - b.astype(int)).max(-1) > 16).mean()
    print(f"{r.name:28s} {100*d:6.3f}% pixels differ")
EOF
fi

echo "figures -> ${FIG}"
[[ ${#FAILED[@]} -eq 0 ]] || { echo "FAILED: ${FAILED[*]}"; exit 1; }
