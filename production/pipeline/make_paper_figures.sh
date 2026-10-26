#!/usr/bin/env bash
# Regenerate every figure in popGMM_paper_arXiv/Figures/ with the per-figure
# scripts in figures/ (one script per paper PDF, same file name).
#
#   bash pipeline/make_paper_figures.sh [--paper-bug] [--compare <arXiv Figures dir>] [name ...]
#
#   name         paper figure stem(s), e.g. mdc_Rz pull_boxplot (default: all 13)
#   --paper-bug  reproduce the published versions of the three figures whose
#                generating code had a bug (corner_sanity, mdc_Rz, gwtc3_Rz; see README)
#   --compare    render each regenerated PDF and the paper's at 100 dpi and
#                report the fraction of differing pixels
#
# The scripts read data relative to production/, so this runs from the data
# root: the parent of pipeline/ when it holds sim_cat6_sharp_w8/, else
# $POPNFLOW_PROD. Output goes ONLY to pipeline/output/Figures/ (+ logs/).
set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${PIPE}/../sim_cat6_sharp_w8" ]]; then
    PROD="$(cd "${PIPE}/.." && pwd)"
else
    PROD="${POPNFLOW_PROD:-/home/manchun.yeung/population/simon/popnflow/production}"
fi
cd "${PROD}"
PY="${PYTHON:-/home/manchun.yeung/.conda/envs/pop/bin/python}"   # needs jax, dynesty, corner, fitz
FIG="${PIPE}/output/Figures"
LOG="${PIPE}/output/logs"
mkdir -p "${FIG}" "${LOG}"
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "data root: ${PROD}"

ALL=(corner_sanity mdc_neff_alpha_gamma mdc_xi_scan mdc_variance_Nsamp
     pp_plot_per_parameter pull_boxplot mdc_corner_pos mdc_Rm1 mdc_Rz
     gwtc3_corner_pos gwtc3_Rm1 gwtc3_Rz gmm_fit_m1)
BUGGY=" corner_sanity mdc_Rz gwtc3_Rz "
PAPER_BUG=0
COMPARE=""
FIGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --paper-bug) PAPER_BUG=1; shift ;;
        --compare)   COMPARE="$2"; shift 2 ;;
        *)           FIGS+=("$1"); shift ;;
    esac
done
[[ ${#FIGS[@]} -eq 0 ]] && FIGS=("${ALL[@]}")

FAILED=()
for f in "${FIGS[@]}"; do
    extra=()
    [[ ${PAPER_BUG} -eq 1 && "${BUGGY}" == *" ${f} "* ]] && extra=(--paper-bug)
    echo "[${f}] running (log: output/logs/${f}.log)"
    if ! "${PY}" "${PIPE}/figures/fig_${f}.py" --out "${FIG}/${f}.pdf" "${extra[@]}" \
            > "${LOG}/${f}.log" 2>&1; then
        echo "[${f}] FAILED, see ${LOG}/${f}.log"; FAILED+=("$f")
    fi
done

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
