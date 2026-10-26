#!/usr/bin/env bash
# Report whether each pipeline/ copy still matches its original in production/
# (SOURCES.md5 records the md5 at copy time). Data root as in make_paper_figures.sh.
#   bash pipeline/check_sources.sh
set -uo pipefail
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "${PIPE}/../sim_cat6_sharp_w8" ]]; then PROD="$(cd "${PIPE}/.." && pwd)"
else PROD="${POPNFLOW_PROD:-/home/manchun.yeung/population/simon/popnflow/production}"; fi

status=0
while read -r md5 copy _ orig; do
    now_copy=$(md5sum < "${PIPE}/${copy}" | cut -c1-32)
    if [[ ! -f "${PROD}/${orig}" ]]; then
        printf '%-48s original missing (%s)\n' "${copy}" "${orig}"; status=1; continue
    fi
    now_orig=$(md5sum < "${PROD}/${orig}" | cut -c1-32)
    if   [[ ${now_copy} != "${md5}" ]]; then printf '%-48s COPY EDITED since gathering\n' "${copy}"; status=1
    elif [[ ${now_orig} != "${md5}" ]]; then printf '%-48s original changed: %s\n' "${copy}" "${orig}"; status=1
    else printf '%-48s ok\n' "${copy}"; fi
done < "${PIPE}/SOURCES.md5"
exit ${status}
