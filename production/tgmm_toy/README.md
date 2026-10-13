# tgmm_toy — does the per-event GMM leak, and does a truncated GMM fix it?

Prompted by Mancarella's comment on the paper (truncated GMM, P2600539 App. D).
1D toy: posterior samples on D = [-1, 1], N = 5000, fit with the PRODUCTION
recipe (`sim_cat_fit_event_gmms.py`: sklearn full-cov, reg_covar=1e-6,
n_init=5, BIC argmin over K=1..20), turned into a per-event density four ways:

| name | what |
|---|---|
| `gmm` | plain GMM, unbounded |
| `box` | production met3/met4: GMM × 1[x in sample 0.02–99.98 pct], not renormalised (`--pe-bounds-pct 0.02 99.98`) |
| `gmm_trunc` | same GMM restricted to D and renormalised (post-hoc truncation) |
| `tgmm` | mixture of normals each truncated+normalised on D, fit by ML (jax grad), own BIC-selected K |

## Run

    python tgmm_toy.py --outdir out --njobs 96     # ~50 min on ~16 effective cores
    python plot_tgmm_toy.py out                     # tables + fig1, fig2
    python ks_tgmm_toy.py out                       # KS table
    python plot_one_case.py out "peak at edge TN(1, 0.3)" 0

Use the `pop` env (needs jax + sklearn).

## Results (2026-10-13)

**Part 1 – single posteriors (10 seeds each).** The plain GMM leaks 0.1% (interior)
to ~1% (edge-touching) of its mass outside D. The box removes only ~0.05% of
in-domain mass, so GMM / box / post-hoc trunc are the same density inside D.
The real difference is shape: to fake a hard edge BIC picks K≈7–11, the fit
tracks sampling noise (±10–25% wiggles) and drops off before the edge.
TGMM uses K=1 (correct family here).

KS distance to the true CDF, sup|F_fit − F_true| (mean over seeds; 5% one-sample bound 0.019):

| case | GMM | box (prod.) | post-hoc | TGMM |
|---|---|---|---|---|
| interior TN(0,0.3) | 0.006 | 0.006 | 0.006 | 0.006 |
| near edge TN(0.85,0.2) | 0.013 | 0.018 | 0.017 | 0.006 |
| peak at edge TN(1,0.3) | 0.013 | 0.019 | 0.018 | 0.006 |
| ramp ∝ (1+x)^3 | 0.015 | 0.020 | 0.019 | 0.010 |
| uniform | 0.013 | 0.015 | 0.014 | 0.005 |

The excess is concentrated in the outer 10% of the domain. Truncating after the
fit renormalises the edge deficit over the whole domain, so box/post-hoc are
slightly *worse* than plain GMM in CDF terms. Formal KS (held-out draws, which
need the two-sample bound 0.027) rejects nothing — KS is insensitive at edges.
See `out/case_peakatedgeTN103_seed0.png`.

**Part 2 – hierarchical λ** (pop p(x|λ) ∝ e^{λx}, λ_true = 2, flat PE prior on D,
Gaussian noise σ ~ U(0.1, 0.5) → exact truncated-normal posteriors; 20 catalogs ×
200 events; no selection). Shift of the λ posterior mean vs the EXACT posterior:

| method | pop support = D | pop on [-1.5,1.5] (see note) |
|---|---|---|
| GMM | −0.171 ± 0.008 σ | −0.001 ± 0.001 σ |
| box (prod.) | **−0.186 ± 0.008 σ** | −0.050 ± 0.001 σ |
| post-hoc trunc | −0.171 ± 0.008 σ | −0.047 ± 0.001 σ |
| TGMM | **+0.001 ± 0.003 σ** | −0.000 ± 0.001 σ |

All 20 catalogs shift low for the GMM variants in the pop-support = D case
(−0.11 to −0.28 σ); widths are unchanged. A fixed per-event bias grows ∝ √N in
σ units.

Note on the wide-population column: this scenario is MISSPECIFIED by
construction (population normalised on [-1.5, 1.5] but only events with true
x in D are drawn, no selection term), so even the EXACT posterior is off the
truth by −1.34 (14 σ). Method-vs-exact shifts are all ≤ 0.05 σ, and the plain
GMM's leak (which gets population weight there) happens to cancel its edge
deficit. Treat it as inconclusive; a proper test needs a selection term for
the PE-prior bound.

**Caveats.** The TGMM is the exact model family for these truncated-normal
posteriors, which flatters it; on the ramp it sometimes underfits (K=1,
early-stop with patience=2 in the K scan). The production q ≤ 1 boundary is the
diagonal m2 = m1 in (m1_det, m2_det), which an axis-aligned box cannot truncate at all.
