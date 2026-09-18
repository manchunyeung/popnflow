# Is the term-(B) floor real, and what makes it?

Complementary to `production/fitvar_npe_scan/` (which established `B ~ N_PE^-0.43`
on the MDC and `-0.80` on sim_cat6). That scan varied N_PE by resampling each
event's fixed 3343 PE samples **with replacement**, so by construction it could
not separate "B genuinely stops falling" from "the resampling stops resolving".
These four tests separate them.

**Headline: the floor is real — it is not a resampling artefact — and ~21% of it
is EM optimiser noise, concentrated in exactly the events that dominate B.**

Reproduces the published scan first: aggregate `B ~ N_PE^(-0.438)` vs their
`-0.429` (`summarize_published.py`).

## 1. Bootstrap granularity — REFUTED (`run_arms.py`)

A with-replacement resample of size *m* from *n* atoms holds only
`n(1-e^{-m/n})` distinct points (63% at m=n), whereas a real PE run of *m*
samples has *m*. That deficit grows with m/n and mechanically flattens the
curve; it accounts for **26%** of the departure from -1 on arithmetic alone.

So the scan was re-run with a second arm drawing **without** replacement
(`rng.permutation(nsamp)[:N]`) — N genuinely distinct samples, exactly what a
shorter PE run delivers — corrected by the finite-population factor
`(n-1)/(n-N)`, and capped at `N <= n/2` where that first-order correction is
trustworthy. 69 events, R=60, identical Λ draws / q* pool / GMM settings.

| N_PE | boot | srs (FPC-corrected) | FPC | srs/boot |
|-----:|-----:|-----:|----:|----:|
| 209 | 0.7483 | 0.7492 | 1.066 | **1.001** |
| 418 | 0.4795 | 0.5130 | 1.143 | 1.070 |
| 836 | 0.3551 | 0.4350 | 1.333 | 1.225 |
| 1671 | 0.2843 | 0.4373 | 1.999 | 1.538 |

The arms agree to **0.1%** at N/n = 0.06, which cross-validates both estimators.
Over 209→836 they give `-0.642` and `-0.547` (uncorrected `-0.646`). Neither is
1/N. **The bootstrap is not manufacturing the floor.** The two diverge as N/n
grows, which is the FPC's first-order limit, not a granularity effect — the
granularity bias would push the other way.

## 2. Duplicated PE samples — REFUTED (`check_duplicates.py`)

If the MDC posteriors had been produced by resampling weighted nested-sampling
output, the nominal 3343 would hide far fewer distinct locations and nothing
could recover 1/N. All 69 events have **3343/3343 distinct** samples. Dead.

## 3. There is no floor in the estimator itself (`run_fresh.py`)

Nothing above N_PE = 3343 is reachable by resampling the real file, so: build a
reference truth (K=20 GMM on the event's full samples), draw **fresh iid**
samples at N up to 106,976, fit the production K=7 GMM, same q* pool, R=40.
Preserves the real misspecification (K=7 approximating a richer density).

5 of 6 events give clean `1/N` — `N*V` flat across a 512x range:

| ev | p_fresh (all) | p_fresh (N>=3343) | p_bootstrap (published) |
|---:|---:|---:|---:|
| 55 | -0.968 | -1.022 | -0.071 |
| 17 | -1.100 | -1.063 | +0.126 |
| 60 | -1.074 | -1.132 | -0.035 |
| 39 | **-0.206** | **-0.401** | -0.123 |
| 43 | -0.864 | -0.628 | -1.052 |
| 6 | -1.017 | -1.016 | -0.893 |

So M-estimator asymptotics do hold for this estimator when the density is
smooth. **The floor is a property of the real posteriors, not of the machinery.**

Not a pre-asymptotic window effect either — on the *same* N range the real-data
arms can probe (209→1671):

| ev | fresh | boot (real) | srs (real) |
|---:|---:|---:|---:|
| 55 | -0.768 | -0.126 | +0.001 |
| 17 | -1.129 | -0.011 | -0.246 |
| 60 | -0.832 | -0.025 | +0.006 |
| 43 | -1.386 | -1.163 | -0.966 |
| 6 | -1.242 | -0.716 | -0.712 |

Controls 43/6 scale in both. 55/17/60 scale on a smooth surrogate and are flat
on real data. Event 39 is flat in **both** — a separate, geometry-intrinsic case.

Caveat: the surrogate is a GMM, so its tails are Gaussian mixtures. If the real
posteriors have heavier tails than any K=20 GMM this arm is optimistic.
`--inflate` probes that and was not swept.

## 4. EM optimiser noise — CONFIRMED, ~21% of B (`run_emvar.py`)

Hold the **data completely fixed** (all 3343 samples every replicate) and vary
only the EM `random_state`. Any surviving variance is pure optimiser noise and
cannot depend on N_PE. 69 events, R=60, n_init=5 (production).

```
emseed (data fixed, EM seed varies) : V = 0.0414
both   (data resampled + seed varies): V = 0.2126
published B(N_PE=3343) = 0.1945 ; two-component B_inf = 0.173
```

**EM-only variance is 21% of B(3343) and 24% of B_inf** — and it is concentrated
in precisely the events that carry B:

| ev | p_e (pub) | V_pub | V_emseed | V_both | sd(lower_bound_) | distinct optima /60 |
|---:|---:|---:|---:|---:|---:|---:|
| 39 | -0.123 | 0.01174 | 0.01522 | 0.01561 | 4.0e-02 | 34 |
| 47 | -0.267 | 0.00905 | 0.01264 | 0.01095 | 3.6e-02 | 55 |
| 55 | -0.071 | 0.01317 | 0.00624 | 0.01392 | 2.5e-02 | 27 |
| 60 | -0.035 | 0.01197 | 0.00293 | 0.01184 | 1.6e-02 | 34 |

For events 39 and 47 the EM-only variance **equals or exceeds** the full
bootstrap variance: essentially all of their contribution is optimiser noise,
not sampling noise. EM reaches 27-55 *distinct* optima out of 60 replicates on
these events, and `corr(sd(lower_bound_), V_emseed) = +0.673`. That is direct
evidence of a multimodal K=7 likelihood surface, and it explains why event 39 is
flat even on a smooth surrogate.

## What this changes

Nothing in the paper's headline: the floor is real on real data, both arms agree,
and `fitvar-npe-scaling`'s conclusion stands. What it adds is that the deferred
mitigation is no longer a guess — **bagging / more EM restarts attacks a measured
21% of B and most of the variance in the top-4 events**, because that component
is optimiser variance, which averaging removes by construction. The remaining
~79% is genuine sampling variance of the density estimate on real posteriors.

Untested: whether raising `n_init` (or averaging the existing `n_boot` fits)
actually moves B. `run_emvar.py --n-init` sweeps the first lever directly.

## Files

- `summarize_published.py` — reproduces the published scan, per-event exponents
- `run_arms.py` / `analyze_arms.py` — with- vs without-replacement arms
- `run_fresh.py` / `analyze_fresh.py` — fresh-sample convergence to N=107k
- `run_emvar.py` / `analyze_emvar.py` — EM-only variance at fixed data
- `check_duplicates.py` — distinct-sample audit

All read the MDC inputs read-only from `production/`; nothing here writes to it.
