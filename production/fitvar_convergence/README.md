# Fit variance (term B): what it measures, whether it converges, and can we lower it

Companion to `production/fitvar_npe_scan/`, which established `B ~ N_PE^-0.43` on
the MDC and `-0.80` on sim_cat6. This directory asks three further questions:

1. is the floor real, or an artefact of how the scan resamples? (**real**)
2. what actually produces it? (**partly EM optimiser noise**)
3. can bagging lower it? (**~28% of the variance, but ~6% once its bias is counted**)

**The dataset is `sim_cat6_sharp_w8`. The MDC is retired** — see
`dataset-is-simcat6-not-mdc` memory. Sections marked MDC below were measured
before that was established and must not be quoted for sim_cat6; the floor has a
different severity and a different per-event driver on the two catalogs.

---

## 0. What the estimator actually samples (asked by Simon's advisor)

Three independent sampling layers, easy to conflate:

**Layer 1 — the bootstrap training draws. This is what B is a variance *over*.**
Per event, `n_boot = 50` replicates; each draws `n_train` indices **with
replacement** from that event's own PE posterior samples
(`make_diagnostic_plots.py:1598`, `rs.integers(0, nsamp, size=n_train)`). With
`--fitvar-train-frac 1.0` (the default, and what the paper run used),
`n_train = nsamp` = 5000 on sim_cat6, 3343 on the MDC.

So the count equals N_PE, but *with replacement*: each replicate holds only
`n(1-e^-1) ≈ 3161` distinct points of 5000, whereas the production density
estimate is fitted on all 5000 distinct. That is standard bootstrap construction
(resample size is meant to equal the original), but it is also why the N_PE scan
is one-sided: `train_frac` can only go **below** nsamp, and drawing n of n
*without* replacement returns the identical set every time (zero variance). There
is no without-replacement option at m = n.

**Layer 2 — the evaluation pool (the Monte-Carlo integral over θ).**
`M` draws from the defensive mixture `q* = 0.9·q_E + 0.1·q_broad`:
- `q_E` = (1/69) Σ_e p̂_e, the ensemble of per-event GMMs — 90% of M, split
  evenly across the 69 events;
- `q_broad` = a **K=3** full-covariance GMM on all events' pooled PE samples,
  covariances ×4 — 10% of M. This bounds the importance weights where the
  population model has support but individual events do not.

`log q*` is evaluated as the mixture at every point regardless of which component
produced it. Points are drawn in detector frame `(m1det, m2det, dL)`, converted
to source frame, then thinned by a validity mask (m1src∈[6,70], m2src∈[6,m1src],
z∈[0,1.9], dL≥50, finite Jacobian):
- sim_cat6 paper run: 20,000 requested → **16,408 valid** (82.0%)
- MDC: 500,000 → 445,469 (89.1%)

**Layer 3 — the Λ draws.** 200 coordinates from the **hyper-posterior** (met3 for
sim_cat6, met4 for MDC), not the prior box. B is quoted as the median over them.
B is a *function of Λ*, not a constant — hence the 16–84% bands everywhere.

**The estimator.** Per event, per Λ, per replicate b:

```
ln L_e(Λ) = log[ (1/M) Σ_j p_pop(θ_j|Λ)·J_j / q*(θ_j) · p̂_e^(b)(θ_j) ]
V_e(Λ)    = Var_b[ ln L_e(Λ) ]
B(Λ)      = Σ_e V_e(Λ)                      (make_diagnostic_plots.py:1632)
```

The sum over events is **exact, not an approximation**: training sets are
independent across events, so there is no cross-event covariance at the fit level
(`:1575`). B is therefore the variance of the *total* hierarchical log-likelihood
at fixed Λ. The selection term ξ(Λ) uses the injection set and never touches the
per-event GMMs, so it carries no fit variance — B captures all of it.

### Is the M-scan adequate? (`check_mscan.py`)

B is the intercept of `V̂(M) = B + c/M`. How much is measured vs extrapolated:

| run | M_max | c/M at largest M | at smallest scan point | B |
|---|---:|---:|---:|---:|
| MDC | 445,469 | 1.0% of B | 8.2% | 0.1945 |
| **sim_cat6 (paper)** | **16,408** | **7.9% of B** | **63.2%** | **0.0605** |
| sim_cat6 (200k rescan) | 163,222 | 1.5% of B | 11.9% | **0.0553** |

**The paper's number used a 27× smaller pool**, where the lowest scan point is
63% MC contamination and the intercept is doing real work rather than applying a
small correction. The better-resolved rescan gives 0.0553, ~1.5σ below the quoted
0.0605 ± 0.0034. Live issue for the paper, independent of everything else here.

---

## 1. The per-event K error (found by Simon)

`K_PER_EVENT_FITVAR = 7` (`:1316`) is **not the estimator the analysis uses.**
`sim_cat_inference.py` loads pre-fitted GMMs from `sim_cat6_sharp_w8/event_gmms.h5`
and takes each event's K from the loaded shape — `--kfixed 7` is recorded in the
run info but is explicitly moot in that code path ("the loaded GMM's own shape IS
the K for that event", `sim_cat_inference.py:1028-1040`).

Those K's are the **BIC-argmin** selection in `event_gmms.h5`:

```
min=3  median=5  max=10  mean=5.03      only 5 of 69 events are K=7
```

Verified against the run's recorded `K_per_event`: `k_argmin` matches at all 69
events, `k_plateau` differs at 3 — so it is argmin. `_dataset.per_event_K()` loads
them and **refuses to proceed** if the file and the run record disagree.

So the published B, its −0.80 exponent and its B_inf floor all describe a
uniform-K=7 estimator, over-parameterised on 59 of 69 events. Anything quoted
from `fitvar_npe_scan` inherits this.

---

## 2. Can bagging lower B? (`run_bagging.py`)

Replace the per-event estimate with an ensemble average of **5** GMMs, each fitted
to its own bootstrap resample:  `p̄_e = (1/5) Σ_k GMM_k`. Averaging is in
**density space** (`logsumexp(log p_k) − log K`); averaging GMM *parameters* would
be meaningless since components are not aligned across fits. Equivalently a
35-component mixture with weights/5; it stays normalised, so the IS estimator is
unchanged.

**B must be measured with a nested bootstrap** — the ensemble lives *inside* each
data draw:

```
outer b = 1..50 :  D*_b = 5000 drawn with replacement from X   <- a new "PE run"
  plain  arm   :  1 GMM fitted to D*_b                          (published estimator)
  bagged arm   :  k = 1..5 : 5000 drawn with replacement from D*_b -> GMM_k, averaged
B = Var_b[ ln L ]
```

The members resample **D\*_b, not X** — that keeps the bagged estimator a function
of its own outer draw. Reusing the existing `n_boot` fits as the ensemble (the
shortcut noted in the `fitvar-npe-scaling` memory) would average over the very
draws whose spread defines B and collapse it artifactually. The outer stream is
the *identical* RNG stream as production, so the plain arm reproduces the
published B and the arms are **paired** — the ratio is far better determined than
either B alone. Cost: 6 fits × 50 draws × 69 events = 20,700.

### Results

| | uniform K=7 | **per-event K (correct)** |
|---|---:|---:|
| plain B | 0.0576 | **0.0598** |
| bagged K=5 B | 0.0409 | **0.0433** |
| ratio bagged/plain | 0.710 | **0.724** (27.6% cut) |
| Λ-**independent** shift | −0.2157 nats | −1.0822 nats |
| Λ-**dependent** shift | 0.0239 nats | **0.1133 nats** |
| shift / sqrt(B_plain) | 0.099 | **0.463** |

Plain B at per-event K (0.0598) reproduces the published rescan (0.0553 ± 0.0022)
to ~1σ — the correctness check on the whole setup.

**The variance reduction is real and survives the K correction** (~28%), and
correcting K barely moved the baseline (+4%). But **the bias cost is 4.6× worse at
the correct K**: at median K=5 each member has less capacity, so averaging five
departs further from a single fit than it did at K=7.

Counting it properly — bagging saves 0.0598 − 0.0433 = 0.0165 of variance and
introduces bias² = 0.1133² = 0.0128:

```
plain  0.0598          bagged  0.0433 + 0.0128 = 0.0561
=> net ~6% improvement, not 28%
```

A two-point fit to `Var = a + b/K` gives a = 0.0392, i.e. a **34% ceiling** as
K→∞, with K=5 already capturing ~80% of it. Two points determine a two-parameter
form exactly, so that is an interpolation, not a tested model.

### Caveats that bound how hard this can be leaned on

- The 0.1133 nats is measured **against the current estimator, not against
  truth**. Neither fit is unbiased. It establishes that bagging *moves* the answer
  by about half the noise it removes — **not** that it moves it the wrong way.
  Settling direction needs a ground-truth density, which real events do not have.
- The Λ-**independent** −1.08 nats cancels in the hyper-posterior (a constant
  offset in ln L(Λ) does not change posterior shape) but would **not** cancel in an
  evidence comparison between population models.
- The Λ-dependent shift is a *systematic*: unlike variance it does not average
  down over events or repeated runs.
- Ensemble diversity comes **only from the resampled data** (`random_state` fixed).
  Members still reach different EM optima because k-means init is data-dependent.
  Also varying the EM seed per member would add diversity and might beat the 34%
  ceiling, which assumes members differ only through their data. **Untested.**

---

## 3. MDC sections — RETIRED, do not quote for sim_cat6

Measured before the MDC was retired. The *methodology* carries over; the numbers
do not, because the floor has a different driver on the two catalogs (89% vs 47%
N_PE-independent; high-mass driver on MDC, low-mass on sim_cat6).

Reproduces the published scan first: aggregate `B ~ N_PE^(-0.438)` vs their
`-0.429` (`summarize_published.py`).

**Bootstrap granularity — REFUTED** (`run_arms.py`). A with-replacement resample
of size m from n holds only `n(1-e^{-m/n})` distinct points (63% at m=n), which
mechanically flattens the curve and accounts for 26% of the departure from −1 on
arithmetic alone. So the scan was rerun with a **without-replacement** arm (N
genuinely distinct samples, as a shorter PE run gives), corrected by the
finite-population factor `(n-1)/(n-N)` and capped at N ≤ n/2:

| N_PE | boot | srs (FPC-corrected) | srs/boot |
|-----:|-----:|-----:|----:|
| 209 | 0.7483 | 0.7492 | **1.001** |
| 418 | 0.4795 | 0.5130 | 1.070 |
| 836 | 0.3551 | 0.4350 | 1.225 |
| 1671 | 0.2843 | 0.4373 | 1.538 |

They agree to **0.1%** at N/n = 0.06 (cross-validating both estimators) and give
−0.642 / −0.547 over 209→836. Neither is 1/N: **the bootstrap is not
manufacturing the floor.**

**Duplicated PE samples — REFUTED** (`check_duplicates.py`). All 69 events are
3343/3343 distinct.

**No floor in the estimator itself** (`run_fresh.py`). Fresh iid draws from a K=20
GMM surrogate, fitted with the production K=7, give clean 1/N out to
**N_PE = 106,976** for 5 of 6 events — `N·V` flat over a 512× range — including on
the same N window where the real data is flat. The floor is a property of the real
posteriors, not the machinery. (Caveat: a GMM surrogate has Gaussian-mixture
tails; `--inflate` probes that and was not swept.)

**EM optimiser noise — CONFIRMED, ~21%** (`run_emvar.py`). Holding the data
completely fixed and varying only the EM `random_state`:

```
emseed (data fixed, EM seed varies)  : V = 0.0414   = 21% of B(3343)=0.1945, 24% of B_inf
both   (data resampled + seed varies): V = 0.2126
```

Concentrated in the events that carry B — for events 39 and 47 the EM-only
variance *equals* the full bootstrap variance. EM reaches 27–55 distinct optima
out of 60 replicates there, and `corr(sd(lower_bound_), V_emseed) = +0.673`.

---

## Files

| file | what |
|---|---|
| `_dataset.py` | sim_cat6 vs MDC path sets; `per_event_K()`; `check_nsamp()` guard |
| `run_bagging.py` / `bagging_simcat6_K5_perevent.npz` | **the bagging test** |
| `check_mscan.py` | how much of B is 1/M extrapolation |
| `summarize_published.py` | reproduces the published scan, per-event exponents |
| `plot_B_vs_npe_simcat6_nofit.py` | sim_cat6 B-vs-N_PE, no fitted curve |
| `run_plain_scan.py` | B vs N_PE at per-event K — **written but NOT run** (declined) |
| `run_arms.py`, `run_fresh.py`, `run_emvar.py`, `run_bootbias.py` | MDC-era (§3); only `run_emvar`/`run_bootbias` are ported to `--dataset simcat6` |

All read from `production/` read-only; nothing here writes to it. Every script
takes `--dataset` (default `simcat6`) and aborts if `nsamp` is not 5000.

## Open / not done

- `run_bootbias.py` is ported to sim_cat6 but **never run** — it tests whether the
  size-n with-replacement resample is faithful to the production fit.
- EM-seed-diverse bagging (see caveats above).
- K=10 bagging, to test the `a + b/K` ceiling.
- The MDC EM-noise result (§3) has **not** been re-measured on sim_cat6.
