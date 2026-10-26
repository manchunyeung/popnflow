#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gmm_fit.py
===============

Goodness-of-fit test for the per-event GMMs used as the density estimate of the
single-event posteriors p(theta | d_i).

The test draws samples from each fitted GMM and compares them against the PE
samples the GMM was trained on, using two-sample Kolmogorov-Smirnov (KS) and
k-sample Anderson-Darling (AD) statistics. It runs both per-dimension (marginal)
comparisons and a "sliced" comparison over random projections of the joint
density, and it calibrates every statistic against a PE-vs-PE split-half
baseline so the numbers stay interpretable at large sample size.

Coordinate convention (matches production / the IS cache):
    column 0 = m1_det   column 1 = m2_det   column 2 = dL

Why a PE-vs-PE baseline (read this before trusting any p-value)
--------------------------------------------------------------
With N ~ 3000+ PE samples per event, a two-sample KS/AD test has enormous power:
it will return p ~ 0 even for a GMM that is, for any practical purpose, a perfect
fit. Raw p-values are therefore *not* a useful pass/fail criterion here. What is
meaningful is the *size* of the discrepancy (the KS statistic D, the AD A^2),
compared against the discrepancy you get just from splitting the PE samples
themselves into two halves. That split-half baseline (a) sets the scale of D/A^2
achievable under a perfect fit at this N, and (b) inherits the same MCMC
autocorrelation as the real samples, so the iid assumption being violated affects
baseline and test identically. The headline number is the EXCESS FACTOR
D(GMM vs PE) / median D(PE vs PE): ~1 means the GMM-vs-PE gap is indistinguishable
from PE sampling noise; >> 1 means a real misfit.

Mass-ordering
-------------
The production density is symmetrized over m1 <-> m2, and PE samples follow the
m1 >= m2 convention. A raw GMM fit (no symmetrization) emits draws with m1 < m2,
so we fold both sample sets to m1 >= m2 before comparing. This makes the m1/m2
marginals apples-to-apples and reflects the symmetrized density actually used.

Joint vs marginal
-----------------
KS and AD are 1-D. Per-dimension marginals can all pass while the joint
correlation structure is wrong. The sliced test projects both sample sets onto
random unit directions in PE-whitened space and runs the 1-D tests on each
projection, which is sensitive to orientation/correlation errors the marginals
miss.

Usage
-----
    # all 69 events, default sklearn GMM fit mirroring production:
    python3 test_gmm_fit.py --pe-glob "input_data/mdc_*_69rand.txt"

    # programmatic, passing your own (already-fit) GMMs:
    from test_gmm_fit import compare_event
    res = compare_event(pe_samples, gmm=my_gmm, sample_fn=my_sampler)

    uv run python test_gmm_fit.py \
    --json-glob "input_data/home/**/bbh_result.json" \
    --k-csv k_selection_out/per_event_best_K.csv \
    --outfig figures/mdc_gmm_fit_m1.pdf \
    --use-tex --fontsize 20


"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# Production GMM recipe (per-event). Kept here so the test reflects the fit the
# pipeline actually uses. If sklearn is unavailable in your env, pass your own
# fit_fn / sample_fn (e.g. the in-house EM) into the API instead.
# ----------------------------------------------------------------------------
GMM_K = 7
GMM_REG_COVAR = 1e-6
GMM_N_INIT = 5
GMM_SEED = 0


def default_fit_fn(samples: np.ndarray, seed: int = GMM_SEED):
    """Fit a per-event GMM exactly as in production (sklearn)."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=GMM_K,
        covariance_type="full",
        reg_covar=GMM_REG_COVAR,
        n_init=GMM_N_INIT,
        random_state=seed,
    )
    gmm.fit(samples)
    return gmm


def make_k_fit_fn(k: int) -> Callable:
    """Return a fit_fn that uses k components instead of the global default."""
    def _fit(samples: np.ndarray, seed: int = GMM_SEED):
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=GMM_REG_COVAR,
            n_init=GMM_N_INIT,
            random_state=seed,
        )
        gmm.fit(samples)
        return gmm
    return _fit


def default_sample_fn(gmm, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from a fitted GMM.

    Works for an sklearn GaussianMixture. For a custom GMM object, pass your own
    sample_fn(gmm, n, rng) -> (n, D) array.
    """
    # sklearn's .sample uses its own RNG state; reseed for reproducibility.
    try:
        gmm.random_state = int(rng.integers(0, 2**31 - 1))
    except Exception:
        pass
    X, _ = gmm.sample(n)
    return np.asarray(X, dtype=float)


# ----------------------------------------------------------------------------
# IO  (robust auto-detecting loader)
# ----------------------------------------------------------------------------
from collections import Counter

_DELIM_CANDIDATES = [None, ",", "\t", ";", "|"]   # None == any whitespace


def _split(line: str, delim):
    return line.split() if delim is None else [c.strip() for c in line.split(delim)]


def _floatable(tokens) -> bool:
    try:
        for t in tokens:
            float(t)
        return True
    except (ValueError, TypeError):
        return False


def _detect_delimiter(lines: Sequence[str]):
    """Pick the delimiter giving the most consistent (>1) column count."""
    best, best_score = None, -1
    for d in _DELIM_CANDIDATES:
        counts = [len(_split(l, d)) for l in lines if l.strip()]
        counts = [c for c in counts if c > 0]
        if not counts:
            continue
        common, n = Counter(counts).most_common(1)[0]
        score = (common > 1) * n * common      # reward many rows with same >1 width
        if score > best_score:
            best, best_score = d, score
    return best


def _read_table(path: str):
    """Return (data[N,C] float array, header list|None, delimiter).

    Auto-detects the delimiter; recognizes a header either as a leading
    '#'-commented line (np.savetxt convention) or as a non-numeric first row.
    """
    with open(path) as f:
        raw = [ln.rstrip("\n") for ln in f]

    last_comment, data_lines = None, []
    for ln in raw:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            last_comment = s.lstrip("#").strip()
        else:
            data_lines.append(ln)
    if not data_lines:
        raise ValueError(f"{path}: no data rows found")

    delim = _detect_delimiter(data_lines[:50])
    ncol = len(_split(data_lines[0], delim))

    header = None
    if last_comment is not None:                     # header from '# ...' line
        ht = _split(last_comment, delim)
        if len(ht) == ncol:
            header = ht
    if header is None and not _floatable(_split(data_lines[0], delim)):
        header = _split(data_lines[0], delim)        # header as first data row
        data_lines = data_lines[1:]

    rows, skipped = [], 0
    for ln in data_lines:
        toks = _split(ln, delim)
        if len(toks) != ncol or not _floatable(toks):
            skipped += 1
            continue
        rows.append([float(t) for t in toks])
    if not rows:
        raise ValueError(
            f"{path}: could not parse any numeric rows "
            f"(detected delimiter={delim!r}, {ncol} columns). "
            f"Run with --inspect to see the file."
        )
    if skipped > 0.1 * len(data_lines):
        warnings.warn(f"{path}: skipped {skipped} ragged/non-numeric rows")
    return np.asarray(rows, float), header, delim


def inspect_pe_file(path: str, n: int = 4) -> None:
    """Print how the loader sees a file: first lines, delimiter, columns, header."""
    with open(path) as f:
        head = [next(f, "").rstrip("\n") for _ in range(n)]
    print(f"=== {path} ===")
    print("first lines (raw):")
    for ln in head:
        print("   ", repr(ln))
    try:
        data, header, delim = _read_table(path)
        dname = {None: "whitespace", ",": "comma", "\t": "tab",
                 ";": "semicolon", "|": "pipe"}.get(delim, repr(delim))
        print(f"detected delimiter : {dname}")
        print(f"parsed shape       : {data.shape}")
        print(f"header             : {header}")
        if header is not None:
            print("column index -> name:")
            for i, nm in enumerate(header):
                print(f"   [{i}] {nm}")
        print("first parsed row   :", data[0])
    except Exception as e:  # noqa: BLE001
        print("PARSE FAILED:", e)


def load_pe_samples(
    path: str,
    usecols: Sequence[int] = (0, 1, 2),
    names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Load PE samples as an (N, 3) array of (m1_det, m2_det, dL).

    Auto-detects delimiter and header. Select the three columns either by
    position (`usecols`, default 0,1,2) or, if the file has a header, by name
    (`names=("m1_det","m2_det","dL")`). Run `inspect_pe_file(path)` if unsure.
    """
    data, header, _ = _read_table(path)

    if names is not None:
        if header is None:
            raise ValueError(
                f"{path}: names= requested but no header detected. "
                f"Use usecols= with positional indices instead."
            )
        idx = []
        for nm in names:
            if nm not in header:
                raise ValueError(
                    f"{path}: column {nm!r} not in header {header}. "
                    f"Use usecols= or fix the names."
                )
            idx.append(header.index(nm))
    else:
        idx = list(usecols)

    if max(idx) >= data.shape[1]:
        raise ValueError(
            f"{path}: requested columns {idx} but file has only "
            f"{data.shape[1]} columns. Run inspect_pe_file({path!r})."
        )
    arr = data[:, idx]
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def load_bilby_json(path: str,
                    m1_col: str = "mass_1",
                    m2_col: str = "mass_2",
                    dl_col: str = "luminosity_distance") -> np.ndarray:
    """Load (N, 3) array of (m1_det, m2_det, dL) from a bilby bbh_result.json.

    The JSON stores detector-frame mass_1 / mass_2 and luminosity_distance
    directly in posterior['content'].  Returns an (N_samples, 3) float array.
    """
    import json
    with open(path) as fh:
        d = json.load(fh)
    content = d["posterior"]["content"]
    arr = np.column_stack([
        np.asarray(content[m1_col], float),
        np.asarray(content[m2_col], float),
        np.asarray(content[dl_col], float),
    ])
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def load_events_from_param_files(
    m1_path: str,
    m2_path: str,
    dL_path: str,
    *,
    n_events: Optional[int] = 69,
    n_per_event=None,
    layout: str = "event_major",
) -> list[np.ndarray]:
    """Load PE samples stored one-parameter-per-file (flat single columns) and
    split them back into per-event (N_samp, 3) arrays of (m1_det, m2_det, dL).

    Each file is read as a flat column (m1rand, m2rand, dLrand). The three are
    stacked, then partitioned into events.

    Event partitioning (set ONE of n_per_event / n_events; the other is inferred):
        n_per_event : int                -> equal blocks of this many samples
        n_per_event : sequence of ints   -> explicit per-event counts (unequal)
        n_events    : int (default 69)   -> equal blocks of N_total // n_events

    layout : "event_major"  -> file is [event0 samples..., event1 samples..., ...]
             "sample_major" -> file is [sample0 of all events, sample1 ..., ...]
                               (i.e. the flat column came from a (N_samp, N_evt)
                                array raveled in C order)
    """
    cols = [_read_table(p)[0].ravel() for p in (m1_path, m2_path, dL_path)]
    lens = [len(c) for c in cols]
    if len(set(lens)) != 1:
        raise ValueError(
            f"parameter files have unequal length: "
            f"m1={lens[0]}, m2={lens[1]}, dL={lens[2]}"
        )
    n_total = lens[0]
    X = np.column_stack(cols)  # (N_total, 3)

    # explicit unequal counts
    if n_per_event is not None and not np.isscalar(n_per_event):
        counts = [int(c) for c in n_per_event]
        if sum(counts) != n_total:
            raise ValueError(f"counts sum to {sum(counts)}, file has {n_total} rows")
        events, s = [], 0
        for c in counts:
            events.append(X[s : s + c])
            s += c
    else:
        # equal blocks
        if n_per_event is None:
            if n_events is None:
                raise ValueError("provide n_events or n_per_event")
            if n_total % n_events:
                raise ValueError(
                    f"{n_total} samples not divisible by n_events={n_events}: "
                    f"samples per event are unequal. Pass n_per_event as a list "
                    f"of counts (or an index file) instead."
                )
            n_per_event = n_total // n_events
        else:
            n_per_event = int(n_per_event)
        n_events = n_total // n_per_event
        if layout == "event_major":
            ev = X.reshape(n_events, n_per_event, 3)
        elif layout == "sample_major":
            ev = X.reshape(n_per_event, n_events, 3).transpose(1, 0, 2)
        else:
            raise ValueError("layout must be 'event_major' or 'sample_major'")
        events = [ev[i] for i in range(n_events)]

    out = []
    for e in events:
        finite = np.isfinite(e).all(axis=1)
        out.append(np.ascontiguousarray(e[finite]))
    return out


# ----------------------------------------------------------------------------
# Core statistics
# ----------------------------------------------------------------------------
def fold_mass_order(X: np.ndarray, m1_col: int = 0, m2_col: int = 1) -> np.ndarray:
    """Return a copy with rows reordered so column m1_col >= column m2_col."""
    Y = np.array(X, dtype=float, copy=True)
    swap = Y[:, m1_col] < Y[:, m2_col]
    Y[swap, m1_col], Y[swap, m2_col] = Y[swap, m2_col], Y[swap, m1_col]
    return Y


def _ad_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Anderson-Darling. Returns (normalized A^2 statistic, p-value).

    scipy caps the analytic significance level to [0.001, 0.25] and warns when
    the statistic is outside the interpolation table; we suppress that and just
    report the (possibly capped) value alongside the statistic, which is the
    quantity that actually matters here.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.anderson_ksamp([a, b])
        p = float(getattr(res, "pvalue", res.significance_level))
        return float(res.statistic), p
    except (ValueError, ZeroDivisionError):
        return np.nan, np.nan


def ks_ad_1d(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sample KS and AD on 1-D arrays a, b."""
    ks = stats.ks_2samp(a, b, method="auto")
    ad_stat, ad_p = _ad_2samp(a, b)
    return {
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "ad_A2": ad_stat,
        "ad_p": ad_p,
    }


def _whitener(ref: np.ndarray):
    """Return (mean, W) so that (X - mean) @ W has ~identity covariance in ref."""
    mu = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False)
    cov = np.atleast_2d(cov)
    # Jitter for numerical safety on near-degenerate covariances.
    cov = cov + 1e-12 * np.eye(cov.shape[0])
    L = np.linalg.cholesky(cov)
    W = np.linalg.inv(L).T
    return mu, W


def sliced_tests(
    A: np.ndarray,
    B: np.ndarray,
    n_slices: int,
    rng: np.random.Generator,
    whiten_ref: Optional[np.ndarray] = None,
) -> dict:
    """1-D KS/AD over n_slices random projections in (optionally) whitened space.

    whiten_ref defines the whitening transform (use the PE samples) so the random
    directions are isotropic in a sensible metric. Returns median and worst-case
    summaries across directions.
    """
    D = A.shape[1]
    if whiten_ref is not None and D > 1:
        mu, W = _whitener(whiten_ref)
        Aw = (A - mu) @ W
        Bw = (B - mu) @ W
    else:
        Aw, Bw = A, B

    dirs = rng.standard_normal((n_slices, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    ks_Ds, ad_A2s = [], []
    for d in dirs:
        pa = Aw @ d
        pb = Bw @ d
        r = ks_ad_1d(pa, pb)
        ks_Ds.append(r["ks_D"])
        ad_A2s.append(r["ad_A2"])
    ks_Ds = np.asarray(ks_Ds)
    ad_A2s = np.asarray(ad_A2s)
    return {
        "ks_D_median": float(np.nanmedian(ks_Ds)),
        "ks_D_max": float(np.nanmax(ks_Ds)),
        "ad_A2_median": float(np.nanmedian(ad_A2s)),
        "ad_A2_max": float(np.nanmax(ad_A2s)),
    }


def pe_split_baseline(
    pe: np.ndarray,
    n_slices: int,
    rng: np.random.Generator,
    n_repeats: int = 8,
    cols: Sequence[int] = (0, 1, 2),
) -> dict:
    """Split-half PE-vs-PE baseline: the discrepancy expected under a perfect fit
    at this N and autocorrelation. Returns median per-dim and sliced KS D."""
    n = pe.shape[0]
    half = n // 2
    ks_per_dim  = {c: [] for c in cols}
    ad_per_dim  = {c: [] for c in cols}
    ks_sliced, ad_sliced = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        a, b = pe[perm[:half]], pe[perm[half : 2 * half]]
        for c in cols:
            r = ks_ad_1d(a[:, c], b[:, c])
            ks_per_dim[c].append(r["ks_D"])
            ad_per_dim[c].append(r["ad_A2"])
        sl = sliced_tests(a, b, n_slices, rng, whiten_ref=pe)
        ks_sliced.append(sl["ks_D_median"])
        ad_sliced.append(sl["ad_A2_median"])
    return {
        "ks_D_per_dim":  {c: float(np.median(v)) for c, v in ks_per_dim.items()},
        "ks_D_sliced":   float(np.median(ks_sliced)),
        "ad_A2_per_dim": {c: float(np.median(v)) for c, v in ad_per_dim.items()},
        "ad_A2_sliced":  float(np.median(ad_sliced)),
    }


# ----------------------------------------------------------------------------
# Per-event comparison
# ----------------------------------------------------------------------------
DIM_NAMES = {0: "m1_det", 1: "m2_det", 2: "dL"}


@dataclass
class EventResult:
    name: str
    n_pe: int
    n_train: int = 0          # samples used to fit the GMM
    n_test: int = 0           # samples in the held-out test set
    k_components: int = 0     # number of GMM components used
    per_dim: dict = field(default_factory=dict)   # col -> ks/ad dict
    sliced: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    excess: dict = field(default_factory=dict)    # col/"sliced" -> D_gmm / D_base
    flagged: bool = False


def compare_event(
    pe_samples: np.ndarray,
    gmm=None,
    *,
    sample_fn: Callable = default_sample_fn,
    fit_fn: Callable = default_fit_fn,
    name: str = "event",
    cols: Sequence[int] = (0, 1, 2),
    m1_col: int = 0,
    m2_col: int = 1,
    n_gmm: Optional[int] = None,
    n_slices: int = 64,
    baseline_repeats: int = 8,
    excess_threshold: float = 3.0,
    rng: Optional[np.random.Generator] = None,
    train_frac: float = 0.8,
) -> EventResult:
    """Compare GMM draws against the PE samples for a single event.

    When train_frac < 1.0 (default 0.8), the PE samples are split: the GMM is
    fit on the train fraction and evaluated on the held-out test fraction.  The
    baseline (PE-vs-PE split-half) is also computed on the test set so that both
    the test statistic and the baseline operate at the same sample size.

    If `gmm` is None, one is fit from the training samples via `fit_fn`.
    Pass a pre-fit GMM (and matching `sample_fn`) to skip fitting entirely.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pe = np.asarray(pe_samples, dtype=float)

    # ── 80/20 train/test split ────────────────────────────────────────────────
    if train_frac < 1.0 and gmm is None:
        n_total = pe.shape[0]
        n_train = max(1, int(n_total * train_frac))
        perm = rng.permutation(n_total)
        pe_train = pe[perm[:n_train]]
        pe_eval  = pe[perm[n_train:]]
    else:
        pe_train = pe
        pe_eval  = pe

    if gmm is None:
        gmm = fit_fn(pe_train)

    # Infer K from the fitted GMM if possible (sklearn stores n_components).
    k_used = int(getattr(gmm, "n_components", 0))

    # Match GMM draw count to the evaluation set size.
    n_draw = n_gmm if n_gmm is not None else pe_eval.shape[0]
    gmm_samp = np.asarray(sample_fn(gmm, n_draw, rng), dtype=float)

    # Fold both sets to m1 >= m2 (symmetrized density / PE convention).
    pe_eval_f = fold_mass_order(pe_eval, m1_col, m2_col)
    gmm_f     = fold_mass_order(gmm_samp, m1_col, m2_col)

    res = EventResult(
        name=name,
        n_pe=pe.shape[0],
        n_train=pe_train.shape[0],
        n_test=pe_eval.shape[0],
        k_components=k_used,
    )

    # Per-dimension marginals (on held-out test set)
    for c in cols:
        res.per_dim[c] = ks_ad_1d(pe_eval_f[:, c], gmm_f[:, c])

    # Sliced (joint) test in test-set-whitened space
    res.sliced = sliced_tests(gmm_f, pe_eval_f, n_slices, rng, whiten_ref=pe_eval_f)

    # Baseline: PE-vs-PE split-half computed on the *test set* so the baseline
    # and the GMM test operate at the same effective sample size.
    res.baseline = pe_split_baseline(pe_eval_f, n_slices, rng, baseline_repeats, cols)

    # Excess factors for KS; raw statistic + p-value for AD
    # (AD is not bounded below so the ratio approach breaks: GMM can fit better
    #  than the PE split-half, driving the ratio negative. Report A2 and p directly.)
    eps = 1e-12
    for c in cols:
        ks_base = res.baseline["ks_D_per_dim"][c]
        res.excess[f"ks_{c}"]  = res.per_dim[c]["ks_D"]  / (ks_base + eps)
        res.excess[f"ad_{c}"]  = res.per_dim[c]["ad_A2"]   # raw A2
        res.excess[f"adp_{c}"] = res.per_dim[c]["ad_p"]    # p-value
    res.excess["ks_sliced"] = res.sliced["ks_D_median"]  / (res.baseline["ks_D_sliced"] + eps)
    res.excess["ad_sliced"]  = res.sliced["ad_A2_median"]  # raw A2
    res.excess["adp_sliced"] = res.sliced["ad_A2_median"]  # placeholder; p not stored per-slice

    # Flag on KS excess only (AD p-value flagging handled in print_summary)
    res.flagged = (
        max(res.excess[f"ks_{c}"] for c in cols) > excess_threshold
        or res.excess["ks_sliced"] > excess_threshold
    )
    return res


# ----------------------------------------------------------------------------
# Catalog-level runner
# ----------------------------------------------------------------------------
def run_events(
    pe_events: Sequence[np.ndarray],
    *,
    labels: Optional[Sequence[str]] = None,
    n_slices: int = 64,
    baseline_repeats: int = 8,
    excess_threshold: float = 3.0,
    seed: int = 0,
    fit_fn: Callable = default_fit_fn,
    fit_fns: Optional[Sequence[Callable]] = None,
    sample_fn: Callable = default_sample_fn,
    gmms: Optional[Sequence] = None,
    train_frac: float = 0.8,
) -> list[EventResult]:
    """Run the test over a list of in-memory per-event (N_samp, 3) arrays.

    fit_fns  : per-event list of fit functions (overrides fit_fn when provided).
               Useful for applying a different K per event from a CSV.
    train_frac: fraction of PE samples used for GMM training; the rest are
               held out for testing (default 0.8).
    """
    rng = np.random.default_rng(seed)
    cols = tuple(range(pe_events[0].shape[1]))
    results: list[EventResult] = []
    for i, pe in enumerate(pe_events):
        gmm    = gmms[i]    if gmms    is not None else None
        fn     = fit_fns[i] if fit_fns is not None else fit_fn
        r = compare_event(
            pe,
            gmm=gmm,
            sample_fn=sample_fn,
            fit_fn=fn,
            name=labels[i] if labels is not None else f"event_{i:03d}",
            cols=cols,
            n_slices=n_slices,
            baseline_repeats=baseline_repeats,
            excess_threshold=excess_threshold,
            rng=rng,
            train_frac=train_frac,
        )
        results.append(r)
    return results


def run_all(
    pe_paths: Sequence[str],
    *,
    usecols: Sequence[int] = (0, 1, 2),
    names: Optional[Sequence[str]] = None,
    n_slices: int = 64,
    baseline_repeats: int = 8,
    excess_threshold: float = 3.0,
    seed: int = 0,
    fit_fn: Callable = default_fit_fn,
    sample_fn: Callable = default_sample_fn,
    gmms: Optional[Sequence] = None,
    train_frac: float = 0.8,
) -> list[EventResult]:
    """Run the test over a list of PE sample files (one per event, columnar)."""
    events = [load_pe_samples(p, usecols=usecols, names=names) for p in pe_paths]
    labels = [os.path.basename(p) for p in pe_paths]
    return run_events(
        events,
        labels=labels,
        n_slices=n_slices,
        baseline_repeats=baseline_repeats,
        excess_threshold=excess_threshold,
        seed=seed,
        fit_fn=fit_fn,
        sample_fn=sample_fn,
        gmms=gmms,
        train_frac=train_frac,
    )


def print_summary(results: Sequence[EventResult], cols: Sequence[int] = (0, 1, 2)):
    """Catalog summary: KS excess factors (row 1) + raw AD A² with p-values (row 2)."""
    dnames = [DIM_NAMES.get(c, f"x{c}") for c in cols]
    cw, nw = 10, 28
    AD_P_THRESH = 0.001

    # Show train/test counts if split was used
    show_split = any(r.n_train > 0 and r.n_train != r.n_pe for r in results)
    show_k     = any(r.k_components > 0 for r in results)
    extra_hdr  = ""
    if show_k:     extra_hdr += f" {'K':>3}"
    if show_split: extra_hdr += f"  {'n_train':>7} {'n_test':>6}"

    dim_hdr = " ".join(f"{n:>{cw}}" for n in dnames)
    print(f"{'event':<{nw}} {'N_pe':>6}{extra_hdr} {'stat':>6}  {dim_hdr}  {'sliced':>{cw}}  flag")
    sep = "-" * (nw + 8 + 7 + len(extra_hdr) + (cw + 1) * (len(cols) + 1) + 7)
    print(sep)

    n_ks_flag, n_ad_flag = 0, 0
    worst_ks, worst_ad = [], []

    for r in results:
        flag_str = "  *  " if r.flagged else "     "
        extra_val = ""
        if show_k:     extra_val += f" {r.k_components:>3}"
        if show_split: extra_val += f"  {r.n_train:>7} {r.n_test:>6}"

        # Row 1: KS excess (ratio to PE split-half baseline)
        ks_row = f"{r.name:<{nw}} {r.n_pe:>6}{extra_val} {'KS exc':>6}  "
        ks_row += " ".join(f"{r.excess[f'ks_{c}']:>{cw}.2f}" for c in cols)
        ks_row += f"  {r.excess['ks_sliced']:>{cw}.2f}  {flag_str}"
        print(ks_row)

        # Row 2: AD A² raw statistic (negative = GMM fits better than PE split-half)
        ad_flagged = any(r.excess[f"adp_{c}"] <= AD_P_THRESH for c in cols)
        ad_row = f"{'':<{nw}} {'':<6}{' ' * len(extra_val)} {'AD A²':>6}  "
        ad_row += " ".join(f"{r.excess[f'ad_{c}']:>{cw}.1f}" for c in cols)
        ad_row += f"  {r.excess['ad_sliced']:>{cw}.1f}"
        ad_row += f"  {'[p<.001]' if ad_flagged else ''}"
        print(ad_row)

        n_ks_flag  += int(r.flagged)
        n_ad_flag  += int(ad_flagged)
        worst_ks.append((max(r.excess[f"ks_{c}"] for c in cols), r.name))
        worst_ad.append((max(r.excess[f"ad_{c}"] for c in cols), r.name))

    print(sep)
    print(f"KS: {n_ks_flag}/{len(results)} flagged (excess > threshold).")
    print(f"AD: {n_ad_flag}/{len(results)} with p < {AD_P_THRESH} in ≥1 dimension.")
    worst_ks.sort(reverse=True); worst_ad.sort(reverse=True)
    print("Worst KS excess:")
    for ex, nm in worst_ks[:5]:
        print(f"  {nm:<{nw}} {ex:6.2f}")
    print("Worst AD A² (raw; larger = worse; negative = GMM beats split-half):")
    for ex, nm in worst_ad[:5]:
        print(f"  {nm:<{nw}} {ex:8.1f}")
    print(
        "\nKS excess ~ 1: GMM-vs-PE gap within PE sampling noise (good).\n"
        "AD A² < 0: GMM fits better than a PE half-split (excellent).\n"
        "AD A² large and p < 0.001: statistically significant misfit.\n"
        "KS + AD both flagging the same event = strong evidence of real misfit."
    )


# ----------------------------------------------------------------------------
# Catalog-level histogram figure
# ----------------------------------------------------------------------------
DIM_LABELS_TEX = {0: r"$m_1^{\rm det}$", 1: r"$m_2^{\rm det}$", 2: r"$d_L$"}


def make_summary_histograms(
    results: Sequence[EventResult],
    cols: Sequence[int] = (0, 1, 2),
    out_path: str = "gmm_fit_histograms.pdf",
    use_tex: bool = False,
    math_labels: bool | None = None,
    fontsize: float = 20,
    include_sliced: bool = True,
):
    """Two-row histogram grid: raw KS D_GMM (top) and AD A² (bottom) for all events.

    Columns are the per-dimension marginals, plus the sliced (joint) test
    unless `include_sliced=False`. The KS row plots the raw two-sample D
    statistic (GMM vs. held-out PE), not divided by the PE split-half
    baseline. AD A² < 0 means the GMM fits better than a PE split-half.

    Styling follows the shared convention used by the other paper diagnostic
    scripts (see `setup_style()` in pp_bias_diagnostics.py / run_all_diagnostics.py):
    Agg backend, optional LaTeX + Computer Modern serif fonts, seaborn "talk"
    context with the colorblind palette, and a single `fontsize` knob for
    axis labels / titles / legends.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["text.usetex"] = use_tex
    if use_tex:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
    sns.set_context("talk", font_scale=1.3)
    sns.set_palette("colorblind")
    matplotlib.rcParams["axes.labelsize"] = fontsize
    matplotlib.rcParams["axes.titlesize"] = fontsize
    matplotlib.rcParams["legend.fontsize"] = fontsize * 0.8
    matplotlib.rcParams["xtick.labelsize"] = fontsize * 0.7
    matplotlib.rcParams["ytick.labelsize"] = fontsize * 0.7

    keys = list(cols)
    if include_sliced:
        keys = keys + ["sliced"]
    n_panels = len(keys)

    panel_width = max(6.5, 4.2 * n_panels)
    fig, axes = plt.subplots(2, n_panels, figsize=(panel_width, 7),
                              sharey="row", squeeze=False)

    # collect data
    ks_data  = {c: [r.per_dim[c]["ks_D"]  for r in results] for c in cols}
    ad_data  = {c: [r.excess[f"ad_{c}"]   for r in results] for c in cols}
    if include_sliced:
        ks_data["sliced"] = [r.sliced["ks_D_median"] for r in results]
        ad_data["sliced"] = [r.excess["ad_sliced"] for r in results]

    # `math_labels` decouples "typeset the labels as math" from "use real LaTeX".
    # Needed because text.usetex silently drops minus signs on this host (the AD
    # A^2 axis is negative for most events), so the CM look has to come from
    # mathtext instead -- but mathtext still wants the $...$ form, and the plain
    # "D_GMM" fallback renders as "D<junk>GMM" in cmr10, which has no underscore.
    if math_labels is None:
        math_labels = use_tex
    ks_xlabel = r"$D_{\rm GMM}$" if math_labels else "D_GMM"
    ad_xlabel = r"$A^2$ "

    for col_idx, key in enumerate(keys):
        ax_ks = axes[0, col_idx]
        ax_ad = axes[1, col_idx]

        ks_vals = np.array(ks_data[key])
        ad_vals = np.array(ad_data[key])

        # KS histogram (raw D_GMM, not divided by the PE split-half baseline)
        ax_ks.hist(ks_vals, bins=15, color="steelblue", edgecolor="white", linewidth=0.5)
        ax_ks.set_xlabel(ks_xlabel, fontsize=fontsize * 0.8)
        if col_idx == 0:
            ax_ks.set_ylabel("events", fontsize=fontsize * 0.85)

        # AD A² histogram
        ax_ad.hist(ad_vals, bins=15, color="darkorange", edgecolor="white", linewidth=0.5)
        ax_ad.axvline(0.0, color="k", ls="--", lw=1.5, label="baseline")
        ax_ad.set_xlabel(ad_xlabel, fontsize=fontsize * 0.8)
        if col_idx == 0:
            ax_ad.set_ylabel("events", fontsize=fontsize * 0.85)
        ax_ad.legend(fontsize=fontsize * 0.6, frameon=False)

    # row2_label = r"AD $A^2$"
    # axes[1, 0].annotate(roxy=(0, 0.5), xycoords="axes fraction",
    #                      xytext=(-0.45, 0.5), textcoords="axes fraction",
    #                      fontsize=fontsiw2_label, ze * 0.85, rotation=90, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[histogram] saved -> {out_path}")


# ----------------------------------------------------------------------------
# Optional figure
# ----------------------------------------------------------------------------
def make_event_figure(result: EventResult, pe: np.ndarray, gmm_samp: np.ndarray,
                      cols: Sequence[int], out_path: str):
    """ECDF overlays of PE vs GMM per dimension for one event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pe_f = fold_mass_order(pe)
    gmm_f = fold_mass_order(gmm_samp)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.4))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        for data, lab, ls in [(pe_f[:, c], "PE", "-"), (gmm_f[:, c], "GMM", "--")]:
            xs = np.sort(data)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, ls, label=lab)
        ax.set_title(f"{DIM_NAMES.get(c, c)}  (D/base={result.excess[f'ks_{c}']:.2f})")
        ax.set_ylabel("CDF")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(result.name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Mode A: one-parameter-per-file (m1det_69rand / m2det_69rand / dLrand)  -- your layout
    ap.add_argument("--m1-file", help="flat file of m1det samples (all events), e.g. mdc_m1det_69rand.txt")
    ap.add_argument("--m2-file", help="flat file of m2det samples (all events), e.g. mdc_m2det_69rand.txt")
    ap.add_argument("--dl-file", help="flat file of dL samples (all events)")
    ap.add_argument("--n-events", type=int, default=69,
                    help="number of events to split the flat columns into")
    ap.add_argument("--n-per-event", type=int, default=None,
                    help="samples per event (overrides n-events if set)")
    ap.add_argument("--layout", default="event_major",
                    choices=["event_major", "sample_major"],
                    help="how events are ordered within each flat column")
    # Mode B: one-file-per-event, columnar
    ap.add_argument("--pe-glob", default=None,
                    help="glob for per-event columnar files (alternative to --m*-file)")
    ap.add_argument("--usecols", default="0,1,2",
                    help="column indices for (m1_det,m2_det,dL) in columnar mode")
    # Mode C: bilby bbh_result.json files (one per event)
    ap.add_argument("--json-glob", default=None,
                    help="glob for per-event bilby bbh_result.json files, e.g. "
                         "'input_data/home/**/bbh_result.json'. "
                         "Extracts mass_1, mass_2, luminosity_distance from "
                         "posterior['content'].")
    # Shared
    ap.add_argument("--inspect", action="store_true",
                    help="print how the loader parses the input files, then exit")
    ap.add_argument("--n-slices", type=int, default=64)
    ap.add_argument("--baseline-repeats", type=int, default=8)
    ap.add_argument("--excess-threshold", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of PE samples used to train the GMM; the "
                         "rest are held out for testing (default: 0.8).")
    ap.add_argument("--k-csv", default=None,
                    help="CSV with per-event best K (e.g. k_selection_out/per_event_best_K.csv). "
                         "Must have columns 'event' and the column named by --k-col.")
    ap.add_argument("--k-col", default="k_argmin",
                    help="Column in --k-csv to use as the number of GMM components "
                         "[k_argmin].")
    ap.add_argument("--outfig", default=None,
                    help="Save a histogram summary figure to this path (e.g. gmm_fit.pdf). "
                         "If omitted, no figure is produced.")
    ap.add_argument("--use-tex", action="store_true",
                    help="Use LaTeX rendering for figure labels (matches the "
                         "styling convention in pp_bias_diagnostics.py / "
                         "run_all_diagnostics.py).")
    ap.add_argument("--fontsize", type=float, default=20,
                    help="Axis-label / title font size for --outfig [20].")
    ap.add_argument("--results-cache", type=Path, default=None,
                    help="Path to cache the computed per-event results (pickle). "
                         "If it exists, results are loaded from it instead of "
                         "re-running GMM fitting + KS/AD tests. If it doesn't "
                         "exist, results are computed normally and saved here "
                         "for next time.")
    args = ap.parse_args()

    param_mode = bool(args.m1_file or args.m2_file or args.dl_file)
    json_mode  = bool(args.json_glob)

    # ── load per-event best K from CSV ────────────────────────────────────────
    k_map: dict[str, int] = {}
    if args.k_csv:
        import csv
        with open(args.k_csv) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                k_map[row["event"]] = int(float(row[args.k_col]))
        print(f"[K] Loaded best-K for {len(k_map)} events from {args.k_csv} "
              f"(column '{args.k_col}'). "
              f"K range: {min(k_map.values())}–{max(k_map.values())}.")

    if args.inspect:
        targets = ([args.m1_file, args.m2_file, args.dl_file] if param_mode
                   else sorted(glob.glob(args.pe_glob or ""))[:1])
        for t in targets:
            if t:
                inspect_pe_file(t)
                print()
        return

    results = None
    from_cache = False
    if args.results_cache is not None and args.results_cache.exists():
        print(f"[cache] loading cached results <- {args.results_cache}")
        with open(args.results_cache, "rb") as fh:
            results = pickle.load(fh)
        from_cache = True
        print(f"[cache] loaded {len(results)} event results from cache.")

    if from_cache:
        pass
    elif json_mode:
        json_paths = sorted(glob.glob(args.json_glob, recursive=True))
        if not json_paths:
            raise SystemExit(f"No JSON files match {args.json_glob!r}")
        print(f"Found {len(json_paths)} bilby result JSON files.\n")
        events, labels = [], []
        for p in json_paths:
            try:
                arr = load_bilby_json(p)
                events.append(arr)
                labels.append(os.path.basename(os.path.dirname(p)))
            except Exception as exc:
                warnings.warn(f"Skipping {p}: {exc}")
        print(f"Loaded {len(events)} events; "
              f"{events[0].shape[0]} samples in event 0.\n")
        fit_fns_list = ([make_k_fit_fn(k_map[lb]) if lb in k_map else default_fit_fn
                         for lb in labels]
                        if k_map else None)
        results = run_events(
            events,
            labels=labels,
            n_slices=args.n_slices,
            baseline_repeats=args.baseline_repeats,
            excess_threshold=args.excess_threshold,
            seed=args.seed,
            fit_fns=fit_fns_list,
            train_frac=args.train_frac,
        )

    elif param_mode:
        if not (args.m1_file and args.m2_file and args.dl_file):
            raise SystemExit("param mode needs --m1-file, --m2-file, and --dl-file")
        events = load_events_from_param_files(
            args.m1_file, args.m2_file, args.dl_file,
            n_events=args.n_events, n_per_event=args.n_per_event,
            layout=args.layout,
        )
        print(f"Loaded {len(events)} events; "
              f"{events[0].shape[0]} samples in event 0 "
              f"(layout={args.layout}).\n")
        results = run_events(
            events,
            n_slices=args.n_slices,
            baseline_repeats=args.baseline_repeats,
            excess_threshold=args.excess_threshold,
            seed=args.seed,
            train_frac=args.train_frac,
        )

    else:
        pe_glob = args.pe_glob or "input_data/mdc_*_69rand.txt"
        pe_paths = sorted(glob.glob(pe_glob))
        if not pe_paths:
            raise SystemExit(f"No PE files match {pe_glob!r}")
        print(f"Found {len(pe_paths)} PE files.\n")
        usecols = tuple(int(x) for x in args.usecols.split(","))
        results = run_all(
            pe_paths,
            usecols=usecols,
            n_slices=args.n_slices,
            baseline_repeats=args.baseline_repeats,
            excess_threshold=args.excess_threshold,
            seed=args.seed,
            train_frac=args.train_frac,
        )

    if not from_cache and args.results_cache is not None:
        args.results_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_cache, "wb") as fh:
            pickle.dump(results, fh)
        print(f"[cache] saved {len(results)} event results -> {args.results_cache}")

    # m1_det, m2_det, and dL each run the same GMM-vs-PE goodness-of-fit test and
    # tell the same story, so a single representative dimension is enough here.
    plot_cols = (0,)
    print_summary(results, cols=plot_cols)
    if args.outfig:
        make_summary_histograms(
            results,
            cols=plot_cols,
            out_path=args.outfig,
            use_tex=args.use_tex,
            fontsize=args.fontsize,
            include_sliced=False,
        )


if __name__ == "__main__":
    _main()
