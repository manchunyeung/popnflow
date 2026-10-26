#!/usr/bin/env python
"""
sbc_prior.py
============
The recovery prior, as a drawable object.

SBC is only valid if Lambda_j is drawn from the IDENTICAL distribution the
sampler uses in recovery (spec 3.1). The sampler's prior lives inline in
`sim_cat_inference.py` as module-level `lower_bound`/`upper_bound` arrays plus
a linear `prior_transform` -- there is no prior object to import, and the file
CANNOT be imported at all (it parses argv and loads HDF5 at import time).

Rather than fork the numbers, this module declares them once and
`test_prior_agreement.py` lifts the real arrays out of `sim_cat_inference.py`
with `ast` and asserts they match. So the production file is never touched, and
the seam is still guarded by a test that can actually fail.

THE PRIOR IS A SEPARABLE UNIFORM BOX WITH NO JOINT CONSTRAINTS. That is not an
assumption -- it is checked. The one constraint that could bite is that the
Gaussian peak must sit inside the mass range, and it does so automatically:
  mu in [20, 50], m_min in [2, 10], m_max in [50, 100]
  => mu >= 20 > 10 >= m_min   and   mu <= 50 <= m_max   for every box point.
`assert_no_joint_constraint_violations()` re-derives this at runtime.
"""
from __future__ import annotations

import numpy as np

# Order is load-bearing: it matches sim_cat_inference.py's `labels`, which is
# the column order of the dynesty samples the PIT step reads.
PARAM_NAMES = ["gamma", "m_min", "m_max", "alpha", "beta", "mu", "sigma", "f1"]

# sim_cat_inference.py uses these longer names in `labels`; keep the mapping so
# the agreement test can compare like with like.
INFERENCE_LABELS = ["gamma", "m_min_1", "m_max_1", "alpha_1", "beta",
                    "mu", "sigma", "f1"]

LOWER = np.array([0.0,  2.0,  50.0, 0.0, 0.0,  20.0, 1.0,  0.0])
UPPER = np.array([10.0, 10.0, 100.0, 6.0, 6.0, 50.0, 10.0, 1.0])

# The fixed-Lambda MDC truth, for the boundary-distance diagnostic (spec 7).
LAMBDA_MDC = dict(gamma=3.0, m_min=6.0, m_max=70.0, alpha=4.0,
                  beta=1.5, mu=35.0, sigma=4.0, f1=0.04)

NDIM = len(PARAM_NAMES)


def prior_transform(u):
    """Unit cube -> parameter space. Mirrors sim_cat_inference.py:847."""
    u = np.asarray(u, dtype=float)
    return u * (UPPER - LOWER) + LOWER


def inverse_transform(theta):
    """Parameter space -> unit cube. Used to check draws land in [0,1]."""
    theta = np.asarray(theta, dtype=float)
    return (theta - LOWER) / (UPPER - LOWER)


def draw(n, rng):
    """Draw n hyperparameter vectors from the prior. Returns (n, NDIM)."""
    return prior_transform(rng.random((n, NDIM)))


def to_dict(theta):
    """(NDIM,) array -> the dict sim_cat_03's --true-pop-json expects."""
    return {k: float(v) for k, v in zip(PARAM_NAMES, np.asarray(theta))}


def is_finite_logprior(theta):
    """Spec 3.1: every drawn Lambda_j must have finite log-prior."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return np.all((theta >= LOWER) & (theta <= UPPER), axis=1)


def assert_no_joint_constraint_violations(n=200_000, seed=0):
    """Verify the box needs no rejection sampling, rather than assuming it.

    Checks the two orderings the model would require: the peak inside the mass
    range, and m_min < m_max. If either could ever fail, SBC would need
    rejection sampling on the joint and this function is where you would find
    out.
    """
    rng = np.random.default_rng(seed)
    t = draw(n, rng)
    d = {k: t[:, i] for i, k in enumerate(PARAM_NAMES)}
    problems = []
    if not np.all(d["m_min"] < d["m_max"]):
        problems.append("m_min < m_max violated")
    if not np.all(d["mu"] >= d["m_min"]):
        problems.append("mu >= m_min violated")
    if not np.all(d["mu"] <= d["m_max"]):
        problems.append("mu <= m_max violated")
    if problems:
        raise AssertionError(
            "prior box requires rejection sampling: " + "; ".join(problems))
    return True


def summary():
    lines = [f"{'param':>8} {'low':>8} {'high':>8} {'MDC truth':>10} "
             f"{'truth pctile':>13}"]
    for i, k in enumerate(PARAM_NAMES):
        t = LAMBDA_MDC[k]
        pct = (t - LOWER[i]) / (UPPER[i] - LOWER[i])
        lines.append(f"{k:>8} {LOWER[i]:>8.2f} {UPPER[i]:>8.2f} "
                     f"{t:>10.2f} {pct:>13.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    assert_no_joint_constraint_violations()
    print("\nno joint-constraint violations over 200k draws "
          "-> plain uniform draws are valid, no rejection needed")
