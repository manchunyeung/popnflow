"""Dataset selection for the fit-variance diagnostics.

`make_diagnostic_plots.py` is invoked with `--run mdc` for BOTH datasets (the
flag picks the likelihood/plot branch, not the data), and every path default
points at the MDC. Reaching sim_cat6 means overriding all of them, so keep that
in one place rather than repeating it in each script.

sim_cat6_sharp_w8 is the catalog the paper uses. The MDC is retired.
"""
import os

PROD = '/home/manchun.yeung/population/simon/popnflow/production'

# The fitvar pool cache is dataset-specific; --fitvar-pool-size then caps M.
_SIMCAT6 = dict(
    indir='sim_cat6_sharp_w8/inference',
    pe_m1det='sim_cat6_sharp_w8/paper_diagnostics/m1det.txt',
    pe_m2det='sim_cat6_sharp_w8/paper_diagnostics/m2det.txt',
    pe_dL='sim_cat6_sharp_w8/paper_diagnostics/dL.txt',
    injections='sim_cat6_sharp_w8/paper_diagnostics/injections_endo3schema.h5',
    pool_cache='sim_cat6_sharp_w8/paper_diagnostics/pool_fitvar_cat000.npz',
    extra=['--sampler-pattern', 'simcat_cat{i:03d}_{method}.h5',
           '--method-patterns', 'met3=simcatnm_cat{i:03d}_{method}.h5'],
    nsamp_expected=5000,
)

_MDC = dict(
    indir='results/data',
    pe_m1det='input_data/mdc_m1det_69rand.txt',
    pe_m2det='input_data/mdc_m2det_69rand.txt',
    pe_dL='input_data/mdc_dL_69rand.txt',
    injections='input_data/endo3_bbhpop-LIGO-T2100113-v12.hdf5',
    pool_cache='cache/pool_1M.npz',
    extra=[],
    nsamp_expected=3343,
)

DATASETS = {'simcat6': _SIMCAT6, 'mdc': _MDC}


def build_argv(dataset, outdir, n_coords, seed):
    """sys.argv for importing make_diagnostic_plots against `dataset`."""
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; pick from {list(DATASETS)}")
    d = DATASETS[dataset]
    p = lambda rel: os.path.join(PROD, rel)
    return [
        'make_diagnostic_plots.py', '--run', 'mdc', '--catalog', '0',
        '--indir', p(d['indir']),
        '--outdir', outdir,
        '--pe-m1det', p(d['pe_m1det']),
        '--pe-m2det', p(d['pe_m2det']),
        '--pe-dL', p(d['pe_dL']),
        '--injection-file', p(d['injections']),
        '--pool-cache', p(d['pool_cache']),
        '--fitvar-n-coords', str(n_coords),
        '--seed', str(seed),
        # cheapest task that satisfies the parser, and it validates the very
        # density-evaluation path these diagnostics reuse
        '--fitvar-moment-check',
    ] + d['extra']


def per_event_K(dataset, catalog=0):
    """The per-event GMM component counts the INFERENCE actually uses.

    sim_cat_inference.py loads pre-fitted GMMs (`--gmm-file`) and takes each
    event's K from the loaded shape -- `--kfixed` is recorded but moot. Those
    K's are the BIC-argmin selection in event_gmms.h5 and run 3..10 (median 5);
    only 5 of 69 events are K=7. The fitvar diagnostic's uniform
    K_PER_EVENT_FITVAR=7 therefore measures a DIFFERENT estimator than the one
    being used for inference, which is why this exists.

    Returns an int array of length Nobs, ordered as the catalog's events.
    """
    import json

    import h5py
    import numpy as np

    if dataset != 'simcat6':
        raise SystemExit(f"per-event K is only wired up for simcat6, not {dataset!r}")
    info = os.path.join(PROD, f'sim_cat6_sharp_w8/inference/simcat_cat{catalog:03d}_info.json')
    with open(info) as fh:
        d = json.load(fh)
    idx = d['event_indices']
    with h5py.File(os.path.join(PROD, 'sim_cat6_sharp_w8/event_gmms.h5'), 'r') as f:
        K = np.array([f[f'pool_{i:03d}'].attrs['k_argmin'] for i in idx], dtype=int)
    rec = np.asarray(d['K_per_event'], dtype=int)
    if not np.array_equal(K, rec):
        raise SystemExit("[dataset] k_argmin does not match the run's recorded "
                         "K_per_event -- refusing to guess which the inference used.")
    print(f"[dataset] per-event K from event_gmms.h5 (BIC argmin): "
          f"min={K.min()} median={int(np.median(K))} max={K.max()} "
          f"mean={K.mean():.2f}; {(K == 7).sum()}/{len(K)} events are K=7")
    return K


def check_nsamp(dataset, nsamp):
    """Guard against silently running on the wrong catalog."""
    want = DATASETS[dataset]['nsamp_expected']
    if nsamp != want:
        raise SystemExit(
            f"[dataset] expected nsamp={want} for {dataset!r} but loaded {nsamp}. "
            f"A diagnostic printing nsamp=3343 is on the MDC.")
    print(f"[dataset] {dataset}: nsamp={nsamp} per event -- OK")
