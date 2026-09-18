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


def check_nsamp(dataset, nsamp):
    """Guard against silently running on the wrong catalog."""
    want = DATASETS[dataset]['nsamp_expected']
    if nsamp != want:
        raise SystemExit(
            f"[dataset] expected nsamp={want} for {dataset!r} but loaded {nsamp}. "
            f"A diagnostic printing nsamp=3343 is on the MDC.")
    print(f"[dataset] {dataset}: nsamp={nsamp} per event -- OK")
