#!/usr/bin/env python
import os, sys
PROD = '/home/manchun.yeung/population/simon/popnflow/production'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

def prior_transform(u): return u
def loglike_method_1(theta): return 0.0
def loglike_method_3(theta): return 0.0
def loglike_method_4(theta): return 0.0


from _dataset import build_argv
sys.argv = build_argv('simcat6', os.path.join(HERE, 'paper_plots'), 50, 777)
# Add the flags to actually run the fitvar scan and plot
sys.argv.extend(['--make-fitvar-plot', '--fitvar-n-jobs', '30', '--fitvar-nboot', '50'])

sys.path.insert(0, PROD)
import make_diagnostic_plots as MD

if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'paper_plots'), exist_ok=True)
    MD.main()
