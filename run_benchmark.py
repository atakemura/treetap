from wrapper_anchor import run_experiment as anchor_experiment
from wrapper_baseline import run_experiment as baseline_experiment
from wrapper_pyxai_dt import run_experiment as pyxai_dt_experiment
from wrapper_pyxai_rf import run_experiment as pyxai_rf_experiment
from tree_asp.utils import time_print

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    datasets = [
        'census',
        'credit_taiwan',
        'credit_german',
        'adult',
        'compas',
        'autism',
        'breast',
        'cars',
        'credit_australia',
        'heart',
        'ionosphere',
        'kidney',
        'krvskp',
        'voting',
    ]

    for d_idx, data in enumerate(datasets):
        time_print('dataset {} {}/{}'.format(data, d_idx+1, len(datasets)))

        # baseline and anchor cannot be executed together on the same dataset, because anchor requires discretized data
        time_print('dataset {} {}/{} baseline'.format(data, d_idx+1, len(datasets)))
        baseline_experiment(data)

        time_print('dataset {} {}/{} anchor'.format(data, d_idx+1, len(datasets)))
        anchor_experiment(data)

        time_print('dataset {} {}/{} pyxai dt'.format(data, d_idx+1, len(datasets)))
        pyxai_dt_experiment(data)

        time_print('dataset {} {}/{} pyxai rf'.format(data, d_idx+1, len(datasets)))
        pyxai_rf_experiment(data)

        time_print('dataset {} {}/{} done'.format(data, d_idx+1, len(datasets)))
        time_print('='*80)
