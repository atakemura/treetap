from wrapper_tree_local_explanation_dt import run_experiment as local_dt_experiment
from wrapper_tree_local_explanation_lgb import run_experiment as local_lgb_experiment
from wrapper_tree_local_explanation_rf import run_experiment as local_rf_experiment
from wrapper_tree_global_explanation_lgb import run_experiment as global_lgb_experiment
from wrapper_tree_global_explanation_rf import run_experiment as global_rf_experiment
from wrapper_tree_global_explanation_dt import run_experiment as global_dt_experiment
from tree_asp.utils import time_print


import warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    datasets = [
        'autism',
        'breast',
        'cars',
        'credit_australia',
        'heart',
        'ionosphere',
        'kidney',
        'krvskp',
        'voting',
        'credit_taiwan',
        'credit_german',
        'adult',
        'compas',
        'census',
    ]

    for d_idx, data in enumerate(datasets):
        time_print('dataset {} {}/{}'.format(data, d_idx+1, len(datasets)))

        time_print('dataset {} {}/{} global '.format(data, d_idx+1, len(datasets)))
        global_dt_experiment(data)
        global_rf_experiment(data)
        global_lgb_experiment(data)

        time_print('dataset {} {}/{} local'.format(data, d_idx+1, len(datasets)))
        local_dt_experiment(data)
        local_rf_experiment(data)
        local_lgb_experiment(data)

        time_print('dataset {} {}/{} done'.format(data, d_idx+1, len(datasets)))
        time_print('='*80)
