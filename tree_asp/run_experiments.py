import weka.core.jvm as jvm

from anchor_eval.wrapper import run_experiment as anchor_experiment
from benchmark_eval.wrapper import run_experiment as baseline_experiment
from weka_eval.wrapper import run_experiment as weka_experiment
from tree_local_explanation_dt import run_experiment as local_dt_experiment
from tree_local_explanation_lgb import run_experiment as local_lgb_experiment
from tree_local_explanation_rf import run_experiment as local_rf_experiment
from tree_global_explanation_lgb import run_experiment as global_lgb_experiment
from tree_global_explanation_rf import run_experiment as global_rf_experiment
from tree_global_explanation_dt import run_experiment as global_dt_experiment
from utils import time_print


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

    try:
        jvm.start()
        for d_idx, data in enumerate(datasets):
            time_print('dataset {} {}/{}'.format(data, d_idx+1, len(datasets)))

            time_print('baseline')
            baseline_experiment(data)

            time_print('weka')
            weka_experiment(data)

            time_print('anchor')
            anchor_experiment(data)

            time_print('global')
            global_dt_experiment(data)
            global_rf_experiment(data)
            global_lgb_experiment(data)

            time_print('local')
            local_dt_experiment(data)
            local_rf_experiment(data)
            local_lgb_experiment(data)

            time_print('dataset {} done'.format(data))
            time_print('='*80)
    finally:
        jvm.stop()
