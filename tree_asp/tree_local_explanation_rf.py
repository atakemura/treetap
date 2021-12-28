import json
import os
import numpy as np
import pickle
import subprocess
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from category_encoders.one_hot import OneHotEncoder
from tqdm import tqdm
from timeit import default_timer as timer
from psutil import cpu_count

from rule_extractor import RFLocalRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from hyperparameter import optuna_random_forest
from rule import Rule
from utils import load_data, time_print


SEED = 2020
NUM_CPU = cpu_count(logical=False) - 1


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
        # avoid special character error
        X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def run_one_round(dataset_name,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    experiment_tag = 'rf_{}_{}'.format(dataset_name, fold)
    exp_dir = './tmp/journal/local'
    # try model pickling - if this does not work save best params and fit again
    model_path = os.path.join(exp_dir, experiment_tag+'_rfmodel.pkl')
    param_path = os.path.join(exp_dir, experiment_tag+'_rfmodel_params.pkl')

    local_tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out_local.txt'.format(experiment_tag))
    tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))

    le_log_json = os.path.join(exp_dir, 'local_explanation.json')

    n_local_instances = 100

    time_print('=' * 30 + experiment_tag + '=' * 30)
    start = timer()

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    metric_averaging = 'micro' if y_valid.nunique() > 2 else 'binary'

    time_print('rf-training start')
    rf_start = timer()
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_in:
            rf = pickle.load(model_in)
        with open(param_path, 'rb') as param_in:
            hyperparams = pickle.load(param_in)
    else:
        hyperparams = optuna_random_forest(x_train, y_train, random_state=SEED)
        rf = RandomForestClassifier(**hyperparams, random_state=SEED, n_jobs=NUM_CPU)
        rf.fit(x_train, y_train)
        with open(model_path, 'wb') as model_out:
            pickle.dump(rf, model_out, protocol=pickle.HIGHEST_PROTOCOL)
        with open(param_path, 'wb') as param_out:
            pickle.dump(hyperparams, param_out, protocol=pickle.HIGHEST_PROTOCOL)
    rf_end = timer()
    time_print('rf-training completed {} seconds | {} from start'.format(round(rf_end - rf_start),
                                                                    round(rf_end - start)))

    rf_vanilla_pred = rf.predict(x_valid)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, rf_vanilla_pred),
                       'precision': precision_score(y_valid, rf_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, rf_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, rf_vanilla_pred, average=metric_averaging),
                       'auc':       roc_auc_score(y_valid, rf_vanilla_pred)}

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        outfile.write('class(1).'.format(int(y_train.nunique() - 1)))

    time_print('local explanation start')

    local_rf_extractor = RFLocalRuleExtractor()
    local_rf_extractor.fit(x_train, y_train, model=rf, feature_names=feature_names)

    sample_idx = x_valid.sample(n_local_instances, replace=True).index
    sampled_x_valid, sampled_y_valid = x_valid.loc[sample_idx], y_valid.loc[sample_idx]

    encoding_dict = {'acc_cov':  './asp_encoding/local_accuracy_coverage.lp',
                     'prec_cov': './asp_encoding/local_precision_coverage.lp',
                     'prec_rec': './asp_encoding/local_precision_recall.lp'}

    for enc_idx, (enc_k, enc_v) in enumerate(encoding_dict.items()):
        le_start = timer()
        time_print('local explanation enc {} {}/{}'.format(enc_k, enc_idx + 1, len(encoding_dict)))
        le_score_store = {}

        for s_idx, v_idx in enumerate(sample_idx):
            time_print('local explanation {}/{}'.format(s_idx+1, n_local_instances))
            # given a single data point, find paths and rules that fire, leading to the conclusion
            local_asp_prestr = local_rf_extractor.transform(x_valid.loc[[v_idx]], y_valid.loc[v_idx], model=rf)
            if len(local_asp_prestr) > 1:
                assert False  # safety, we're explaining only 1 sample at a time, for now

            with open(local_tmp_pattern_file, 'w', encoding='utf-8') as outfile:
                outfile.write(local_asp_prestr[0])

            try:
                o = subprocess.run(['clingo', enc_v,
                                    local_tmp_pattern_file, '0',
                                    ], capture_output=True, timeout=600)
                clingo_completed = True
            except subprocess.TimeoutExpired:
                o = None
                clingo_completed = False

            if clingo_completed:
                answers, clasp_info = generate_answers(o.stdout.decode())
            else:
                answers, clasp_info = None, None

            scores = []
            if clingo_completed and clasp_info is not None:
                for ans_idx, ans_set in enumerate(answers):
                    if not ans_set.is_optimal:
                        continue
                    rules = []
                    for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                        pat_idx = ans[-1][0]
                        pat = local_rf_extractor.rules_[pat_idx]  # type: Rule
                        rules.append(pat)
                    # break
                    rule_classifier = RuleClassifier(rules, default_class=0)
                    rule_classifier.fit(x_train, y_train)

                    rule_pred_idx = rule_classifier.predict_index(sampled_x_valid)

                    # coverage
                    cov = rule_pred_idx.shape[0] / float(sample_idx.shape[0])
                    # precision
                    prc = np.mean((rf.predict(sampled_x_valid.loc[rule_pred_idx]) > 0.5).astype(int) ==
                                  (rf.predict(x_valid.loc[[v_idx]]) > 0.5).astype(int))

                    rule_pred_metrics = {'local_coverage': cov,
                                         'local_precision': prc}
                    scores.append((ans_idx, rule_pred_metrics))
            le_score_store[s_idx] = scores

        le_end = timer()
        time_print('local explanation completed {} seconds | {} from start'.format(round(le_end - le_start),
                                                                                   round(le_end - start)))
        le_out_dict = {
            # experiment
            'model': 'RandomForest',
            'experiment': experiment_tag,
            'dataset': dataset_name,
            'n_estimators': hyperparams['n_estimators'],
            'max_depth': hyperparams['max_depth'],
            # 'encoding': encoding,
            'clingo_completed': clingo_completed,
            # clasp
            'models': clasp_info.stats['Models'],
            'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            # 'optimal': int(clasp_info.stats['Optimal']),
            'clasp_time': clasp_info.stats['Time'],
            'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            # 'lgb_n_nodes': len(rf_extractor.conditions_),
            # 'lgb_n_patterns': len(rf_extractor.rules_),
            'hyperparams': hyperparams,
            # timer
            'py_total_time': le_end - start,
            'py_rf_time': rf_end - rf_start,
            # 'py_ext_time': ext_end - ext_start,
            'py_local_explanation_time': le_end - le_start,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            'local_encoding': enc_k,
            'local_encoding_file': enc_v,
            'local_explanation_scores': le_score_store
        }

        with open(le_log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(le_out_dict)+'\n')

    time_print('completed {} from start'.format(round(timer() - start)))


if __name__ == '__main__':
    start_time = timer()

    debug_mode = True

    if debug_mode:
        data = [
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
            # 'eeg',
            'census',
            # 'synthetic_1'
            'adult',
            'credit_german',
            'compas'
                ]
    else:
        data = ['breast_sk', 'iris', 'wine',
                'autism', 'breast', 'cars', 'credit_australia',
                'heart', 'ionosphere', 'kidney', 'krvskp', 'voting']
        n_estimators = [10]
        max_depths = [5]
        encodings = ['skyline', 'maximal', 'closed']
        asprin_pref = ['pareto_1', 'pareto_2', 'lexico']

    for d in tqdm(data):
        run_experiment(d)
    end_time = timer()
    e = end_time - start_time
    time_print('Time elapsed(s): {}'.format(e))
