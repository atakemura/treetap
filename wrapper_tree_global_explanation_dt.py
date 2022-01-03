import json
import os
import subprocess
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from category_encoders.one_hot import OneHotEncoder
from timeit import default_timer as timer
from copy import deepcopy

from hyperparameter import optuna_decision_tree
from rule_extractor import DTGlobalRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from rule import Rule
from utils import load_data
from tree_asp.utils import time_print


SEED = 2020


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
        # avoid special character error
        X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def run_one_round(dataset_name,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    experiment_tag = 'global_dt_{}_{}'.format(dataset_name, fold)
    exp_dir = 'tree_asp/tmp/journal/global'
    tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out.txt'.format(experiment_tag))
    tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))
    log_json = os.path.join(exp_dir, 'global_output.json')
    log_json_quali = os.path.join(exp_dir, 'global_output_quali.json')

    time_print('=' * 30 + experiment_tag + '=' * 30)
    start = timer()

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    metric_averaging = 'micro' if y_valid.nunique() > 2 else 'binary'

    dt_start = timer()
    best_params = optuna_decision_tree(x_train, y_train)
    dt = DecisionTreeClassifier(**best_params, random_state=SEED)
    dt.fit(x_train, y_train)
    dt_end = timer()

    dt_vanilla_pred = dt.predict(x_valid)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, dt_vanilla_pred),
                       'precision': precision_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'auc':       roc_auc_score(y_valid, dt_vanilla_pred)}

    ext_start = timer()
    dt_extractor = DTGlobalRuleExtractor()
    dt_extractor.fit(x_train, y_train, model=dt, feature_names=feature_names)
    res_str = dt_extractor.transform(x_train, y_train)
    ext_end = timer()

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(res_str)

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        outfile.write('class(1).'.format(int(y_train.nunique() - 1)))

    encoding_dict = {
        'acc_cov':  'tree_asp/asp_encoding/global_accuracy_coverage.lp',
        'prec_cov': 'tree_asp/asp_encoding/global_precision_coverage.lp',
        'prec_rec': 'tree_asp/asp_encoding/global_precision_recall.lp'}

    for enc_idx, (enc_k, enc_v) in enumerate(encoding_dict.items()):
        clingo_start = timer()
        time_print('clingo_start')
        try:
            o = subprocess.run(['clingo', enc_v,
                                tmp_class_file, tmp_pattern_file, '0', '--parallel-mode=8,split'
                                ], capture_output=True, timeout=600)
            clingo_completed = True
        except subprocess.TimeoutExpired:
            o = None
            clingo_completed = False
        clingo_end = timer()
        time_print('clingo completed {} seconds | {} from start'.format(round(clingo_end - clingo_start),
                                                                        round(clingo_end - start)))

        if clingo_completed:
            answers, clasp_info = generate_answers(o.stdout.decode())
        else:
            answers, clasp_info = None, None
        end = timer()

        if clingo_completed and clasp_info is not None:
            py_rule_start = timer()
            time_print('py rule evaluation start')
            scores = []
            for ans_idx, ans_set in enumerate(answers):
                if not ans_set.is_optimal:
                    continue
                rules = []
                for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                    pat_idx = ans[-1][0]
                    pat = dt_extractor.rules_[pat_idx]  # type: Rule
                    rules.append(pat)
                # break
                rule_classifier = RuleClassifier(rules)
                rule_classifier.fit(x_train, y_train)
                rule_pred = rule_classifier.predict(x_valid)
                rule_pred_metrics = {'accuracy': accuracy_score(y_valid, rule_pred),
                                     'precision': precision_score(y_valid, rule_pred, average=metric_averaging),
                                     'recall': recall_score(y_valid, rule_pred, average=metric_averaging),
                                     'f1': f1_score(y_valid, rule_pred, average=metric_averaging),
                                     'auc': roc_auc_score(y_valid, rule_pred)}
                scores.append((ans_idx, rule_pred_metrics))
            py_rule_end = timer()
            time_print('py rule evaluation completed {} seconds | {} from start'.format(
                round(py_rule_end - py_rule_start), round(py_rule_end - start)))

            out_dict = {
                # experiment
                'model': 'DecisionTree',
                'experiment': experiment_tag,
                'dataset': dataset_name,
                # 'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                # 'asprin_preference': asprin_pref,
                'clingo_completed': clingo_completed,
                # clasp
                'models': clasp_info.stats['Models'],
                'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
                # 'optimal': int(clasp_info.stats['Optimal']),
                'clasp_time': clasp_info.stats['Time'],
                'clasp_cpu_time': clasp_info.stats['CPU Time'],
                # dt related
                'dt_n_nodes': len(dt_extractor.conditions_),
                'dt_n_patterns': len(dt_extractor.rules_),
                'hyperparams': best_params,
                # timer
                'py_total_time': end - start,
                'py_dt_time': dt_end - dt_start,
                'py_ext_time': ext_end - ext_start,
                'py_asprin_time': clingo_end - clingo_start,
                # metrics
                'fold': fold,
                'vanilla_metrics': vanilla_metrics,
                'global_encoding': enc_k,
                'global_encoding_file': enc_v,
                'rule_metrics': scores,
            }
        else:
            out_dict = {
                # experiment
                'model': 'DecisionTree',
                'experiment': experiment_tag,
                'dataset': dataset_name,
                # 'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                # 'asprin_preference': asprin_pref,
                'clingo_completed': clingo_completed,
                # # clasp
                # 'models': int(clasp_info.stats['Models']),
                # 'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
                # 'optimal': int(clasp_info.stats['Optimal']),
                # 'clasp_time': clasp_info.stats['Time'],
                # 'clasp_cpu_time': clasp_info.stats['CPU Time'],
                # dt related
                'dt_n_nodes': len(dt_extractor.conditions_),
                'dt_n_patterns': len(dt_extractor.rules_),
                'hyperparams': best_params,
                # timer
                'py_total_time': end - start,
                'py_dt_time': dt_end - dt_start,
                'py_ext_time': ext_end - ext_start,
                'py_asprin_time': clingo_end - clingo_start,
                # metrics
                'fold': fold,
                'vanilla_metrics': vanilla_metrics,
                # 'rule_metrics': rule_pred_metrics,
                'global_encoding': enc_k,
                'global_encoding_file': enc_v,
            }
        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(out_dict)+'\n')

        # this is for qualitative answer pattern only
        verbose = True
        out_quali_start = timer()
        out_quali = deepcopy(out_dict)
        out_quali['rules'] = []
        if clingo_completed:
            for ans_idx, ans_set in enumerate(answers):
                _tmp_rules = []
                if not ans_set.is_optimal:
                    # time_print('Skipping non-optimal answer: {}'.format(ans_set.answer_id))
                    continue
                for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                    pat_idx = ans[-1][0]
                    pat = dt_extractor.rules_[pat_idx]  # type: Rule
                    pat_dict = {
                        'rule_idx': pat.idx,
                        'items': [x.condition_str for x in pat.items],
                        'rule_str': 'class {} IF {}'.format(pat.predict_class, pat.rule_str),
                        'predict_class': int(pat.predict_class),
                        'error_rate': int(pat.error_rate),
                        'accuracy': int(pat.accuracy),
                        'precision': int(pat.precision),
                        'f1_score': int(pat.f1_score),
                        'size': int(pat.size),
                        'support': int(pat.support),
                    }
                    _tmp_rules.append(pat_dict)
                out_quali['rules'].append((ans_idx, _tmp_rules))
        out_quali_end = timer()
        time_print('out_quali end {} seconds | {} from start'.format(round(out_quali_end - out_quali_start),
                                                                     round(out_quali_end - start)))

        if verbose:
            with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
                out_log_quali.write(json.dumps(out_quali)+'\n')

    time_print('completed {} from start'.format(round(timer() - start)))


if __name__ == '__main__':
    start_time = timer()

    for data in [
        'autism',
        'breast',
        'cars',
        'credit_australia',
        'heart',
        'ionosphere',
        'kidney',
        'krvskp',
        'voting',
        'census',
        'synthetic_1',
        'credit_taiwan',
        'credit_german',
        'adult',
        'compas'
    ]:
        time_print('='*40 + data + '='*40)
        run_experiment(data)

    end_time = timer()
    e = end_time - start_time
    time_print('Time elapsed(s): {}'.format(e))
