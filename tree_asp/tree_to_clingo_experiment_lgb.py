import json
import lightgbm as lgb
import os
import pandas as pd
import numpy as np
import subprocess
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from itertools import product
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy

from rule_extractor import LGBMRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from rule import Rule
from utils import load_data


SEED = 2020


def run_experiment(dataset_name, encoding):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name, encoding,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def optuna_lgb(X, y, static_params):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}
    # optuna.logging.set_verbosity(optuna.logging.WARNING)

    def optuna_early_stopping_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if early_stopping_dict['best_score'] is None:
            early_stopping_dict['best_score'] = study.best_value

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            if study.best_value > early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if study.best_value < early_stopping_dict['best_score']:
                early_stopping_dict['best_score'] = study.best_value
                early_stopping_dict['early_stop_count'] = 0
            else:
                early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
                if early_stopping_dict['early_stop_count'] > early_stopping_dict['early_stopping_limit']:
                    study.stop()
                else:
                    early_stopping_dict['early_stop_count'] = early_stopping_dict['early_stop_count'] + 1
        else:
            raise ValueError('Unknown study direction: {}'.format(study.direction))
        return

    def objective(trial: optuna.Trial):
        params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                  'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'num_leaves': trial.suggest_int('num_leaves', 2, 100),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 500),
                  'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-3, 1e+1),
                  'feature_fraction': trial.suggest_uniform('feature_fraction', 0.05, 1.0),
                  'subsample': trial.suggest_uniform('subsample', 0.2, 1.0),
                  'subsample_freq': trial.suggest_int('subsample_freq', 1, 20),
                  'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-5, 10),
                  'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-5, 10),
                  }
        all_params = {**params, **static_params}

        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, all_params['metric'], valid_name='valid')
        num_boost_round = 1000
        early_stopping = 30
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED)
        train_data = lgb.Dataset(x_train, label=y_train)
        valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data)

        model = lgb.train(all_params, train_data,
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping,
                          valid_sets=[valid_data],
                          valid_names=['valid'],
                          callbacks=[pruning_callback],
                          verbose_eval=False)
        if static_params['num_classes'] > 1:
            score = model.best_score['valid']['multi_logloss']
        else:
            score = model.best_score['valid']['binary_logloss']
        return score
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[optuna_early_stopping_callback], n_jobs=1)
    return study.best_params


def run_one_round(dataset_name, encoding,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    experiment_tag = 'lgb_{}_{}_{}'.format(dataset_name, encoding, fold)
    print('=' * 30, experiment_tag, '=' * 30)
    start = timer()


    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    num_classes = y_valid.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'

    lgb_start = timer()
    print('lgb-training start')
    # using native api
    lgb_train = lgb.Dataset(data=x_train,
                            label=y_train)
    lgb_valid = lgb.Dataset(data=x_valid,
                            label=y_valid,
                            reference=lgb_train)

    static_params = {
        'objective': 'multiclass' if num_classes > 2 else 'binary',
        'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
        'num_classes': num_classes if num_classes > 2 else 1,
        'verbosity': -1
    }
    best_params = optuna_lgb(x_train, y_train, static_params)
    hyperparams = {**static_params, **best_params}
    model = lgb.train(params=hyperparams,
                      train_set=lgb_train,
                      valid_sets=[lgb_valid],
                      valid_names=['valid'], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)
    lgb_end = timer()
    print('lgb-training completed {} seconds | {} from start'.format(round(lgb_end - lgb_start),
                                                                     round(lgb_end - start)))

    if num_classes > 2:
        lgb_vanilla_pred = np.argmax(model.predict(x_valid), axis=1)
    else:
        lgb_vanilla_pred = (model.predict(x_valid) > 0.5).astype(int)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, lgb_vanilla_pred),
                       'precision': precision_score(y_valid, lgb_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, lgb_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, lgb_vanilla_pred, average=metric_averaging),
                       'auc':       roc_auc_score(y_valid, lgb_vanilla_pred)}

    ext_start = timer()
    print('rule extraction start')
    lgb_extractor = LGBMRuleExtractor()
    lgb_extractor.fit(x_train, y_train, model=model, feature_names=feature_names)
    res_str = lgb_extractor.transform(x_train, y_train)
    ext_end = timer()
    print('rule extraction completed {} seconds | {} from start'.format(round(ext_end - ext_start),
                                                                        round(ext_end - start)))

    exp_dir = './tmp/test'

    tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out.txt'.format(experiment_tag))
    tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(res_str)

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        # outfile.write('class(0..{}).'.format(int(y_train.nunique() - 1)))
        outfile.write('class(1).')

    # asprin_pareto_1   = './asp_encoding/asprin_pareto_1.lp'
    # asprin_pareto_2   = './asp_encoding/asprin_pareto_2.lp'
    # asprin_lexico     = './asp_encoding/asprin_lexico.lp'
    # asprin_pareto     = './asp_encoding/asprin_pareto.lp'
    # asprin_skyline    = './asp_encoding/skyline.lp'
    # asprin_skyline    = './asp_encoding/asprin_skyline_new.lp'
    # asprin_maximal    = './asp_encoding/maximal.lp'
    # asprin_closed     = './asp_encoding/closed.lp'

    clingo_test         = './asp_encoding/clingo_moo_ruleset.lp'
    clingo_acc          = './asp_encoding/clingo_moo_ruleset_acc.lp'
    clingo_prec         = './asp_encoding/clingo_moo_ruleset_prec.lp'
    clingo_f1size       = './asp_encoding/clingo_moo_ruleset_f1size.lp'
    clingo_acc_sup      = './asp_encoding/clingo_moo_ruleset_acc_support.lp'
    clingo_f1_sup       = './asp_encoding/clingo_moo_ruleset_f1_support.lp'
    clingo_dom_prec     = './asp_encoding/clingo_moo_ruleset_dom_3.lp'
    clingo_dom_rec      = './asp_encoding/clingo_moo_ruleset_dom_4.lp'
    clingo_dom_maxacc   = './asp_encoding/clingo_moo_ruleset_dom_6.lp'

    # clingo_test       = './asp_encoding/maximal_noclass.lp'
    # general_rule_test = './asp_encoding/rule_selection.lp'
    #
    # asprin_enc = {'skyline': asprin_skyline, 'maximal': asprin_maximal,
    #               'closed': asprin_closed, 'general_rule': general_rule_test}
    # asprin_preference = {'pareto_1': asprin_pareto_1, 'pareto_2': asprin_pareto_2,
    #                      'lexico': asprin_lexico, 'pareto_test': asprin_pareto}

    clingo_start = timer()
    print('clingo start')
    try:
        # o = subprocess.run(['asprin', asprin_preference[asprin_pref], asprin_enc[encoding],
        #                     tmp_class_file, tmp_pattern_file, '0', '--parallel-mode=16'
        #                     ], capture_output=True, timeout=3600)
        o = subprocess.run(['clingo', clingo_dom_maxacc, #asprin_pareto_1,
                            tmp_class_file, tmp_pattern_file, '0', '--parallel-mode=8,split'
                            ], capture_output=True, timeout=600)
        clingo_completed = True
    except subprocess.TimeoutExpired:
        o = None
        clingo_completed = False
    clingo_end = timer()
    print('clingo completed {} seconds | {} from start'.format(round(clingo_end - clingo_start),
                                                               round(clingo_end - start)))

    if clingo_completed:
        answers, clasp_info = generate_answers(o.stdout.decode())
    else:
        answers, clasp_info = None, None
    end = timer()
    # print('parsing completed')

    log_json = os.path.join(exp_dir, 'output.json')
    log_json_quali = os.path.join(exp_dir, 'output_quali.json')

    if clingo_completed and clasp_info is not None:
        py_rule_start = timer()
        print('py rule evaluation start')
        scores = []
        for ans_idx, ans_set in enumerate(answers):
            if not ans_set.is_optimal:
                continue
            rules = []
            for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                pat_idx = ans[-1][0]
                pat = lgb_extractor.rules_[pat_idx]  # type: Rule
                rules.append(pat)
            # break
            rule_classifier = RuleClassifier(rules, default_class=0)
            rule_classifier.fit(x_train, y_train)
            rule_pred = rule_classifier.predict(x_valid)
            rule_pred_metrics = {'accuracy': accuracy_score(y_valid, rule_pred),
                                 'precision': precision_score(y_valid, rule_pred, average=metric_averaging),
                                 'recall': recall_score(y_valid, rule_pred, average=metric_averaging),
                                 'f1': f1_score(y_valid, rule_pred, average=metric_averaging),
                                 'auc': roc_auc_score(y_valid, rule_pred)}
            scores.append((ans_idx, rule_pred_metrics))
        py_rule_end = timer()
        print('py rule evaluation completed {} seconds | {} from start'.format(round(py_rule_end - py_rule_start),
                                                                               round(py_rule_end - start)))

        out_dict = {
            # experiment
            'dataset': dataset_name,
            'num_class': num_classes,
            'best_iteration': model.best_iteration,
            'n_estimators': model.num_trees(),
            'max_depth': hyperparams['max_depth'],
            'encoding': encoding,
            'clingo_completed': clingo_completed,
            # clasp
            'models': clasp_info.stats['Models'],
            'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            # 'optimal': int(clasp_info.stats['Optimal']),
            'clasp_time': clasp_info.stats['Time'],
            'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            'lgb_n_nodes': len(lgb_extractor.conditions_),
            'lgb_n_patterns': len(lgb_extractor.rules_),
            'hyperparams': hyperparams,
            # timer
            'py_total_time': end - start,
            'py_lgb_time': lgb_end - lgb_start,
            'py_ext_time': ext_end - ext_start,
            'py_clingo_time': clingo_end - clingo_start,
            'py_rule_time': py_rule_end - py_rule_start,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            # 'rule_metrics': rule_pred_metrics,
            'rule_metrics': scores,
        }
    else:
        out_dict = {
            # experiment
            'dataset': dataset_name,
            'num_class': num_classes,
            'best_iteration': model.best_iteration,
            'n_estimators': model.num_trees(),
            'max_depth': hyperparams['max_depth'],
            'encoding': encoding,
            'clingo_completed': clingo_completed,
            # # clasp
            # 'models': int(clasp_info.stats['Models']),
            # 'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            # 'optimal': int(clasp_info.stats['Optimal']),
            # 'clasp_time': clasp_info.stats['Time'],
            # 'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # lgb related
            'lgb_n_nodes': len(lgb_extractor.conditions_),
            'lgb_n_patterns': len(lgb_extractor.rules_),
            'hyperparams': hyperparams,
            # timer
            'py_total_time': end - start,
            'py_lgb_time': lgb_end - lgb_start,
            'py_ext_time': ext_end - ext_start,
            'py_clingo_time': clingo_end - clingo_start,
            'py_rule_time': 0,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            # 'rule_metrics': rule_pred_metrics,
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
                # print('Skipping non-optimal answer: {}'.format(ans_set.answer_id))
                continue
            for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                pat_idx = ans[-1][0]
                pat = lgb_extractor.rules_[pat_idx]  # type: Rule
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
    print('out_quali end {} seconds | {} from start'.format(round(out_quali_end - out_quali_start),
                                                            round(out_quali_end - start)))

    if verbose:
        with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
            out_log_quali.write(json.dumps(out_quali)+'\n')

    print('completed {} from start'.format(round(timer() - start)))


if __name__ == '__main__':
    start_time = timer()

    debug_mode = True

    if debug_mode:
        data = ['autism', 'breast',
                #'cars',
                #'credit_australia',
                #'heart', 'ionosphere', 'kidney', 'krvskp', 'voting',
                #'credit_taiwan',
                # 'eeg',
                #'census',
                # 'kdd99',
                # 'airline'
                # 'synthetic_1'
        ]
        encodings = ['skyline']
        asprin_pref = ['pareto_1']
    else:
        data = ['autism', 'breast', 'cars', 'credit_australia',
                'heart', 'ionosphere', 'kidney', 'krvskp', 'voting',
                'credit_taiwan',
                'eeg',
                'census',
                # 'kdd99',
                # 'airline'
                ]
        encodings = ['skyline', 'maximal', 'closed']
        # asprin_pref = ['pareto_1', 'pareto_2', 'lexico']
        asprin_pref = ['pareto_1']

    combinations = product(data, encodings)
    for cond_tuple in tqdm(combinations):
        run_experiment(*cond_tuple)
    end_time = timer()
    e = end_time - start_time
    print('Time elapsed(s): {}'.format(e))
