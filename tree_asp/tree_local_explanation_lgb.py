import json
import lightgbm as lgb
import os
import numpy as np
import subprocess
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy

from hyperparameter import optuna_lgb
from rule_extractor import LGBMGlobalRuleExtractor, LGBMLocalRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from rule import Rule
from utils import load_data, time_print


SEED = 2020


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def run_one_round(dataset_name,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    experiment_tag = 'lgb_{}_{}'.format(dataset_name, fold)
    exp_dir = './tmp/journal/local'
    # if model exists, skip training
    model_path = os.path.join(exp_dir, experiment_tag+'_lgbmodel.bst')
    param_path = os.path.join(exp_dir, experiment_tag+'_lgbmodel_params.pkl')
    extractor_path = os.path.join(exp_dir, experiment_tag+'_extractor.pkl')
    tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out.txt'.format(experiment_tag))
    tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))

    log_json = os.path.join(exp_dir, 'output.json')
    log_json_quali = os.path.join(exp_dir, 'output_quali.json')
    le_log_json = os.path.join(exp_dir, 'local_explanation.json')

    n_local_instances = 100

    time_print('=' * 30 + experiment_tag + '=' * 30)
    start = timer()

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    num_classes = y_valid.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'

    time_print('lgb-training start')
    lgb_start = timer()

    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        with open(param_path, 'rb') as param_in:
            hyperparams = pickle.load(param_in)
    else:
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
        model.save_model(model_path)
        with open(param_path, 'wb') as param_out:
            pickle.dump(hyperparams, param_out, protocol=pickle.HIGHEST_PROTOCOL)

    lgb_end = timer()
    time_print('lgb-training completed {} seconds | {} from start'.format(round(lgb_end - lgb_start),
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

    time_print('rule extraction start')
    ext_start = timer()
    # if already fit extractor exists, skip

    if os.path.exists(extractor_path):
        with open(extractor_path, 'rb') as ext_pkl:
            lgb_extractor = pickle.load(ext_pkl)
        res_str = lgb_extractor.transform(x_train, y_train)
    else:
        lgb_extractor = LGBMGlobalRuleExtractor()
        lgb_extractor.fit(x_train, y_train, model=model, feature_names=feature_names)
        res_str = lgb_extractor.transform(x_train, y_train)

        with open(extractor_path, 'wb') as ext_out:
            pickle.dump(lgb_extractor, ext_out, protocol=pickle.HIGHEST_PROTOCOL)
    ext_end = timer()
    time_print('rule extraction completed {} seconds | {} from start'.format(round(ext_end - ext_start),
                                                                        round(ext_end - start)))

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
    clingo_dom_maxacc   = './asp_encoding/clingo_moo_ruleset_dom_7.lp'

    # clingo_test       = './asp_encoding/maximal_noclass.lp'
    # general_rule_test = './asp_encoding/rule_selection.lp'
    #
    # asprin_enc = {'skyline': asprin_skyline, 'maximal': asprin_maximal,
    #               'closed': asprin_closed, 'general_rule': general_rule_test}
    # asprin_preference = {'pareto_1': asprin_pareto_1, 'pareto_2': asprin_pareto_2,
    #                      'lexico': asprin_lexico, 'pareto_test': asprin_pareto}

    clingo_start = timer()
    time_print('clingo start')
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
        time_print('py rule evaluation completed {} seconds | {} from start'.format(round(py_rule_end - py_rule_start),
                                                                               round(py_rule_end - start)))

        out_dict = {
            # experiment
            'dataset': dataset_name,
            'num_class': num_classes,
            'best_iteration': model.best_iteration,
            'n_estimators': model.num_trees(),
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
            # 'encoding': encoding,
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
                # time_print('Skipping non-optimal answer: {}'.format(ans_set.answer_id))
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
    time_print('out_quali end {} seconds | {} from start'.format(round(out_quali_end - out_quali_start),
                                                            round(out_quali_end - start)))

    if verbose:
        with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
            out_log_quali.write(json.dumps(out_quali)+'\n')

    # local explanation using rules (not through clingo because it's not supposed to be a global approximation)
    time_print('local explanation start')
    le_start = timer()

    local_lgb_extractor = LGBMLocalRuleExtractor()
    local_lgb_extractor.fit(x_train, y_train, model=model, feature_names=feature_names)

    sample_idx = x_valid.sample(n_local_instances, replace=True).index
    sampled_x_valid, sampled_y_valid = x_valid.loc[sample_idx], y_valid.loc[sample_idx]

    le_score_store = {}

    for s_idx, v_idx in enumerate(sample_idx):
        time_print('local explanation {}/{}'.format(s_idx+1, n_local_instances))
        # given a single data point, find paths and rules that fire, leading to the conclusion
        local_asp_prestr = local_lgb_extractor.transform(x_valid.loc[[v_idx]], y_valid.loc[v_idx], model=model)
        if len(local_asp_prestr) > 1:
            assert False  # safety, we're explaining only 1 sample at a time, for now

        local_tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out_local.txt'.format(experiment_tag))
        with open(local_tmp_pattern_file, 'w', encoding='utf-8') as outfile:
            outfile.write(local_asp_prestr[0])

        try:
            o = subprocess.run(['clingo', './asp_encoding/clingo_local_explanation.lp',
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

        # log_json = os.path.join(exp_dir, 'output.json')
        # log_json_quali = os.path.join(exp_dir, 'output_quali.json')

        scores = []
        if clingo_completed and clasp_info is not None:
            for ans_idx, ans_set in enumerate(answers):
                if not ans_set.is_optimal:
                    continue
                rules = []
                for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                    pat_idx = ans[-1][0]
                    pat = local_lgb_extractor.rules_[pat_idx]  # type: Rule
                    rules.append(pat)
                # break
                rule_classifier = RuleClassifier(rules, default_class=0)
                rule_classifier.fit(x_train, y_train)

                rule_pred_idx = rule_classifier.predict_index(sampled_x_valid)

                # coverage
                cov = rule_pred_idx.shape[0] / float(sample_idx.shape[0])
                # precision
                prc = np.mean((model.predict(sampled_x_valid.loc[rule_pred_idx]) > 0.5).astype(int) ==
                              (model.predict(x_valid.loc[[v_idx]]) > 0.5).astype(int))

                rule_pred_metrics = {'local_coverage': cov,
                                     'local_precision': prc}
                scores.append((ans_idx, rule_pred_metrics))
        le_score_store[s_idx] = scores

    le_end = timer()
    time_print('local explanation completed {} seconds | {} from start'.format(round(le_end - le_start),
                                                                          round(le_end - start)))
    le_out_dict = {
        # experiment
        'dataset': dataset_name,
        'num_class': num_classes,
        'best_iteration': model.best_iteration,
        'n_estimators': model.num_trees(),
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
        'lgb_n_nodes': len(lgb_extractor.conditions_),
        'lgb_n_patterns': len(lgb_extractor.rules_),
        'hyperparams': hyperparams,
        # timer
        'py_total_time': end - start,
        'py_lgb_time': lgb_end - lgb_start,
        'py_ext_time': ext_end - ext_start,
        'py_local_explanation_time': le_end - le_start,
        # metrics
        'fold': fold,
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
            # 'kdd99',
            # 'airline'
            'synthetic_1'
            'adult',
            'credit_german',
            'compas'
        ]
    else:
        data = ['autism', 'breast', 'cars', 'credit_australia',
                'heart', 'ionosphere', 'kidney', 'krvskp', 'voting',
                'credit_taiwan',
                'eeg',
                'census',
                # 'kdd99',
                # 'airline'
                ]

    for d in tqdm(data):
        run_experiment(d)
    end_time = timer()
    e = end_time - start_time
    time_print('Time elapsed(s): {}'.format(e))
