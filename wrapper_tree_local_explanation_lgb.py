import json
import lightgbm as lgb
import os
import numpy as np
import subprocess
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from timeit import default_timer as timer

from tree_asp.rule_extractor import LGBMLocalRuleExtractor
from tree_asp.classifier import RuleClassifier
from tree_asp.clasp_parser import generate_answers
from tree_asp.rule import Rule
from tree_asp.utils import time_print
from hyperparameter import optuna_lgb
from utils import load_data


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
    exp_dir = 'tree_asp/tmp/journal/local'

    # if model exists, skip training
    model_path = os.path.join(exp_dir, experiment_tag+'_lgbmodel.bst')
    param_path = os.path.join(exp_dir, experiment_tag+'_lgbmodel_params.pkl')

    local_tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out_local.txt'.format(experiment_tag))
    tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))

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
                          valid_names=['valid'],
                          num_boost_round=1000,
                          callbacks=[lgb.callback.early_stopping(50, verbose=False)])
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

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        # outfile.write('class(0..{}).'.format(int(y_train.nunique() - 1)))
        outfile.write('class(1).')

    time_print('local explanation start')

    local_lgb_extractor = LGBMLocalRuleExtractor()
    local_lgb_extractor.fit(x_train, y_train, model=model, feature_names=feature_names)

    sample_idx = x_valid.sample(n_local_instances, replace=True, random_state=SEED).index
    sampled_x_valid, sampled_y_valid = x_valid.loc[sample_idx], y_valid.loc[sample_idx]

    encoding_dict = {'acc_cov':  'tree_asp/asp_encoding/local_accuracy_coverage.lp',
                     'prec_cov': 'tree_asp/asp_encoding/local_precision_coverage.lp',
                     'prec_rec': 'tree_asp/asp_encoding/local_precision_recall.lp'}

    for enc_idx, (enc_k, enc_v) in enumerate(encoding_dict.items()):
        le_start = timer()
        time_print('\tlocal explanation enc {} {}/{}'.format(enc_k, enc_idx+1, len(encoding_dict)))
        le_score_store = {}
        le_rule_store = {}
        le_time_taken = []

        for s_idx, v_idx in enumerate(sample_idx):
            if ((s_idx+1) % 10) == 0:
                time_print('\t\tlocal explanation {}/{}'.format(s_idx+1, n_local_instances))
            # given a single data point, find paths and rules that fire, leading to the conclusion
            local_asp_prestr = local_lgb_extractor.transform(x_valid.loc[[v_idx]], y_valid.loc[v_idx], model=model)
            if len(local_asp_prestr) > 1:
                assert False  # safety, we're explaining only 1 sample at a time, for now

            with open(local_tmp_pattern_file, 'w', encoding='utf-8') as outfile:
                outfile.write(local_asp_prestr[0])

            clingo_completed = False
            clingo_start = timer()
            try:
                o = subprocess.run(['clingo', enc_v,
                                    local_tmp_pattern_file, '0',
                                    ], capture_output=True, timeout=600)
                clingo_completed = True
                clingo_end = timer()
                le_time_taken.append(clingo_end - clingo_start)
            except subprocess.TimeoutExpired:
                o = None
                clingo_completed = False
                le_time_taken.append(600)  # timed out

            if clingo_completed:
                answers, clasp_info = generate_answers(o.stdout.decode())
            else:
                answers, clasp_info = None, None

            scores = []
            local_rules = []
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
                    local_rules.append((ans_idx, [f'{r.predict_class} IF {r.rule_str}' for r in rules]))
            le_score_store[s_idx] = scores
            le_rule_store[s_idx] = local_rules

        le_end = timer()
        time_print('local explanation completed {} seconds | {} from start'.format(round(le_end - le_start),
                                                                                   round(le_end - start)))
        le_out_dict = {
            # experiment
            'model': 'LightGBM',
            'experiment': experiment_tag,
            'dataset': dataset_name,
            'num_class': num_classes,
            'best_iteration': model.best_iteration,
            'n_estimators': model.num_trees(),
            'max_depth': hyperparams['max_depth'],
            # 'encoding': encoding,
            'clingo_completed': clingo_completed,
            'hyperparams': hyperparams,
            # timer
            'py_total_time': le_end - start,
            'py_lgb_time': lgb_end - lgb_start,
            # 'py_ext_time': ext_end - ext_start,
            'py_local_explanation_time': le_end - le_start,
            'py_local_explanation_clingo_time': sum(le_time_taken),
            'py_local_explanation_clingo_time_array': le_time_taken,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            'local_encoding': enc_k,
            'local_encoding_file': enc_v,
            'local_explanation_scores': le_score_store,
            'local_explanation_rules': le_rule_store
        }

        with open(le_log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(le_out_dict)+'\n')

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
