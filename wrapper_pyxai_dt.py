import json
import os
import numpy as np
import pickle
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from category_encoders.one_hot import OneHotEncoder
from timeit import default_timer as timer

from tree_asp.classifier import RuleClassifier
from tree_asp.rule import Rule, Condition
from tree_asp.utils import time_print
from hyperparameter import optuna_decision_tree
from utils import load_data

from pyxai import Learning, Explainer


SEED = 2020


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
        # avoid special character error
        operators = [('>=', '_ge_'),
                     ('<=', '_le_'),
                     ('>',  '_gt_'),
                     ('<',  '_lt_'),
                     ('!=', '_nq_'),
                     ('=',  '_eq_')]
        for op_s, op_r in operators:
            X = X.rename(columns=lambda x: re.sub(op_s, op_r, x))
        X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def pyxai_tuple_to_rule(pyxai_tuple: tuple[int], px_explainer: Explainer, predict_class: int):
    # tuple of int into rules with feature names
    feature_names_conditions = px_explainer.to_features(pyxai_tuple, eliminate_redundant_features=False)
    conditions = [Condition(0, cond_str) for cond_str in feature_names_conditions]
    rule_str = f'{predict_class} IF ' + ' AND '.join(feature_names_conditions)
    rule = Rule(None,
                rule_str,
                conditions,
                None,
                len(conditions),
                None,
                None,
                None,
                None,
                None,
                predict_class,
                predict_class)
    return rule


def run_one_round(dataset_name,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    experiment_tag = 'dt_{}_{}'.format(dataset_name, fold)
    exp_dir = 'tree_asp/tmp/journal/local'
    # try model pickling - if this does not work save best params and fit again
    model_path = os.path.join(exp_dir, experiment_tag+'_dtmodel.pkl')
    param_path = os.path.join(exp_dir, experiment_tag+'_dtmodel_params.pkl')

    le_log_json = os.path.join(exp_dir, 'local_explanation_pyxai.json')

    n_local_instances = 100

    time_print('=' * 30 + experiment_tag + '=' * 30)
    start = timer()

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    metric_averaging = 'micro' if y_valid.nunique() > 2 else 'binary'

    time_print('dt-training start')
    dt_start = timer()
    if os.path.exists(model_path):
        with open(model_path, 'rb') as model_in:
            dt = pickle.load(model_in)
        with open(param_path, 'rb') as param_in:
            hyperparams = pickle.load(param_in)
    else:
        hyperparams = optuna_decision_tree(x_train, y_train, random_state=SEED)
        dt = DecisionTreeClassifier(**hyperparams, random_state=SEED)
        dt.fit(x_train, y_train)
        with open(model_path, 'wb') as model_out:
            pickle.dump(dt, model_out, protocol=pickle.HIGHEST_PROTOCOL)
        with open(param_path, 'wb') as param_out:
            pickle.dump(hyperparams, param_out, protocol=pickle.HIGHEST_PROTOCOL)
    dt_end = timer()
    time_print('dt-training completed {} seconds | {} from start'.format(round(dt_end - dt_start),
                                                                         round(dt_end - start)))

    dt_vanilla_pred = dt.predict(x_valid)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, dt_vanilla_pred),
                       'precision': precision_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'auc':       roc_auc_score(y_valid, dt_vanilla_pred)}

    time_print('local explanation start')
    le_start = timer()
    # pyxai start

    # sample number of instances for evaluation
    sample_idx = x_valid.sample(n_local_instances, replace=True, random_state=SEED).index
    sampled_x_valid, sampled_y_valid = x_valid.loc[sample_idx], y_valid.loc[sample_idx]

    # import model
    px_learner, px_model = Learning.import_models(dt, feature_names=feature_names)

    le_direct_score_store, le_direct_rule_store = {}, {}
    le_direct_time_taken = []
    le_sufficient_score_store, le_sufficient_rule_store = {}, {}
    le_sufficient_time_taken = []

    for s_idx, v_idx in enumerate(sample_idx):
        if ((s_idx + 1) % 10) == 0:
            time_print('\t\tlocal explanation {}/{}'.format(s_idx + 1, n_local_instances))

        px_explainer = Explainer.initialize(px_model, instance=x_valid.loc[v_idx])
        ## direct explanation
        time_print('direct explanation')
        # direct could be tuple of tuples of condition strings
        start_direct_gen = timer()
        direct = px_explainer.direct_reason()
        end_direct_gen = timer()
        le_direct_time_taken.append(end_direct_gen - start_direct_gen)
        direct_rules = []
        if type(direct[0]) == tuple:
            for dr in direct:
                direct_rules.append(pyxai_tuple_to_rule(dr, px_explainer, px_explainer.predict(x_valid.loc[v_idx])))
        else:
            direct_rules.append(pyxai_tuple_to_rule(direct, px_explainer, px_explainer.predict(x_valid.loc[v_idx])))

        # make a classifier
        dr_rule_classifiers = [RuleClassifier([dr], default_class=0) for dr in direct_rules]

        direct_scores = []
        direct_local_rules = []
        for drc_idx, dr_rule_classifier in enumerate(dr_rule_classifiers):
            rule_pred_idx = dr_rule_classifier.predict_index(sampled_x_valid)

            # coverage
            cov = rule_pred_idx.shape[0] / float(sample_idx.shape[0])
            # precision
            prc = np.mean((dt.predict(sampled_x_valid.loc[rule_pred_idx]) > 0.5).astype(int) ==
                          (dt.predict(x_valid.loc[[v_idx]]) > 0.5).astype(int))

            rule_pred_metrics = {'local_coverage': cov, 'local_precision': prc}
            direct_scores.append((drc_idx, rule_pred_metrics))
            direct_local_rules.append((drc_idx, [f'{r.rule_str}' for r in dr_rule_classifier.rules]))
        le_direct_rule_store[s_idx] = direct_local_rules
        le_direct_score_store[s_idx] = direct_scores

        time_print('sufficient explanation')
        ## sufficient explanation
        # sufficient could be tuple of tuples of condition strings
        start_sufficient_gen = timer()
        sufficient = px_explainer.sufficient_reason()
        end_sufficent_gen = timer()
        le_sufficient_time_taken.append(end_sufficent_gen - start_sufficient_gen)
        sufficient_rules = []
        if type(sufficient[0]) == tuple:
            for sr in sufficient:
                sufficient_rules.append(pyxai_tuple_to_rule(sr, px_explainer, px_explainer.predict(x_valid.loc[v_idx])))
        else:
            sufficient_rules.append(pyxai_tuple_to_rule(sufficient, px_explainer, px_explainer.predict(x_valid.loc[v_idx])))

        # make a classifier
        sr_rule_classifiers = [RuleClassifier([sr], default_class=0) for sr in sufficient_rules]

        sufficient_scores = []
        sufficient_local_rules = []
        for src_idx, sr_rule_classifier in enumerate(sr_rule_classifiers):
            rule_pred_idx = sr_rule_classifier.predict_index(sampled_x_valid)

            # coverage
            cov = rule_pred_idx.shape[0] / float(sample_idx.shape[0])
            # precision
            prc = np.mean((dt.predict(sampled_x_valid.loc[rule_pred_idx]) > 0.5).astype(int) ==
                          (dt.predict(x_valid.loc[[v_idx]]) > 0.5).astype(int))

            rule_pred_metrics = {'local_coverage': cov, 'local_precision': prc}
            sufficient_scores.append((src_idx, rule_pred_metrics))
            sufficient_local_rules.append((src_idx, [f'{r.rule_str}' for r in sr_rule_classifier.rules]))
        le_sufficient_rule_store[s_idx] = sufficient_local_rules
        le_sufficient_score_store[s_idx] = sufficient_scores

    le_end = timer()
    time_print('local explanation completed {} seconds | {} from start'.format(round(le_end - le_start),
                                                                               round(le_end - start)))
    le_out_dict = {
        # experiment
        'model': 'DecisionTree',
        'experiment': experiment_tag,
        'dataset': dataset_name,
        'max_depth': hyperparams['max_depth'],
        'hyperparams': hyperparams,
        # timer
        'py_total_time': le_end - start,
        'py_dt_time': dt_end - dt_start,
        # 'py_ext_time': ext_end - ext_start,
        'py_local_explanation_time': le_end - le_start,
        'py_pyxai_direct_time_array': le_direct_time_taken,
        'py_pyxai_direct_time': np.sum(le_direct_time_taken),
        'py_pyxai_sufficient_time_array': le_sufficient_time_taken,
        'py_pyxai_sufficient_time': np.sum(le_sufficient_time_taken),
        # metrics
        'fold': fold,
        'vanilla_metrics': vanilla_metrics,
        'local_explanation_direct_scores': le_direct_score_store,
        'local_explanation_direct_rules': le_direct_rule_store,
        'local_explanation_sufficient_scores': le_sufficient_score_store,
        'local_explanation_sufficient_rules': le_sufficient_rule_store
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
