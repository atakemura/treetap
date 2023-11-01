import pandas as pd
import numpy as np
import os
import lightgbm as lgb
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from rulefit import RuleFit
from category_encoders.one_hot import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from psutil import cpu_count

from hyperparameter import optuna_lgb, optuna_random_forest, optuna_rulefit, optuna_decision_tree
from utils import load_data
from tree_asp.utils import time_print


SEED = 2020
NUM_CPU = cpu_count(logical=False)


def run_experiment(dataset_name):
    exp_dir = 'tree_asp/tmp/journal/benchmark'
    log_json = os.path.join(exp_dir, 'baseline.json')

    X, y = load_data(dataset_name)
    # one hot encoded X for random forest
    cat_X = None
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        cat_X = oh.fit_transform(X)

    # multilabel case
    num_classes = y.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        time_print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        dt_start = timer()
        time_print('dt optuna start...')
        if cat_X is not None:
            dt_best_params = optuna_decision_tree(cat_X.iloc[train_idx], y_train)
            dt_optuna_end = timer()
            dt_fit_start = timer()
            dt = DecisionTreeClassifier(**dt_best_params, random_state=SEED)
            dt.fit(cat_X.iloc[train_idx], y_train)
            dt_fit_end = timer()
            y_pred = dt.predict(cat_X.iloc[valid_idx])
        else:
            dt_best_params = optuna_decision_tree(x_train, y_train)
            dt_optuna_end = timer()
            dt_fit_start = timer()
            dt = DecisionTreeClassifier(**dt_best_params, random_state=SEED)
            dt.fit(x_train, y_train)
            dt_fit_end = timer()
            y_pred = dt.predict(x_valid)
        dt_end = timer()
        f1 = f1_score(y_valid, y_pred)
        time_print('dt fold {} f1_score {}'.format(f_idx+1, round(f1, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}
        dt_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'DecisionTree',
            'dt.best_params': dt_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': dt_end - dt_start,
            'optuna_time': dt_optuna_end - dt_start,
            'fit_excluding_optuna_time': dt_fit_end - dt_fit_start,
            'fit_predict_time': dt_end - dt_fit_start
        }

        rf_start = timer()
        time_print('rf optuna start...')
        if cat_X is not None:
            rf_best_params = optuna_random_forest(cat_X.iloc[train_idx], y_train)
            rf_optuna_end = timer()
            rf_fit_start = timer()
            rf = RandomForestClassifier(**rf_best_params, n_jobs=NUM_CPU, random_state=SEED)
            rf.fit(cat_X.iloc[train_idx], y_train)
            rf_fit_end = timer()
            y_pred = rf.predict(cat_X.iloc[valid_idx])
        else:
            rf_best_params = optuna_random_forest(x_train, y_train)
            rf_optuna_end = timer()
            rf_fit_start = timer()
            rf = RandomForestClassifier(**rf_best_params, n_jobs=NUM_CPU, random_state=SEED)
            rf.fit(x_train, y_train)
            rf_fit_end = timer()
            y_pred = rf.predict(x_valid)
        rf_end = timer()
        f1 = f1_score(y_valid, y_pred)
        rf_vanilla_pred = y_pred  # copy for using in fidelity calculation later
        time_print('rf fold {} f1_score {}'.format(f_idx+1, round(f1, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}
        rf_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'RandomForest',
            'rf.best_params': rf_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': rf_end - rf_start,
            'optuna_time': rf_optuna_end - rf_start,
            'fit_excluding_optuna_time': rf_fit_end - rf_fit_start,
            'fit_predict_time': rf_end - rf_fit_start
        }

        lgb_start = timer()
        time_print('lgb optuna start...')
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
        lgb_best_params = optuna_lgb(x_train, y_train, static_params)
        lgb_optuna_end = timer()
        lgb_fit_start = timer()
        lgb_hyperparams = {**static_params, **lgb_best_params}
        lgb_model = lgb.train(params=lgb_hyperparams,
                              train_set=lgb_train,
                              valid_sets=[lgb_valid],
                              valid_names=['valid'],
                              num_boost_round=1000,
                              callbacks=[lgb.callback.early_stopping(50, verbose=False)])
        lgb_fit_end = timer()
        if num_classes > 2:
            y_pred = np.argmax(lgb_model.predict(x_valid), axis=1)
        else:
            y_pred = (lgb_model.predict(x_valid) > 0.5).astype(int)
        lgb_end = timer()
        f1 = f1_score(y_valid, y_pred)
        time_print('lgb fold {} f1_score {}'.format(f_idx+1, round(f1, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}
        lgb_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'LightGBM',
            'lgb.best_params': lgb_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': lgb_end - lgb_start,
            'optuna_time': lgb_optuna_end - lgb_start,
            'fit_excluding_optuna_time': lgb_fit_end - lgb_fit_start,
            'fit_predict_time': lgb_end - lgb_fit_start
        }

        rfit_start = timer()
        time_print('rule fit start...')
        if cat_X is not None:
            rfit_best_params = optuna_rulefit(cat_X.iloc[train_idx], y_train, rf_params=rf_best_params)
            rfit_optuna_end = timer()
            rfit_fit_start = timer()
            rf = RandomForestClassifier(n_jobs=1, random_state=SEED, **rf_best_params)
            rfit = RuleFit(**rfit_best_params, tree_generator=rf, rfmode='classify', n_jobs=NUM_CPU, random_state=SEED)
            rfit.fit(cat_X.iloc[train_idx], y_train, feature_names=cat_X.columns)
            rfit_fit_end = timer()
            try:
                y_pred = rfit.predict(cat_X.iloc[valid_idx])
            except IndexError:
                y_pred = None
        else:
            rfit_best_params = optuna_rulefit(x_train, y_train, rf_params=rf_best_params)
            rfit_optuna_end = timer()
            rfit_fit_start = timer()
            rf = RandomForestClassifier(n_jobs=1, random_state=SEED, **rf_best_params)
            rfit = RuleFit(**rfit_best_params, tree_generator=rf, rfmode='classify', n_jobs=NUM_CPU, random_state=SEED)
            rfit.fit(x_train, y_train, feature_names=x_train.columns)
            rfit_fit_end = timer()
            try:
                y_pred = rfit.predict(x_valid)
            except IndexError:
                y_pred = None
        rfit_end = timer()
        if y_pred is None:  # RuleFit failed to find any rules in decision trees
            rfit_dict = {
                'dataset': dataset_name,
                'fold': f_idx,
                'model': 'RuleFit',
                'rfit.best_20_rules_support': 'FAILED',
                'rfit.n_rules': 'FAILED',
                'rfit.best_params': 'None',
                'vanilla_metrics': 0,
                'fidelity_metrics': 0,
                'total_time': rfit_end - rfit_start,
                'optuna_time': rfit_optuna_end - rfit_start,
                'fit_excluding_optuna_time': rfit_fit_end - rfit_fit_start,
                'fit_predict_time': rfit_end - rfit_fit_start
            }
        else:  # success
            f1 = f1_score(y_valid, y_pred)
            time_print('rfit fold {} f1_score {}'.format(f_idx+1, round(f1, 2)))
            vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                               'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                               'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                               'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                               'auc':       roc_auc_score(y_valid, y_pred)}
            # RuleFit fidelity metrics
            fidelity_metrics = {'accuracy': accuracy_score(rf_vanilla_pred, y_pred),
                                'precision': precision_score(rf_vanilla_pred, y_pred, average=metric_averaging),
                                'recall': recall_score(rf_vanilla_pred, y_pred, average=metric_averaging),
                                'f1': f1_score(rf_vanilla_pred, y_pred, average=metric_averaging),
                                'auc': roc_auc_score(rf_vanilla_pred, y_pred)}

            rules = rfit.get_rules()
            rules = rules[rules.coef != 0].sort_values('support', ascending=False)
            n_rules = rules.shape[0]
            top_rules = rules.head(20)  # type: pd.DataFrame
            rfit_dict = {
                'dataset': dataset_name,
                'fold': f_idx,
                'model': 'RuleFit',
                'rfit.best_20_rules_support': top_rules.to_json(orient='records'),
                'rfit.n_rules': n_rules,
                'rfit.best_params': rfit_best_params,
                'vanilla_metrics': vanilla_metrics,
                'fidelity_metrics': fidelity_metrics,
                'total_time': rfit_end - rfit_start,
                'optuna_time': rfit_optuna_end - rfit_start,
                'fit_excluding_optuna_time': rfit_fit_end - rfit_fit_start,
                'fit_predict_time': rfit_end - rfit_fit_start
            }
            # saving rule table to csv
            rfit_fname = os.path.join(exp_dir, f'rulefit_{dataset_name}_{f_idx}.csv')
            rules.to_csv(rfit_fname, index=False, header=True)

        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(dt_dict) + '\n')
            out_log_json.write(json.dumps(rf_dict) + '\n')
            out_log_json.write(json.dumps(lgb_dict) + '\n')
            out_log_json.write(json.dumps(rfit_dict) + '\n')


if __name__ == '__main__':
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
