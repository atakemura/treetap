# Random Forest
# LightGBM
# RuleFit

import pandas as pd
import numpy as np
import optuna

from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from rulefit import RuleFit
from category_encoders.one_hot import OneHotEncoder
import lightgbm as lgb

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from pathlib import Path
import os
from timeit import default_timer as timer


SEED = 2020


def optuna_random_forest(X, y):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}

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
        # numeric: n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf
        # choice: criterion(gini, entropy)
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500, 10),
                  'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'min_samples_split': trial.suggest_float('min_samples_split', 0.05, 0.5, step=0.01),
                  'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.05, 0.5, step=0.01),
                  'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5, step=0.01),
                  'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                  }
        rf = RandomForestClassifier(**params, random_state=SEED, n_jobs=1)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


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
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def optuna_rulefit(X, y, rf_params=None):
    early_stopping_dict = {'early_stopping_limit': 30,
                           'early_stop_count': 0,
                           'best_score': None}

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
        # numeric: tree_size, sample_fract, max_rules, memory_par,
        # bool: lin_standardise, lin_trim_quantile,
        params = {# 'tree_size': trial.suggest_int('tree_size', 50, 500, 10),
                  # 'sample_fract': trial.suggest_float('sample_fract', 0.01, 1.0, step=0.02),
                  # 'max_rules': trial.suggest_int('max_rules', 10, 100),
                  'memory_par': trial.suggest_float('memory_par', 0.0, 1.0, step=0.1),
                  'lin_standardise': trial.suggest_categorical('lin_standardise', [True, False]),
                  'lin_trim_quantile': trial.suggest_categorical('lin_trim_quantile', [True, False]),
        }
        rf = RandomForestClassifier(n_jobs=1, random_state=SEED, **rf_params)
        rfit = RuleFit(tree_generator=rf, max_rules=500, rfmode='classify', n_jobs=1, random_state=SEED, **params)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED)
        rfit.fit(x_train, y_train, feature_names=x_train.columns)
        try:
            y_pred = rfit.predict(x_valid)
        # this sometimes raises IndexError rulefit.py:281 res_[:,coefs!=0]=res
        except IndexError:
            return 0   # skip this trial
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def run_experiment(dataset_name):
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
        print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        rf_start = timer()
        print('rf optuna start...')
        if cat_X is not None:
            rf_best_params = optuna_random_forest(cat_X.iloc[train_idx], y_train)
            rf_optuna_end = timer()
            rf = RandomForestClassifier(**rf_best_params, n_jobs=-1, random_state=SEED)
            rf.fit(cat_X.iloc[train_idx], y_train)
            y_pred = rf.predict(cat_X.iloc[valid_idx])
        else:
            rf_best_params = optuna_random_forest(x_train, y_train)
            rf_optuna_end = timer()
            rf = RandomForestClassifier(**rf_best_params, n_jobs=-1, random_state=SEED)
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_valid)
        rf_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('rf fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}
        rf_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'RandomForest',
            # 'rf.model': str(rf.model),
            # 'rf.model.graph': rf.model.graph,
            'rf.best_params': rf_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': rf_end - rf_start,
            'optuna_time': rf_optuna_end - rf_start,
            'fit_predict_time': rf_end - rf_optuna_end
        }

        lgb_start = timer()
        print('lgb optuna start...')
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
        lgb_hyperparams = {**static_params, **lgb_best_params}
        lgb_model = lgb.train(params=lgb_hyperparams,
                              train_set=lgb_train,
                              valid_sets=[lgb_valid],
                              valid_names=['valid'], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=False)
        lgb_end = timer()
        if num_classes > 2:
            y_pred = np.argmax(lgb_model.predict(x_valid), axis=1)
        else:
            y_pred = (lgb_model.predict(x_valid) > 0.5).astype(int)
        lgb_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('lgb fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}
        lgb_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'LightGBM',
            # 'lgb.model': str(lgb.model),
            'lgb.best_params': lgb_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': lgb_end - lgb_start,
            'optuna_time': lgb_optuna_end - lgb_start,
            'fit_predict_time': lgb_end - lgb_optuna_end
        }

        rfit_start = timer()
        print('rule fit start...')
        if cat_X is not None:
            rfit_best_params = optuna_rulefit(cat_X.iloc[train_idx], y_train, rf_params=rf_best_params)
            rfit_optuna_end = timer()
            rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **rf_best_params)
            rfit = RuleFit(**rfit_best_params, tree_generator=rf, rfmode='classify', n_jobs=-1, random_state=SEED)
            rfit.fit(cat_X.iloc[train_idx], y_train, feature_names=cat_X.columns)
            try:
                y_pred = rfit.predict(cat_X.iloc[valid_idx])
            except IndexError:
                y_pred = None
        else:
            rfit_best_params = optuna_rulefit(x_train, y_train, rf_params=rf_best_params)
            rfit_optuna_end = timer()
            rf = RandomForestClassifier(n_jobs=-1, random_state=SEED, **rf_best_params)
            rfit = RuleFit(**rfit_best_params, tree_generator=rf, rfmode='classify', n_jobs=-1, random_state=SEED)
            rfit.fit(x_train, y_train, feature_names=x_train.columns)
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
                'total_time': rfit_end - rfit_start,
                'optuna_time': rfit_optuna_end - rfit_start,
                'fit_predict_time': rfit_end - rfit_optuna_end
            }
        else:  # success
            acc = accuracy_score(y_valid, y_pred)
            print('rfit fold {} acc {}'.format(f_idx+1, round(acc, 2)))
            vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                               'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                               'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                               'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                               'auc': roc_auc_score(y_valid, y_pred)}
            rules = rfit.get_rules()
            rules = rules[rules.coef != 0].sort_values('support', ascending=False)
            n_rules = rules.shape[0]
            top_rules = rules.head(20)  # type: pd.DataFrame
            rfit_dict = {
                'dataset': dataset_name,
                'fold': f_idx,
                'model': 'RuleFit',
                # 'rfit.model': str(rfit.model),
                'rfit.best_20_rules_support': top_rules.to_json(orient='records'),
                'rfit.n_rules': n_rules,
                'rfit.best_params': rfit_best_params,
                'vanilla_metrics': vanilla_metrics,
                'total_time': rfit_end - rfit_start,
                'optuna_time': rfit_optuna_end - rfit_start,
                'fit_predict_time': rfit_end - rfit_optuna_end
            }

        # exp_dir = '../tmp/experiment_benchmark'
        exp_dir = '../tmp/journal/benchmark'
        log_json = os.path.join(exp_dir, 'output.json')
        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(rf_dict) + '\n')
            out_log_json.write(json.dumps(lgb_dict) + '\n')
            out_log_json.write(json.dumps(rfit_dict) + '\n')


def load_data(dataset_name):
    # there is no categorical feature in these sklearn datasets.
    sklearn_data = {'iris': load_iris,
                    'breast_sk': load_breast_cancer,
                    'wine': load_wine}
    # the following contains a mix of categorical and numerical features.
    datasets = ['autism', 'breast', 'cars',
                'credit_australia', 'heart', 'ionosphere',
                'kidney', 'krvskp', 'voting', 'census', 'airline',
                'synthetic_1',
                'kdd99', 'eeg', 'credit_taiwan',
                'credit_german', 'adult', 'compas']
    if dataset_name in sklearn_data.keys():
        load_data_method = sklearn_data[dataset_name]
        data_obj = load_data_method()
        feat = data_obj['feature_names']
        data = data_obj['data']
        target = data_obj['target']

        df = pd.DataFrame(data, columns=feat).assign(target=target)
        X, y = df[feat], df['target']
        dataset = (X, y)
    elif dataset_name in datasets:
        dataset_dir = Path('../../datasets/datasets/') / dataset_name

        raw = pd.read_csv(Path(dataset_dir / dataset_name).with_suffix('.csv'))
        with open(dataset_dir / 'schema.json', 'r') as infile:
            schema = json.load(infile)
        for c in schema['categorical_columns']:
            raw.loc[:, c] = raw.loc[:, c].astype('category')
        raw_x = raw[[c for c in raw.columns if c != schema['label_column']]].copy()
        if schema['id_col'] != "":
            raw_x.drop(schema['id_col'], axis=1, inplace=True)
        raw_y = raw[schema['label_column']]
        dataset = (raw_x, raw_y)
    else:
        raise ValueError('unrecognized dataset name: {}'.format(dataset_name))
    return dataset


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
        # 'airline',
        # 'eeg',
        # 'kdd99',
        'synthetic_1',
        'credit_taiwan',
        'credit_german',
        'adult',
        'compas'
    ]:
        print('='*40, data, '='*40)
        run_experiment(data)
