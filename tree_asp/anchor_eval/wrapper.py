import pandas as pd
import numpy as np
import optuna

from sklearn.datasets import load_wine, load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from category_encoders.one_hot import OneHotEncoder
import lightgbm as lgb
from anchor import anchor_tabular
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from pathlib import Path
import os
from timeit import default_timer as timer
from pprint import pprint


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


def anchor_explain_single(row, explainer, model, threshold=.95, random_state=SEED):
    np.random.seed(random_state)
    ae = explainer.explain_instance(row, model.predict, threshold=threshold)
    return ae


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

        if cat_X is not None:
            x_train = cat_X.iloc[train_idx]
            x_valid = cat_X.iloc[valid_idx]

        rf_start = timer()
        print('rf optuna start...')
        rf_best_params = optuna_random_forest(x_train, y_train)
        rf_optuna_end = timer()
        rf = RandomForestClassifier(**rf_best_params, n_jobs=-1, random_state=SEED)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        rf_fit_predict_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('rf fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}

        rf_anchor_start = timer()
        print('rf anchor start...')
        explainer = anchor_tabular.AnchorTabularExplainer(y.unique(), x_train.columns,
                                                          x_train.to_numpy())
        rf_train_covs = []
        rf_train_prcs = []
        rf_valid_covs = []
        rf_valid_prcs = []

        mat_x_valid = x_valid.sample(100, replace=True).to_numpy()
        for v_idx in range(mat_x_valid.shape[0]):
            ae = anchor_explain_single(mat_x_valid[v_idx, :], explainer, rf)
            fit_anchor = np.where(np.all(mat_x_valid[:, ae.features()] == mat_x_valid[v_idx][ae.features()], axis=1))[0]
            cov = fit_anchor.shape[0] / float(mat_x_valid.shape[0])
            prc = np.mean(rf.predict(mat_x_valid[fit_anchor]) == rf.predict(mat_x_valid[v_idx].reshape(1, -1)))
            rf_train_covs.append(ae.coverage())
            rf_train_prcs.append(ae.precision())
            rf_valid_covs.append(cov)
            rf_valid_prcs.append(prc)

        rf_anchor_end = timer()

        anchor_metrics = {'anchor_avg_train_precision': np.mean(rf_train_prcs),
                          'anchor_avg_train_coverage' : np.mean(rf_train_covs),
                          'anchor_avg_valid_precision': np.mean(rf_valid_prcs),
                          'anchor_avg_valid_coverage' : np.mean(rf_valid_covs)}

        rf_end = timer()

        rf_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'RandomForest',
            # 'rf.model': str(rf.model),
            # 'rf.model.graph': rf.model.graph,
            'rf.best_params': rf_best_params,
            'vanilla_metrics': vanilla_metrics,
            'anchor_metrics': anchor_metrics,
            'total_time': rf_end - rf_start,
            'optuna_time': rf_optuna_end - rf_start,
            'fit_predict_time': rf_fit_predict_end - rf_optuna_end,
            'anchor_time': rf_anchor_end - rf_anchor_start,
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
        if num_classes > 2:
            y_pred = np.argmax(lgb_model.predict(x_valid), axis=1)
        else:
            y_pred = (lgb_model.predict(x_valid) > 0.5).astype(int)
        lgb_fit_predict_end = timer()

        acc = accuracy_score(y_valid, y_pred)
        print('lgb fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}

        lgb_anchor_start = timer()
        print('lgb anchor start...')
        explainer = anchor_tabular.AnchorTabularExplainer(y.unique(), x_train.columns,
                                                          x_train.to_numpy())
        lgb_train_covs = []
        lgb_train_prcs = []
        lgb_valid_covs = []
        lgb_valid_prcs = []

        mat_x_valid = x_valid.sample(100, replace=True).to_numpy()
        for v_idx in range(mat_x_valid.shape[0]):
            ae = anchor_explain_single(mat_x_valid[v_idx, :], explainer, lgb_model)
            fit_anchor = np.where(np.all(mat_x_valid[:, ae.features()] == mat_x_valid[v_idx][ae.features()], axis=1))[0]
            cov = fit_anchor.shape[0] / float(mat_x_valid.shape[0])
            prc = np.mean(lgb_model.predict(mat_x_valid[fit_anchor]) == lgb_model.predict(mat_x_valid[v_idx].reshape(1, -1)))
            lgb_train_covs.append(ae.coverage())
            lgb_train_prcs.append(ae.precision())
            lgb_valid_covs.append(cov)
            lgb_valid_prcs.append(prc)

        lgb_anchor_end = timer()
        lgb_end = timer()

        anchor_metrics = {'anchor_avg_train_precision': np.mean(lgb_train_prcs),
                          'anchor_avg_train_coverage' : np.mean(lgb_train_covs),
                          'anchor_avg_valid_precision': np.mean(lgb_valid_prcs),
                          'anchor_avg_valid_coverage' : np.mean(lgb_valid_covs)}

        lgb_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'LightGBM',
            # 'lgb.model': str(lgb.model),
            'lgb.best_params': lgb_best_params,
            'vanilla_metrics': vanilla_metrics,
            'anchor_metrics': anchor_metrics,
            'total_time': lgb_end - lgb_start,
            'optuna_time': lgb_optuna_end - lgb_start,
            'fit_predict_time': lgb_fit_predict_end - lgb_optuna_end,
            'anchor_time': lgb_anchor_end - lgb_anchor_start
        }

        exp_dir = '../tmp/journal/benchmark'
        log_json = os.path.join(exp_dir, 'anchor.json')
        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(rf_dict) + '\n')
            out_log_json.write(json.dumps(lgb_dict) + '\n')


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
                'kdd99', 'eeg', 'credit_taiwan']
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
                 'autism', 'breast',
                 'cars',
                 'credit_australia', 'heart',
                 'ionosphere', 'kidney', 'krvskp', 'voting',
                 'census',
                 # 'airline',
                 # 'eeg',
                 # 'kdd99',
                 'synthetic_1',
                 'credit_taiwan'
                 ]:
        print('='*40, data, '='*40)
        run_experiment(data)
