import numpy as np
import lightgbm as lgb
import json
import os
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from category_encoders.one_hot import OneHotEncoder
from anchor import anchor_tabular
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer

from hyperparameter import optuna_lgb, optuna_random_forest, optuna_decision_tree
from utils import load_data, time_print


SEED = 2020


def anchor_explain_single(row, explainer, model, threshold=.95, random_state=SEED):
    np.random.seed(random_state)
    ae = explainer.explain_instance(row, model.predict, threshold=threshold)
    return ae


def run_experiment(dataset_name):
    exp_dir = './tmp/journal/local'
    log_json = os.path.join(exp_dir, 'anchor.json')
    anchor_n_instances = 100

    X, y = load_data(dataset_name)
    # one hot encoded X for random forest
    cat_X = None
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        cat_X = oh.fit_transform(X)
        # avoid LightGBM Special character JSON error
        cat_X = cat_X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # multilabel case
    num_classes = y.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        time_print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        if cat_X is not None:
            x_train = cat_X.iloc[train_idx]
            x_valid = cat_X.iloc[valid_idx]

        dt_start = timer()
        time_print('dt optuna start...')
        dt_best_params = optuna_decision_tree(x_train, y_train)
        dt_optuna_end = timer()
        dt = DecisionTreeClassifier(**dt_best_params, random_state=SEED)
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_valid)
        dt_fit_predict_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        time_print('dt fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}

        dt_anchor_start = timer()
        time_print('dt anchor start...')
        explainer = anchor_tabular.AnchorTabularExplainer(y.unique(), x_train.columns,
                                                          x_train.to_numpy())
        dt_train_covs = []
        dt_train_prcs = []
        dt_valid_covs = []
        dt_valid_prcs = []

        mat_x_valid = x_valid.sample(anchor_n_instances, replace=True).to_numpy()
        for v_idx in range(mat_x_valid.shape[0]):
            if ((v_idx+1) % 10) == 0:
                time_print('dt anchor fold {} {}/{}'.format(f_idx+1, v_idx+1, mat_x_valid.shape[0]))
            ae = anchor_explain_single(mat_x_valid[v_idx, :], explainer, dt)
            fit_anchor = np.where(np.all(mat_x_valid[:, ae.features()] == mat_x_valid[v_idx][ae.features()], axis=1))[0]
            cov = fit_anchor.shape[0] / float(mat_x_valid.shape[0])
            prc = np.mean(dt.predict(mat_x_valid[fit_anchor]) == dt.predict(mat_x_valid[v_idx].reshape(1, -1)))
            dt_train_covs.append(ae.coverage())
            dt_train_prcs.append(ae.precision())
            dt_valid_covs.append(cov)
            dt_valid_prcs.append(prc)

        dt_anchor_end = timer()

        anchor_metrics = {'anchor_avg_train_precision': np.mean(dt_train_prcs),
                          'anchor_avg_train_coverage' : np.mean(dt_train_covs),
                          'anchor_avg_valid_precision': np.mean(dt_valid_prcs),
                          'anchor_avg_valid_coverage' : np.mean(dt_valid_covs)}

        dt_end = timer()

        dt_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'DecisionTree',
            'dt.best_params': dt_best_params,
            'vanilla_metrics': vanilla_metrics,
            'anchor_metrics': anchor_metrics,
            'total_time': dt_end - dt_start,
            'optuna_time': dt_optuna_end - dt_start,
            'fit_predict_time': dt_fit_predict_end - dt_optuna_end,
            'anchor_time': dt_anchor_end - dt_anchor_start,
        }

        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(dt_dict) + '\n')

        rf_start = timer()
        time_print('rf optuna start...')
        rf_best_params = optuna_random_forest(x_train, y_train)
        rf_optuna_end = timer()
        rf = RandomForestClassifier(**rf_best_params, n_jobs=-1, random_state=SEED)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        rf_fit_predict_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        time_print('rf fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}

        rf_anchor_start = timer()
        time_print('rf anchor start...')
        explainer = anchor_tabular.AnchorTabularExplainer(y.unique(), x_train.columns,
                                                          x_train.to_numpy())
        rf_train_covs = []
        rf_train_prcs = []
        rf_valid_covs = []
        rf_valid_prcs = []

        mat_x_valid = x_valid.sample(anchor_n_instances, replace=True).to_numpy()
        for v_idx in range(mat_x_valid.shape[0]):
            if ((v_idx+1) % 10) == 0:
                time_print('rf anchor fold {} {}/{}'.format(f_idx+1, v_idx+1, mat_x_valid.shape[0]))
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

        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(rf_dict) + '\n')

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
        time_print('lgb fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy':  accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall':    recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1':        f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc':       roc_auc_score(y_valid, y_pred)}

        lgb_anchor_start = timer()
        time_print('lgb anchor start...')
        explainer = anchor_tabular.AnchorTabularExplainer(y.unique(), x_train.columns,
                                                          x_train.to_numpy())
        lgb_train_covs = []
        lgb_train_prcs = []
        lgb_valid_covs = []
        lgb_valid_prcs = []

        mat_x_valid = x_valid.sample(anchor_n_instances, replace=True).to_numpy()
        for v_idx in range(mat_x_valid.shape[0]):
            if ((v_idx+1) % 10) == 0:
                time_print('lgb anchor fold {} {}/{}'.format(f_idx+1, v_idx+1, mat_x_valid.shape[0]))
            # if you don't make a copy, lightgbm complains about memory issue
            v_cp = np.copy(mat_x_valid[v_idx,:].reshape(1,-1))
            ae = anchor_explain_single(v_cp, explainer, lgb_model)
            fit_anchor = np.where(np.all(mat_x_valid[:, ae.features()] == mat_x_valid[v_idx][ae.features()], axis=1))[0]
            v_fac_cp = np.copy(mat_x_valid[fit_anchor])
            cov = fit_anchor.shape[0] / float(mat_x_valid.shape[0])
            prc = np.mean(lgb_model.predict(v_fac_cp) == lgb_model.predict(v_cp))
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

        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(lgb_dict) + '\n')


if __name__ == '__main__':
    for data in [
        'autism',
        # 'breast',
        # 'cars',
        # 'credit_australia',
        # 'heart',
        # 'ionosphere',
        # 'kidney',
        # 'krvskp',
        # 'voting',
        # 'census',
        #  # 'airline',
        #  # 'eeg',
        #  # 'kdd99',
        # 'synthetic_1',
        # 'credit_taiwan'
        # 'credit_german',
        # 'adult',
        # 'compas'
    ]:
        time_print('='*40 + data + '='*40)
        run_experiment(data)
