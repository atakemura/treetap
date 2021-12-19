import pandas as pd
import numpy as np
import weka.core.jvm as jvm
import optuna
import json
import arff
import os

from weka.classifiers import Classifier
from weka.core.converters import load_any_file
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tempfile import NamedTemporaryFile
from timeit import default_timer as timer

from utils import load_data


class WekaJ48:
    def __init__(self, confidence=0.25, min_child_leaf=2, num_folds=3,
                 reduced_error_pruning=False, no_subtree_raising=False, binary_splits=False):
        self.model = None
        self.confidence = confidence
        self.min_child_leaf = min_child_leaf
        self.num_folds = num_folds
        self.reduced_error_pruning = reduced_error_pruning
        self.no_subtree_raising = no_subtree_raising
        self.binary_splits = binary_splits
        self.arff_attr = None

        options = [
            '-C', str(self.confidence),
            '-M', str(round(self.min_child_leaf)),
            '-N', str(round(self.num_folds))
        ]
        if self.reduced_error_pruning:
            options.append('-R')
        if self.no_subtree_raising:
            options.append('-S')
        if self.binary_splits:
            options.append('-B')

        self.model = Classifier(classname='weka.classifiers.trees.J48', options=options)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_arff = self.create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(train)
            # train_filtered = num2nominal.filter(train)
            # train_filtered.class_is_last()
            # self.model.build_classifier(train_filtered)
            self.model.build_classifier(train)
        except Exception as e:
            raise e
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        # weka assumes all examples are labeled even the test examples
        valid_arff = self.create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(valid)
            # valid_filtered = num2nominal.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except Exception as e:
            raise e
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)

    def create_temp_arff(self, df, description=''):
        # check all numeric
        assert len(df.select_dtypes(include=['float', 'int', 'category']).columns) == len(df.columns)
        # convert the last column to categorical
        df.iloc[:, -1] = df.iloc[:, -1].astype('category')
        # numeric = df.select_dtypes(include=['float', 'int']).columns
        categorical = df.select_dtypes(include=['category']).columns
        # numeric_attrs = [(x, 'NUMERIC') for x in numeric]
        # categorical_attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()]) for col in categorical]
        attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()])
                 if col in categorical
                 else (col, 'NUMERIC')
                 for col in df.columns]
        if not self.arff_attr:
            self.arff_attr = attrs
        else:
            attrs = self.arff_attr
        # trick to keep int representation
        df[categorical] = df[categorical].astype('O')
        data = list(df.values)
        result = {
            'attributes': attrs,
            'data': data,
            'description': description,
            'relation': 'data'
        }
        arff_dump = arff.dumps(result)
        tmp = NamedTemporaryFile(suffix='.arff', delete=False)
        try:
            with open(tmp.name, 'w') as fp:
                fp.write(arff_dump)
        except Exception as e:
            os.remove(tmp.name)
            raise e
        return tmp


class WekaRIPPER:
    def __init__(self, num_folds=3, prune=True, no_error_check=False, seed=2020):
        self.model = None
        self.num_folds = num_folds
        self.prune = prune
        self.no_check_error = no_error_check
        self.seed = seed
        self.arff_attr = None

        options = [
            '-F', str(round(self.num_folds)),
            '-S', str(self.seed)
        ]
        if not self.prune:
            options.append('-P')
        if self.no_check_error:
            options.append('-E')

        self.model = Classifier(classname='weka.classifiers.rules.JRip', options=options)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_arff = self.create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(train)
            # train_filtered = num2nominal.filter(train)
            # self.model.build_classifier(train_filtered)
            self.model.build_classifier(train)
        except Exception as e:
            raise e
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        valid_arff = self.create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(valid)
            # valid_filtered = num2nominal.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except Exception as e:
            raise e
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)

    def create_temp_arff(self, df, description=''):
        # check all numeric
        assert len(df.select_dtypes(include=['float', 'int', 'category']).columns) == len(df.columns)
        # convert the last column to categorical
        df.iloc[:, -1] = df.iloc[:, -1].astype('category')
        # numeric = df.select_dtypes(include=['float', 'int']).columns
        categorical = df.select_dtypes(include=['category']).columns
        # numeric_attrs = [(x, 'NUMERIC') for x in numeric]
        # categorical_attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()]) for col in categorical]
        attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()])
                 if col in categorical
                 else (col, 'NUMERIC')
                 for col in df.columns]
        if not self.arff_attr:
            self.arff_attr = attrs
        else:
            attrs = self.arff_attr
        # trick to keep int representation
        df[categorical] = df[categorical].astype('O')
        data = list(df.values)
        result = {
            'attributes': attrs,
            'data': data,
            'description': description,
            'relation': 'data'
        }
        arff_dump = arff.dumps(result)
        tmp = NamedTemporaryFile(suffix='.arff', delete=False)
        try:
            with open(tmp.name, 'w') as fp:
                fp.write(arff_dump)
        except Exception as e:
            os.remove(tmp.name)
            raise e
        return tmp


class WekaPART:
    def __init__(self, confidence=0.25, min_child_leaf=2, num_folds=3,
                 reduced_error_pruning=False, unpruned=False, no_mdl=False, binary_splits=False, seed=2020):
        self.model = None
        self.confidence = confidence
        self.min_child_leaf = min_child_leaf
        self.num_folds = num_folds
        self.reduced_error_pruning = reduced_error_pruning
        self.no_mdl = no_mdl
        self.unpruned = unpruned
        self.binary_splits = binary_splits
        self.seed = seed
        self.arff_attr = None

        options = [
            '-C', str(self.confidence),
            '-M', str(round(self.min_child_leaf)),
            '-N', str(round(self.num_folds)),
            '-Q', str(seed)
        ]
        if self.reduced_error_pruning:
            options.append('-R')
        if self.binary_splits:
            options.append('-B')
        if self.no_mdl:
            options.append('-J')
        if self.unpruned:
            options.append('-U')

        self.model = Classifier(classname='weka.classifiers.rules.PART', options=options)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_arff = self.create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(train)
            # train_filtered = num2nominal.filter(train)
            # self.model.build_classifier(train_filtered)
            self.model.build_classifier(train)
        except Exception as e:
            raise e
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        valid_arff = self.create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            # num2nominal = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal',
            #                      options=['-R', 'last'])
            # num2nominal.inputformat(valid)
            # valid_filtered = num2nominal.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except Exception as e:
            raise e
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)

    def create_temp_arff(self, df, description=''):
        # check all numeric
        assert len(df.select_dtypes(include=['float', 'int', 'category']).columns) == len(df.columns)
        # convert the last column to categorical
        df.iloc[:, -1] = df.iloc[:, -1].astype('category')
        # numeric = df.select_dtypes(include=['float', 'int']).columns
        categorical = df.select_dtypes(include=['category']).columns
        # numeric_attrs = [(x, 'NUMERIC') for x in numeric]
        # categorical_attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()]) for col in categorical]
        attrs = [(col, [str(x) for x in df[col].cat.categories.tolist()])
                 if col in categorical
                 else (col, 'NUMERIC')
                 for col in df.columns]
        if not self.arff_attr:
            self.arff_attr = attrs
        else:
            attrs = self.arff_attr
        # trick to keep int representation
        df[categorical] = df[categorical].astype('O')
        data = list(df.values)
        result = {
            'attributes': attrs,
            'data': data,
            'description': description,
            'relation': 'data'
        }
        arff_dump = arff.dumps(result)
        tmp = NamedTemporaryFile(suffix='.arff', delete=False)
        try:
            with open(tmp.name, 'w') as fp:
                fp.write(arff_dump)
        except Exception as e:
            os.remove(tmp.name)
            raise e
        return tmp


def optuna_weka_j48(X, y):
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
        # numeric: confidence, min_child, num_fold
        # boolean: reduced_error_pruning, no_subtree_raising, binary_splits
        reduced_error_pruning = trial.suggest_categorical('reduced_error_pruning', [True, False])
        no_subtree_raising = trial.suggest_categorical('no_subtree_raising', [True, False])
        binary_splits = trial.suggest_categorical('binary_splits', [True, False])
        min_child_leaf = trial.suggest_int('min_child_leaf', 2, 14, step=4)
        if reduced_error_pruning:
            num_folds = trial.suggest_int('num_folds', 2, 5, step=1)
            confidence = 0.25
        else:
            num_folds = 3
            confidence = trial.suggest_float('confidence', 0.05, 0.4, step=0.05)
        j48 = WekaJ48(confidence=confidence, min_child_leaf=min_child_leaf, num_folds=num_folds,
                      reduced_error_pruning=reduced_error_pruning, no_subtree_raising=no_subtree_raising,
                      binary_splits=binary_splits)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2020)
        j48.fit(x_train, y_train)
        y_pred = j48.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def optuna_weka_ripper(X, y):
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
        # numeric: num_fold
        # boolean: prune, no_error_check
        prune = trial.suggest_categorical('prune', [True, False])
        no_error_check = trial.suggest_categorical('no_error_check', [True, False])
        num_folds = trial.suggest_int('num_folds', 2, 5, step=1)
        ripper = WekaRIPPER(num_folds=num_folds, prune=prune, no_error_check=no_error_check)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2020)
        ripper.fit(x_train, y_train)
        y_pred = ripper.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def optuna_weka_part(X, y):
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
        # numeric: confidence, min_child_leaf, num_folds
        # boolean: reduced_error_pruning, unpruned, no_mdl, binary_splits
        reduced_error_pruning = trial.suggest_categorical('reduced_error_pruning', [True, False])
        no_mdl = trial.suggest_categorical('no_mdl', [True, False])
        binary_splits = trial.suggest_categorical('binary_splits', [True, False])
        min_child_leaf = trial.suggest_int('min_child_leaf', 2, 14, step=4)
        # if unpruned, cannot set confidence, cannot set reduced error
        # if reduced_error, cannot set confidence
        # if num_folds, reduced error must also be set
        if reduced_error_pruning:
            unpruned = False
            num_folds = trial.suggest_int('num_folds', 2, 5, step=1)
            confidence = 0.25
        else:
            unpruned = trial.suggest_categorical('unpruned', [True, False])
            if unpruned:
                num_folds = 3
                confidence = 0.25
            else:
                num_folds = 3
                confidence = trial.suggest_float('confidence', 0.05, 0.4, step=0.05)

        part = WekaPART(confidence=confidence, min_child_leaf=min_child_leaf, num_folds=num_folds,
                        reduced_error_pruning=reduced_error_pruning, unpruned=unpruned,
                        no_mdl=no_mdl, binary_splits=binary_splits)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2020)
        part.fit(x_train, y_train)
        y_pred = part.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def run_experiment(dataset_name):
    exp_dir = './tmp/journal/benchmark'
    log_json = os.path.join(exp_dir, 'weka.json')

    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    # if len(categorical_features) > 0:
    #     oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
    #     X = oh.fit_transform(X)
    feat = X.columns

    # multilabel case
    num_classes = y.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        j48_start = timer()
        j48_best_params = optuna_weka_j48(x_train, y_train)
        j48_optuna_end = timer()
        j48 = WekaJ48(**j48_best_params)
        j48.fit(x_train, y_train)
        y_pred = j48.predict(x_valid)
        j48_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('j48 fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}
        j48_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'WekaJ48',
            'j48.model': str(j48.model),
            'j48.model.graph': j48.model.graph,
            'j48.best_params': j48_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': j48_end - j48_start,
            'optuna_time': j48_optuna_end - j48_start,
            'fit_predict_time': j48_end - j48_optuna_end
        }

        ripper_start = timer()
        ripper_best_params = optuna_weka_ripper(x_train, y_train)
        ripper_optuna_end = timer()
        ripper = WekaRIPPER(**ripper_best_params)
        ripper.fit(x_train, y_train)
        y_pred = ripper.predict(x_valid)
        ripper_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('ripper fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}
        ripper_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'WekaRIPPER',
            'ripper.model': str(ripper.model),
            'ripper.best_params': ripper_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': ripper_end - ripper_start,
            'optuna_time': ripper_optuna_end - ripper_start,
            'fit_predict_time': ripper_end - ripper_optuna_end
        }

        part_start = timer()
        part_best_params = optuna_weka_part(x_train, y_train)
        part_optuna_end = timer()
        part = WekaPART(**part_best_params)
        part.fit(x_train, y_train)
        y_pred = part.predict(x_valid)
        part_end = timer()
        acc = accuracy_score(y_valid, y_pred)
        print('part fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        vanilla_metrics = {'accuracy': accuracy_score(y_valid, y_pred),
                           'precision': precision_score(y_valid, y_pred, average=metric_averaging),
                           'recall': recall_score(y_valid, y_pred, average=metric_averaging),
                           'f1': f1_score(y_valid, y_pred, average=metric_averaging),
                           'auc': roc_auc_score(y_valid, y_pred)}
        part_dict = {
            'dataset': dataset_name,
            'fold': f_idx,
            'model': 'WekaPART',
            'part.model': str(part.model),
            'part.best_params': part_best_params,
            'vanilla_metrics': vanilla_metrics,
            'total_time': part_end - part_start,
            'optuna_time': part_optuna_end - part_start,
            'fit_predict_time': part_end - part_optuna_end
        }

        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(j48_dict) + '\n')
            out_log_json.write(json.dumps(ripper_dict) + '\n')
            out_log_json.write(json.dumps(part_dict) + '\n')


if __name__ == '__main__':
    jvm.start()
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

    jvm.stop()
