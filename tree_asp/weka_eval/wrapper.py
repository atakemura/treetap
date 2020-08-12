import pandas as pd
import numpy as np
import weka.core.jvm as jvm

from weka.classifiers import Classifier
from weka.core.converters import load_any_file
from weka.filters import Filter

from sklearn.metrics import accuracy_score
import json
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from pathlib import Path
import arff
import os
from tempfile import NamedTemporaryFile


class WekaJ48:
    def __init__(self, confidence=0.25, min_child_leaf=2, num_folds=3,
                 reduced_error_pruning = False, no_subtree_raising = False, binary_splits = False):
        self.model = None
        self.confidence = confidence
        self.min_child_leaf = min_child_leaf
        self.num_folds = num_folds
        self.reduced_error_pruning = reduced_error_pruning
        self.no_subtree_raising = no_subtree_raising
        self.binary_splits = binary_splits

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
        train_arff = create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(train)
            train_filtered = ntob.filter(train)
            self.model.build_classifier(train_filtered)
        except:
            raise
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        valid_arff = create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(valid)
            valid_filtered = ntob.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid_filtered):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except:
            raise
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)


class WekaRIPPER:
    def __init__(self, num_folds=3, prune=True, no_error_check=False, seed=2020):
        self.model = None
        self.num_folds = num_folds
        self.prune = prune
        self.no_check_error = no_error_check
        self.seed = seed

        options = [
            '-F', str(round(self.num_folds)),
            '-S', str(self.seed)
        ]
        if self.prune:
            options.append('-P')
        if self.no_check_error:
            options.append('-E')

        self.model = Classifier(classname='weka.classifiers.rules.JRip', options=options)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        train_arff = create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(train)
            train_filtered = ntob.filter(train)
            self.model.build_classifier(train_filtered)
        except:
            raise
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        valid_arff = create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(valid)
            valid_filtered = ntob.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid_filtered):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except:
            raise
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)


class WekaPART:
    def __init__(self, confidence=0.25, min_child_leaf=2, num_folds=3,
                 reduced_error_pruning = False, unpruned = False, no_mdl = False, binary_splits = False, seed=2020):
        self.model = None
        self.confidence = confidence
        self.min_child_leaf = min_child_leaf
        self.num_folds = num_folds
        self.reduced_error_pruning = reduced_error_pruning
        self.no_mdl = no_mdl
        self.unpruned = unpruned
        self.binary_splits = binary_splits
        self.seed = seed

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
        train_arff = create_temp_arff(pd.concat([X, y], axis=1), 'train')
        try:
            train = load_any_file(train_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(train)
            train_filtered = ntob.filter(train)
            self.model.build_classifier(train_filtered)
        except:
            raise
        finally:
            os.remove(train_arff.name)
        return

    def predict(self, X: pd.DataFrame, proba=False, **kwargs):
        _x_valid = X.copy()
        _x_valid['label'] = 0
        valid_arff = create_temp_arff(_x_valid, 'valid')
        try:
            valid = load_any_file(valid_arff.name, class_index='last')
            ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
            ntob.inputformat(valid)
            valid_filtered = ntob.filter(valid)
            pred_array = []
            dist_array = []
            for idx, inst in enumerate(valid_filtered):
                pred = self.model.classify_instance(inst)
                pred_array.append(pred)
                dist = self.model.distribution_for_instance(inst)
                dist_array.append(dist)
        except:
            raise
        finally:
            os.remove(valid_arff.name)
        if proba:
            return np.array(dist_array)
        else:
            return np.array(pred_array)



def create_temp_arff(df, description=''):
    # check all numeric
    assert all(df.select_dtypes(include=['float', 'int']).columns == df.columns)
    attrs = [(x, 'NUMERIC') for x in df.columns]
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
    except:
        os.remove(tmp.name)
        raise
    return tmp


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        j48 = WekaJ48()
        j48.fit(x_train, y_train)
        y_pred = j48.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print('j48 fold {} acc {}'.format(f_idx+1, round(acc, 2)))

        ripper = WekaRIPPER()
        ripper.fit(x_train, y_train)
        y_pred = ripper.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print('ripper fold {} acc {}'.format(f_idx+1, round(acc, 2)))

        part = WekaPART()
        part.fit(x_train, y_train)
        y_pred = part.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print('part fold {} acc {}'.format(f_idx+1, round(acc, 2)))
        # train_arff = create_temp_arff(pd.concat([x_train, y_train], axis=1), 'train')
        # valid_arff = create_temp_arff(pd.concat([x_valid, y_valid], axis=1), 'valid')
        # try:
        #     train = load_any_file(train_arff.name, class_index='last')
        #     valid = load_any_file(valid_arff.name, class_index='last')
        #     print([a.type_str() for a in train.attributes()])
        #
        #     ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
        #     ntob.inputformat(train)
        #
        #     train_filtered = ntob.filter(train)
        #     valid_filtered = ntob.filter(valid)
        #
        #     cls = Classifier(classname='weka.classifiers.trees.J48',
        #                      options=['-C', '0.3']
        #                      )
        #     cls.build_classifier(train_filtered)
        #     print(cls)
        #
        #     for idx, inst in enumerate(valid_filtered):
        #         pred = cls.classify_instance(inst)
        #         dist = cls.distribution_for_instance(inst)
        # except:
        #     raise
        # finally:
        #     os.remove(train_arff.name)
        #     os.remove(valid_arff.name)


def load_data(dataset_name):
    # there is no categorical feature in these sklearn datasets.
    sklearn_data = {'iris': load_iris,
                    'breast_sk': load_breast_cancer,
                    'wine': load_wine}
    # the following contains a mix of categorical and numerical features.
    datasets = ['autism', 'breast', 'cars',
                'credit_australia', 'heart', 'ionosphere',
                'kidney', 'krvskp', 'voting']
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
    jvm.start()
    """
    # data = pd.read_csv('../../datasets/datasets/heart/heart.csv')
    # x = data[[x for x in data.columns if x != 'label']].values
    # y = data['label'].values
    # dataset = create_instances_from_matrices(x, y, name='from matrix')

    dataset = load_any_file('../../datasets/datasets/autism/autism.csv', class_index='last')
    print(dataset.attribute_names(), dataset.num_instances, dataset.class_attribute)
    print([a.type_str() for a in dataset.attributes()])

    ntob = Filter(classname='weka.filters.unsupervised.attribute.NumericToNominal', options=['-R', 'last'])
    ntob.inputformat(dataset)

    filtered = ntob.filter(dataset)

    cls = Classifier(classname='weka.classifiers.trees.J48', options=['-U'])
    cls.build_classifier(filtered)
    print(cls)

    for idx, inst in enumerate(filtered):
        pred = cls.classify_instance(inst)
        dist = cls.distribution_for_instance(inst)
        # print('{}: label_index={} class_distribution={}'.format(idx,pred,dist))
    """

    dataset_name = 'autism'
    run_experiment(dataset_name)

    jvm.stop()