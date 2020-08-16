import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
import json
from category_encoders.one_hot import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from pathlib import Path

from corels import CorelsClassifier

import sys
sys.path.append('../mdlp')

from mdlp.mdlp import MDLPDiscretizer

# note all data must be binary, including features and labels


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    numerical_features = list(X.select_dtypes(include=['float', 'int']).columns)
    if len(numerical_features) > 0:
        mdlp = MDLPDiscretizer(features=numerical_features)
        Xz = mdlp.fit_transform(X, y)  # np.ndarray
        X = pd.DataFrame(data=Xz, columns=X.columns)
        # one hot encoder cannot handle intervals so treat them as strings
        for col in numerical_features:
            X.loc[:, col] = X.loc[:, col].astype('str').astype('category')
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
    feat = X.columns
    print(X.head(1))

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        print('fold={}'.format(f_idx+1))
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        corels = CorelsClassifier(n_iter=10000)
        corels.fit(x_train, y_train, features=list(x_train.columns))
        y_pred = corels.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        print('corels fold {} acc {}'.format(f_idx+1, round(acc, 2)))

def load_data(dataset_name):
    # there is no categorical feature in these sklearn datasets.
    sklearn_data = {'iris': load_iris,
                    'breast_sk': load_breast_cancer,
                    'wine': load_wine}
    # the following contains a mix of categorical and numerical features.
    datasets = ['autism', 'breast', 'cars',
                'credit_australia', 'heart', 'ionosphere',
                'kidney', 'krvskp', 'voting', 'census', 'airline', 'kdd99']
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
    datasets = ['autism', 'breast', 'cars',
                'credit_australia', 'heart', 'ionosphere',
                'kidney', 'krvskp', 'voting']
    # datasets = ['census']
    for data in datasets:
        print('='*20, data, '='*20)
        run_experiment(data)
