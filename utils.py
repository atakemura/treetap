import json
import pandas as pd

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from pathlib import Path


def load_data(dataset_name):
    # there is no categorical feature in these sklearn datasets.
    sklearn_data = {'iris': load_iris,
                    'breast_sk': load_breast_cancer,
                    'wine': load_wine}
    # the following contains a mix of categorical and numerical features.
    datasets = ['autism', 'breast', 'cars',
                'credit_australia', 'credit_taiwan', 'heart', 'ionosphere',
                'kidney', 'krvskp', 'voting',
                # small, 2000 synthetic
                'synthetic_1',
                # 50k
                'eeg',
                # these datasets are large (>200k), use with caution
                'census', 'airline', 'kdd99',
                # benchmark, used in papers
                'adult', 'credit_german', 'compas']
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
        dataset_dir = Path('./datasets/datasets/') / dataset_name

        raw = pd.read_csv(Path(dataset_dir / dataset_name).with_suffix('.csv.gz'))
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


def list_data():
    datasets = [
        'adult', 'autism', 'breast', 'cars', 'census', 'compas',
        'credit_australia', 'credit_german', 'credit_taiwan', 'heart',
        'ionosphere', 'kidney', 'krvskp', 'voting']
    to_df = []
    for d in datasets:
        data_X, data_y = load_data(d)  # pd.DataFrame
        to_df.append({
            'dataset': d,
            'num_instances': int(data_X.shape[0]),
            'num_features': int(data_X.shape[1]),
            'num_features_numerical': len(data_X._get_numeric_data().columns),
            'num_features_categorical': len(data_X.select_dtypes(include=['category']).columns),
            'label_0': int(data_y.value_counts()[0]),
            'label_1': int(data_y.value_counts()[1]),
            'label_ratio_1': round(data_y.value_counts()[1] / len(data_y.values),2)
        })
    print(json.dumps(to_df, indent=4))
