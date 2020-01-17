import lightgbm as lgb
import pandas as pd
import json
import shap

from sklearn.model_selection import train_test_split
from pathlib import Path


SEED = 42


def train_lgb():
    dataset = 'heart'
    dataset_dir = Path('../datasets/datasets/') / dataset

    raw = pd.read_csv(Path(dataset_dir / dataset).with_suffix('.csv'))
    with open(dataset_dir / 'schema.json', 'r') as infile:
        schema = json.load(infile)
    for c in schema['categorical_columns']:
        raw.loc[:, c] = raw.loc[:, c].astype('category')
    raw_x = raw[[c for c in raw.columns if c != schema['label_column']]]
    raw_y = raw[schema['label_column']]
    # holdout set
    x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y,
                                                        stratify=raw_y,
                                                        train_size=0.8,
                                                        random_state=SEED)

    # train/validation set
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train,
                                                stratify=y_train,
                                                train_size=0.8,
                                                random_state=SEED)
    model = lgb.LGBMClassifier(random_state=SEED)
    model.fit(x_tr, y_tr,
              early_stopping_rounds=30,
              eval_set=[(x_val, y_val)])

    model.predict(x_tr.iloc[0:2, :])

    shap_explainer = shap.TreeExplainer(model)
    # using lightgbm's sklearn api results in (n, 2) array if you use the native api it would be (n, )
    shap_values = shap_explainer.shap_values(x_tr)[1]

    print('done')


if __name__ == '__main__':
    train_lgb()
