import json
import os
import pandas as pd
import subprocess
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from category_encoders.one_hot import OneHotEncoder
from itertools import product
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy

from rule_extractor import RFRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from rule import Rule


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
                  'max_depth': trial.suggest_int('max_depth', 1, 30),
                  'min_samples_split': trial.suggest_float('min_samples_split', 0.05, 0.5, step=0.02),
                  'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.05, 0.5, step=0.02),
                  'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5, step=0.02),
                  'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
                  }
        rf = RandomForestClassifier(**params, random_state=SEED)
        x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=SEED)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_valid)
        acc = accuracy_score(y_valid, y_pred)
        return acc
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=100, timeout=1200, callbacks=[optuna_early_stopping_callback])
    return study.best_params


def run_experiment(dataset_name, n_estimators, max_depth, encoding, asprin_pref):
    X, y = load_data(dataset_name)
    categorical_features = list(X.columns[X.dtypes == 'category'])
    if len(categorical_features) > 0:
        oh = OneHotEncoder(cols=categorical_features, use_cat_names=True)
        X = oh.fit_transform(X)
    feat = X.columns

    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=2020)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        run_one_round(dataset_name, n_estimators, max_depth, encoding, asprin_pref,
                      train_idx, valid_idx, X, y, feat, fold=f_idx)


def run_one_round(dataset_name, n_estimators, max_depth, encoding, asprin_pref,
                  train_idx, valid_idx, X, y, feature_names, fold=0):
    print('[rf]', dataset_name, n_estimators, max_depth, encoding, asprin_pref, fold)
    start = timer()

    SEED = 42

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    metric_averaging = 'micro' if y_valid.nunique() > 2 else 'binary'

    rf_start = timer()
    best_params = optuna_random_forest(x_train, y_train)
    rf = RandomForestClassifier(**best_params, random_state=SEED)
    # rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=SEED)
    rf.fit(x_train, y_train)
    rf_end = timer()

    rf_vanilla_pred = rf.predict(x_valid)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, rf_vanilla_pred),
                       'precision': precision_score(y_valid, rf_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, rf_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, rf_vanilla_pred, average=metric_averaging)}

    ext_start = timer()
    rf_extractor = RFRuleExtractor()
    rf_extractor.fit(x_train, y_train, model=rf, feature_names=feature_names)
    res_str = rf_extractor.transform(x_train, y_train)
    ext_end = timer()

    exp_dir = './tmp/experiment_rf'

    tmp_pattern_file = os.path.join(exp_dir, 'pattern_out.txt')
    tmp_class_file = os.path.join(exp_dir, 'n_class.lp')

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(res_str)

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        outfile.write('class(0..{}).'.format(int(y_train.nunique() - 1)))

    asprin_pareto_1   = './asp_encoding/asprin_pareto_1.lp'
    asprin_pareto_2   = './asp_encoding/asprin_pareto_2.lp'
    asprin_lexico     = './asp_encoding/asprin_lexico.lp'
    asprin_skyline    = './asp_encoding/skyline.lp'
    asprin_maximal    = './asp_encoding/maximal.lp'
    asprin_closed     = './asp_encoding/closed.lp'

    asprin_enc = {'skyline': asprin_skyline, 'maximal': asprin_maximal, 'closed': asprin_closed}
    asprin_preference = {'pareto_1': asprin_pareto_1, 'pareto_2': asprin_pareto_2, 'lexico': asprin_lexico}

    asprin_start = timer()
    try:
        o = subprocess.run(['asprin', asprin_preference[asprin_pref], asprin_enc[encoding],
                            tmp_class_file, tmp_pattern_file, '0', '--parallel-mode=8'
                            ], capture_output=True, timeout=120)
        asprin_completed = True
    except subprocess.TimeoutExpired:
        o = None
        asprin_completed = False
    asprin_end = timer()

    if asprin_completed:
        answers, clasp_info = generate_answers(o.stdout.decode())
    else:
        answers, clasp_info = None, None
    end = timer()
    # print('parsing completed')

    log_json = os.path.join(exp_dir, 'output.json')
    log_json_quali = os.path.join(exp_dir, 'output_quali.json')

    if asprin_completed:
        scores = []
        for ans_idx, ans_set in enumerate(answers):
            if not ans_set.is_optimal:
                continue
            rules = []
            for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                pat_idx = ans[-1][0]
                pat = rf_extractor.rules_[pat_idx]  # type: Rule
                rules.append(pat)
            # break
            rule_classifier = RuleClassifier(rules)
            rule_classifier.fit(x_train, y_train)
            rule_pred = rule_classifier.predict(x_valid)
            rule_pred_metrics = {'accuracy': accuracy_score(y_valid, rule_pred),
                                 'precision': precision_score(y_valid, rule_pred, average=metric_averaging),
                                 'recall': recall_score(y_valid, rule_pred, average=metric_averaging),
                                 'f1': f1_score(y_valid, rule_pred, average=metric_averaging)}
            scores.append((ans_idx, rule_pred_metrics))

        out_dict = {
            # experiment
            'dataset': dataset_name,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'encoding': encoding,
            'asprin_preference': asprin_pref,
            'asprin_completed': asprin_completed,
            # clasp
            'models': clasp_info.stats['Models'],
            'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            'optimal': int(clasp_info.stats['Optimal']),
            'clasp_time': clasp_info.stats['Time'],
            'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            'rf_n_nodes': len(rf_extractor.literals_),
            'rf_n_patterns': len(rf_extractor.rules_),
            # timer
            'py_total_time': end - start,
            'py_rf_time': rf_end - rf_start,
            'py_ext_time': ext_end - ext_start,
            'py_asprin_time': asprin_end - asprin_start,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            # 'rule_metrics': rule_pred_metrics,
            'rule_metrics': scores,
        }
    else:
        out_dict = {
            # experiment
            'dataset': dataset_name,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'encoding': encoding,
            'asprin_preference': asprin_pref,
            'asprin_completed': asprin_completed,
            # # clasp
            # 'models': int(clasp_info.stats['Models']),
            # 'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            # 'optimal': int(clasp_info.stats['Optimal']),
            # 'clasp_time': clasp_info.stats['Time'],
            # 'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            'rf_n_nodes': len(rf_extractor.literals_),
            'rf_n_patterns': len(rf_extractor.rules_),
            # timer
            'py_total_time': end - start,
            'py_rf_time': rf_end - rf_start,
            'py_ext_time': ext_end - ext_start,
            'py_asprin_time': asprin_end - asprin_start,
            # metrics
            'fold': fold,
            'vanilla_metrics': vanilla_metrics,
            # 'rule_metrics': rule_pred_metrics,
        }
    with open(log_json, 'a', encoding='utf-8') as out_log_json:
        out_log_json.write(json.dumps(out_dict)+'\n')

    # this is for qualitative answer pattern only
    verbose = True

    out_quali = deepcopy(out_dict)
    out_quali['rules'] = []
    if asprin_completed:
        for ans_idx, ans_set in enumerate(answers):
            _tmp_rules = []
            if not ans_set.is_optimal:
                # print('Skipping non-optimal answer: {}'.format(ans_set.answer_id))
                continue
            for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                pat_idx = ans[-1][0]
                pat = rf_extractor.rules_[pat_idx]  # type: Rule
                pat_dict = {
                    'rule_idx': pat.idx,
                    'items': [x.literal_str for x in pat.items],
                    'rule_str': 'class {} IF {}'.format(pat.predict_class, pat.rule_str),
                    'predict_class': int(pat.predict_class),
                    'error_rate': int(pat.error_rate),
                    'accuracy': int(pat.accuracy),
                    'precision': int(pat.precision),
                    'size': int(pat.size),
                    'support': int(pat.support),
                }
                _tmp_rules.append(pat_dict)
            out_quali['rules'].append((ans_idx, _tmp_rules))


    if verbose:
        with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
            out_log_quali.write(json.dumps(out_quali)+'\n')


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
        dataset_dir = Path('../datasets/datasets/') / dataset_name

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
    start_time = timer()

    debug_mode = True

    if debug_mode:
        data = ['breast_sk', 'wine']
        n_estimators = [10]
        max_depths = [5]
        encodings = ['skyline']
        asprin_pref = ['pareto_1']
    else:
        data = ['breast_sk', 'iris', 'wine',
                'autism', 'breast', 'cars', 'credit_australia',
                'heart', 'ionosphere', 'kidney', 'krvskp', 'voting']
        n_estimators = [10]
        max_depths = [5]
        encodings = ['skyline', 'maximal', 'closed']
        asprin_pref = ['pareto_1', 'pareto_2', 'lexico']

    combinations = product(data, n_estimators, max_depths, encodings, asprin_pref)
    for cond_tuple in tqdm(combinations):
        run_experiment(*cond_tuple)
    end_time = timer()
    e = end_time - start_time
    print('Time elapsed(s): {}'.format(e))
