import json
import os
import pandas as pd
import subprocess

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from category_encoders.one_hot import OneHotEncoder
from itertools import product
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy

from rule_extractor import DTRuleExtractor
from classifier import RuleClassifier
from clasp_parser import generate_answers
from rule import Rule


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
    # run single round of model + asprin
    # print fold into json
    # calculate accuracy scores
    print('[dt]', dataset_name, n_estimators, max_depth, encoding, asprin_pref, fold)
    start = timer()

    SEED = 42

    x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # multilabel case
    metric_averaging = 'micro' if y_valid.nunique() > 2 else 'binary'

    dt_start = timer()
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=SEED)
    dt.fit(x_train, y_train)
    dt_end = timer()

    dt_vanilla_pred = dt.predict(x_valid)
    vanilla_metrics = {'accuracy':  accuracy_score(y_valid, dt_vanilla_pred),
                       'precision': precision_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'recall':    recall_score(y_valid, dt_vanilla_pred, average=metric_averaging),
                       'f1':        f1_score(y_valid, dt_vanilla_pred, average=metric_averaging)}

    ext_start = timer()
    dt_extractor = DTRuleExtractor()
    dt_extractor.fit(x_train, y_train, model=dt, feature_names=feature_names)
    res_str = dt_extractor.transform(x_train, y_train)
    ext_end = timer()

    exp_dir = './tmp/experiment_dt'

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
                pat = dt_extractor.rules_[pat_idx]  # type: Rule
                rules.append(pat)
            # break
            rule_classifier = RuleClassifier(rules)
            rule_classifier.fit(x_train, y_train)
            rule_pred = rule_classifier.predict(x_valid)
            rule_pred_metrics = {'accuracy': accuracy_score(y_valid, rule_pred),
                                 'precision': precision_score(y_valid, rule_pred, average=metric_averaging),
                                 'recall': recall_score(y_valid, rule_pred, average=metric_averaging),
                                 'f1': f1_score(y_valid, rule_pred, average=metric_averaging)}
            scores.append(rule_pred_metrics)

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
            # dt related
            'dt_n_nodes': len(dt_extractor.literals_),
            'dt_n_patterns': len(dt_extractor.rules_),
            # timer
            'py_total_time': end - start,
            'py_dt_time': dt_end - dt_start,
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
            # dt related
            'dt_n_nodes': len(dt_extractor.literals_),
            'dt_n_patterns': len(dt_extractor.rules_),
            # timer
            'py_total_time': end - start,
            'py_dt_time': dt_end - dt_start,
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
                pat = dt_extractor.rules_[pat_idx]  # type: Rule
                pat_dict = {
                    'pattern_idx': pat.idx,
                    'items': [x.literal_str for x in pat.items],
                    'rule_str': 'class {} if {}'.format(pat.predict_class, pat.rule_str),
                    'predict_class': int(pat.predict_class),
                    'error_rate': int(pat.error_rate),
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
        n_estimators = [1]
        max_depths = [5]
        encodings = ['skyline']
        asprin_pref = ['pareto_1']

    else:
        data = ['breast_sk', 'iris', 'wine',
                'autism', 'breast', 'cars', 'credit_australia',
                'heart', 'ionosphere', 'kidney', 'krvskp', 'voting']
        n_estimators = [1]
        max_depths = [5]
        encodings = ['skyline', 'maximal', 'closed']
        asprin_pref = ['pareto_1', 'pareto_2', 'lexico']

    combinations = product(data, n_estimators, max_depths, encodings, asprin_pref)
    for cond_tuple in tqdm(combinations):
        run_experiment(*cond_tuple)
    end_time = timer()
    e = end_time - start_time
    print('Time elapsed(s): {}'.format(e))
