import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from rule_extractor import RFRuleExtractor

import subprocess
import os
from itertools import product
import json
from tqdm import tqdm
from timeit import default_timer as timer
from copy import deepcopy

from clasp_parser import generate_answers
from answers import AnswerSet, ClaspInfo
from pattern import Pattern, Item


def run_experiment(load_data_method, n_estimators, max_depth, encoding):
    print(load_data_method.__name__, n_estimators, max_depth, encoding)
    start = timer()

    data_obj = load_data_method()
    feat = data_obj['feature_names']
    data = data_obj['data']
    target = data_obj['target']

    df = pd.DataFrame(data, columns=feat).assign(target=target)
    X, y = df[feat], df['target']

    rf_start = timer()
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X, y)
    rf_end = timer()

    ext_start = timer()
    rf_extractor = RFRuleExtractor()
    rf_extractor.fit(X, y, model=rf, feature_names=feat)
    res_str = rf_extractor.transform(X, y)
    ext_end = timer()

    exp_dir = './tmp/experiment_rf'

    tmp_pattern_file = os.path.join(exp_dir, 'pattern_out.txt')
    tmp_class_file = os.path.join(exp_dir, 'n_class.lp')

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(res_str)

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        outfile.write('class(0..{}).'.format(int(y.nunique() - 1)))

    asprin_preference = './asp_encoding/asprin_preference.lp'
    asprin_skyline    = './asp_encoding/skyline.lp'
    asprin_maximal    = './asp_encoding/maximal.lp'
    asprin_closed     = './asp_encoding/closed.lp'

    asprin_enc = {'skyline': asprin_skyline, 'maximal': asprin_maximal, 'closed': asprin_closed}

    asprin_start = timer()
    try:
        o = subprocess.run(['asprin', asprin_preference, asprin_enc[encoding],
                            tmp_class_file, tmp_pattern_file, '0',
                            ], capture_output=True, timeout=60)
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
        out_dict = {
            # experiment
            'dataset': load_data_method.__name__,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'encoding': encoding,
            'asprin_completed': asprin_completed,
            # clasp
            'models': int(clasp_info.stats['Models']),
            'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            'optimal': int(clasp_info.stats['Optimal']),
            'clasp_time': clasp_info.stats['Time'],
            'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            'rf_n_nodes': len(rf_extractor.items_),
            'rf_n_patterns': len(rf_extractor.patterns_),
            # timer
            'py_total_time': end - start,
            'py_rf_time': rf_end - rf_start,
            'py_ext_time': ext_end - ext_start,
            'py_asprin_time': asprin_end - asprin_start,
        }
    else:
        out_dict = {
            # experiment
            'dataset': load_data_method.__name__,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'encoding': encoding,
            'asprin_completed': asprin_completed,
            # # clasp
            # 'models': int(clasp_info.stats['Models']),
            # 'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
            # 'optimal': int(clasp_info.stats['Optimal']),
            # 'clasp_time': clasp_info.stats['Time'],
            # 'clasp_cpu_time': clasp_info.stats['CPU Time'],
            # rf related
            'rf_n_nodes': len(rf_extractor.items_),
            'rf_n_patterns': len(rf_extractor.patterns_),
            # timer
            'py_total_time': end - start,
            'py_rf_time': rf_end - rf_start,
            'py_ext_time': ext_end - ext_start,
            'py_asprin_time': asprin_end - asprin_start,
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
                pat = rf_extractor.patterns_[pat_idx]  # type: Pattern
                pat_dict = {
                    'pattern_idx': pat.idx,
                    'items': [x.item_str for x in pat.items],
                    'rule_str': 'class {} if {}'.format(pat.mode_class, pat.pattern_str),
                    'mode_class': int(pat.mode_class),
                    'error_rate': int(pat.error_rate),
                    'size': int(pat.size),
                    'support': int(pat.support),
                }
                _tmp_rules.append(pat_dict)
            out_quali['rules'].append((ans_idx, _tmp_rules))


    if verbose:
        with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
            out_log_quali.write(json.dumps(out_quali)+'\n')


if __name__ == '__main__':
    start_time = timer()
    data = [load_iris, load_breast_cancer, load_wine]
    n_estimators = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    max_depths = [3, 4, 5, 6, 7, 8]
    # n_estimators = [9, 10, 11]
    # max_depths = [5, 6, 7]
    encodings = ['skyline', 'maximal', 'closed']
    # encodings = ['maximal']

    combinations = product(data, n_estimators, max_depths, encodings)
    for cond_tuple in tqdm(combinations):
        run_experiment(*cond_tuple)
    end_time = timer()
    e = end_time - start_time
    print('Time elapsed(s): {}'.format(e))
