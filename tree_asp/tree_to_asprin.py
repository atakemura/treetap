import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from tree_asp.rule_extractor import RFRuleExtractor
from tree_asp.asp_encoding import SkylineSolver, MaximalSolver, ClosedSolver

import subprocess
import sys

from clasp_parser import generate_answers
from answers import AnswerSet, ClaspInfo
from pattern import Pattern, Item


if __name__ == '__main__':
    # iris_obj = load_iris()
    # iris_feat = iris_obj['feature_names']
    # iris_data = iris_obj['data']
    # iris_target = iris_obj['target']
    #
    # iris_df = pd.DataFrame(iris_data, columns=iris_feat).assign(target=iris_target)
    # X, y = iris_df[iris_feat], iris_df['target']
    #
    # rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=4)
    # rf.fit(X, y)
    #
    # a = RFRuleExtractor()
    # a.fit(X, y, model=rf, feature_names=iris_feat)
    #
    # ret_str = a.transform(X, y)
    #
    # tmp_pattern_file = './tmp/pattern_out.txt'
    #
    # with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
    #     outfile.write(ret_str)

    wine_obj = load_wine()
    wine_feat = wine_obj['feature_names']
    wine_data = wine_obj['data']
    wine_target = wine_obj['target']

    wine_df = pd.DataFrame(wine_data, columns=wine_feat).assign(target=wine_target)
    X, y = wine_df[wine_feat], wine_df['target']

    rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=4)
    rf.fit(X, y)

    rf_extractor = RFRuleExtractor()
    rf_extractor.fit(X, y, model=rf, feature_names=wine_feat)

    ret_str = rf_extractor.transform(X, y)

    tmp_pattern_file = './tmp/pattern_out.txt'
    tmp_class_file = './tmp/n_class.lp'

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(ret_str)

    with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
        outfile.write('class(0..{}).'.format(int(y.nunique() - 1)))

    asprin_preference = './asp_encoding/asprin_preference.lp'
    asprin_skyline    = './asp_encoding/skyline.lp'

    o = subprocess.run(['asprin', asprin_preference, asprin_skyline, tmp_class_file, tmp_pattern_file, '0'], capture_output=True)

    answers, clasp_info = generate_answers(o.stdout.decode())

    print('parsing completed')

    for ans_set in answers:
        if not ans_set.is_optimal:
            print('Skipping non-optimal answer: {}'.format(ans_set.answer_id))
            continue
        print(ans_set.answer)
        for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
            if ans[0] != 'selected':
                print('Unsupported answer string, skipping: {}'.format(ans[0]))
            pat_idx = ans[-1][0]
            pat = rf_extractor.patterns_[pat_idx]  # type: Pattern
            print('-'*10)
            print('pattern_idx: item_idx')
            print('{}: {}'.format(pat.idx, pat.items))
            print('class {} if {}'.format(pat.mode_class, pat.pattern_str))
            print('error_rate: {}'.format(pat.error_rate))
            print('size: {}'.format(pat.size))
            print('support: {}'.format(pat.support))
            print('-'*10)
        print('='*80)
