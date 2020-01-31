import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from tree_asp.rule_extractor import RFRuleExtractor
from tree_asp.asp_encoding import SkylineSolver, MaximalSolver, ClosedSolver


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

    a = RFRuleExtractor()
    a.fit(X, y, model=rf, feature_names=wine_feat)

    ret_str = a.transform(X, y)

    tmp_pattern_file = './tmp/pattern_out.txt'

    with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
        outfile.write(ret_str)

