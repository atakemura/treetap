import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from tree_asp.rule_extractor import RFRuleExtractor
from tree_asp.asp_encoding import SkylineSolver


if __name__ == '__main__':
    iris_obj = load_iris()
    iris_feat = iris_obj['feature_names']
    iris_data = iris_obj['data']
    iris_target = iris_obj['target']

    iris_df = pd.DataFrame(iris_data, columns=iris_feat).assign(target=iris_target)
    X, y = iris_df[iris_feat], iris_df['target']

    rf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=4)
    rf.fit(X, y)

    a = RFRuleExtractor()
    a.fit(X, y, model=rf, feature_names=iris_feat)

    ret_str = a.transform(X, y)
    print(ret_str)

    skl = SkylineSolver()
    skl.solve(ret_str)
    print(skl.models)

