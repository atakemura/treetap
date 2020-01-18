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
    # print(ret_str)

    skl = SkylineSolver()
    skl.solve(ret_str)
    # print(skl.models)

    for ans_idx, model in enumerate(skl.models):
        print(ans_idx)
        for m in model:
            # this filter may be unnecessary, depending on the #show setting
            if m.name == 'selected':
                selected_ptn = [p for p in a.patterns_ if p.idx == m.arguments[0].number][0]
                print(selected_ptn)
