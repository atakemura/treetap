import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_wine

from tree_asp.rule_extractor import RFRuleExtractor
from tree_asp.asp_encoding import SkylineSolver, MaximalSolver, ClosedSolver


if __name__ == '__main__':
    IRIS_ONLY = False

    if IRIS_ONLY:
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
                    selected_ptn = [p for p in a.rules_ if p.idx == m.arguments[0].number][0]
                    print(selected_ptn)

    # TODO: CV for best depth and n_estimators
    # TODO: multiple solvers in case one fails
    else:
        for load_f in [load_iris, load_breast_cancer, load_wine]:
            data_obj = load_f()
            feat = data_obj['feature_names']
            data = data_obj['data']
            target = data_obj['target']

            df = pd.DataFrame(data, columns=feat).assign(target=target)
            X, y = df[feat], df['target']

            rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=4)
            rf.fit(X, y)

            a = RFRuleExtractor()
            a.fit(X, y, model=rf, feature_names=feat)

            ret_str = a.transform(X, y)

            n_class = y.max() + 1

            slv = SkylineSolver(n_class=n_class)
            # slv = MaximalSolver(n_class=n_class)
            # slv = ClosedSolver(n_class=n_class)
            slv.solve(ret_str)

            print(load_f)
            for ans_idx, model in enumerate(slv.models):
                print(ans_idx)
                for m in model:
                    if m.name == 'selected':
                        selected_ptn = [p for p in a.rules_ if p.idx == m.arguments[0].number][0]
                        print(selected_ptn)
