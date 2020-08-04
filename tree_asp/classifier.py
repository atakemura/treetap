import pandas as pd
import numpy as np
from typing import List
import dask.dataframe as dd

from pattern import Pattern
from rule_extractor import LT_PATTERN, LE_PATTERN, GT_PATTERN,\
    GE_PATTERN, EQ_PATTERN, NEQ_PATTERN


class RuleClassifier:
    def __init__(self, patterns: List[Pattern], default_class: int=None):
        self.rules = self.sort_rules(patterns)
        self.default_class = default_class
        self.categorical_maps = {}

    def fit(self, X=None, y=None, **params):
        if not self.default_class:
            self.default_class = y.mode()[0]
        categorical_features = list(X.columns[X.dtypes == 'category'])
        if len(categorical_features) > 0:
            for cat_f in categorical_features:
                self.categorical_maps[cat_f] = dict(enumerate(X[cat_f].cat.categories))
        return self

    def predict(self, X: pd.DataFrame, **params):
        # assume pandas data frame instead of numpy array
        # predicted = np.empty(X.shape[0], dtype=np.int32)
        # predicted_ = X.apply(self.predict_single_row, axis=1)

        Xdd = dd.from_pandas(X, npartitions=32)
        predicted_ = Xdd.apply(self.predict_single_row, axis=1, meta=(None, np.int32)).compute(scheduler='processes')
        # assert X.shape[0] == predicted.shape[0]
        assert X.shape[0] == predicted_.shape[0]
        return predicted_

    def sort_rules(self, patterns: List[Pattern]):
        # sort based on error rate, length and class label
        sorted_list = sorted(patterns, key=lambda x: (x.error_rate, x.size, x.mode_class))
        return sorted_list

    def predict_single_row(self, row):
        # returning prediction for a single row
        for pattern in self.rules:
            items = [False] * pattern.size
            for item_idx, item in enumerate(pattern.items):
                item_str = item.item_str
                item_active = False
                if LT_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(LT_PATTERN, 1)
                    # item_active = getattr(row[_rule_field], 'lt')(float(_rule_threshold))
                    item_active = row[_rule_field] < float(_rule_threshold)
                elif LE_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(LE_PATTERN, 1)
                    # item_active = getattr(row[_rule_field], 'le')(float(_rule_threshold))
                    item_active = row[_rule_field] <= float(_rule_threshold)
                elif GT_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(GT_PATTERN, 1)
                    # item_active = getattr(row[_rule_field], 'gt')(float(_rule_threshold))
                    item_active = row[_rule_field] > float(_rule_threshold)
                elif GE_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(GE_PATTERN, 1)
                    # item_active = getattr(row[_rule_field], 'ge')(float(_rule_threshold))
                    item_active = row[_rule_field] >= float(_rule_threshold)
                elif EQ_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(EQ_PATTERN, 1)
                    # inverse_map = dict(enumerate(row[_rule_field].cat.categories))
                    inverse_map = self.categorical_maps[_rule_field]
                    # rule threshold in this case can be like '0||1||2'
                    _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                    # item_active = getattr(row[_rule_field], 'isin')(_cat_th)
                    item_active = row[_rule_field] in _cat_th
                elif NEQ_PATTERN in item_str:
                    _rule_field, _rule_threshold = item_str.rsplit(NEQ_PATTERN, 1)
                    # inverse_map = dict(enumerate(row[_rule_field].cat.categories))
                    inverse_map = self.categorical_maps[_rule_field]
                    # rule threshold in this case can be like '0||1||2'
                    _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                    # note the bitwise complement not
                    item_active = not (row[_rule_field] in _cat_th)
                else:
                    raise ValueError('No key found')
                if item_active:  # this item was active, so check next
                    items[item_idx] = True
                    continue
                else:  # not active then stop searching
                    break
            # item loop completed
            all_items_active = np.alltrue(items)
            if all_items_active:  # if acceptable pattern is found
                return pattern.mode_class
        # exhausted
        return self.default_class