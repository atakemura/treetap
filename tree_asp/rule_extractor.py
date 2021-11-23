import numpy as np
import lightgbm as lgb

from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted

from copy import deepcopy
from pprint import pprint
from functools import reduce
from multiprocessing import Pool
from os import cpu_count
from operator import and_

from rule import Rule, Condition
from utils import timer_exec

LT_PATTERN = ' < '
LE_PATTERN = ' <= '
GT_PATTERN = ' > '
GE_PATTERN = ' >= '
EQ_PATTERN = ' == '
NEQ_PATTERN = ' != '


class OnlyLeafFoundException(BaseException):
    pass


class DTRuleExtractor:
    def __init__(self, verbose=False):
        self.feature_names = None
        self.verbose = verbose
        self.non_rule_keys = ['class', 'condition_length', 'error_rate',
                              'precision', 'recall', 'f1_score',
                              'frequency', 'frequency_am', 'predict_class', 'value',
                              'is_tree_max', 'accuracy']
        self.asp_fact_str = None
        self.fitted_ = False
        self.rules_ = None
        self.conditions_ = None
        self.metric_averaging = None

    def fit(self, X, y, model=None, feature_names=None, **params):
        # validate the input model
        if model is None:
            raise ValueError('model parameter is required.')
        if not isinstance(model, DecisionTreeClassifier):
            raise ValueError('only DecisionTreeClassifier is supported at the moment.')
        check_is_fitted(model, 'tree_')

        num_classes = y.nunique()
        self.metric_averaging = 'micro' if num_classes > 2 else 'binary'

        # decision tree only has 1 tree
        rules = {}
        _, rules[0] = self.export_text_rule_tree(model, X, feature_names)
        rules = self.export_text_rule_dt(rules, X, y)

        # simple rule merging
        printable_dicts = self.asp_dict_from_rules(rules)

        # printable strings
        print_str = self.asp_str_from_dicts(printable_dicts)
        self.asp_fact_str = print_str
        self.fitted_ = True

        return self

    def transform(self, X, y=None, **params):
        return self.asp_fact_str

    def export_text_rule_tree(self, decision_tree, train_data, feature_names=None):
        """
        Given a trained decision_tree instance and trained_data,
        return extracted rules for each decision tree instance
        """
        n_nodes = decision_tree.tree_.node_count
        children_left = decision_tree.tree_.children_left
        children_right = decision_tree.tree_.children_right
        feature = decision_tree.tree_.feature
        threshold = decision_tree.tree_.threshold

        check_is_fitted(decision_tree, 'tree_')
        tree_ = decision_tree.tree_
        if is_classifier(decision_tree):
            class_names = decision_tree.classes_

        # if feature names is specified.
        if feature_names is not None and len(feature_names) != tree_.n_features:
            raise ValueError("feature_names must contain {} elements, got {}".format(tree_.n_features,
                                                                                     len(feature_names)))
        if feature_names is None:
            feature_names = ['feature_{}'.format(i) for i in range(tree_.n_features)]

        # get leave ids
        leave_id = decision_tree.apply(train_data)
        # get path to the leaves
        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path_recursive(0, path_leaf, leaf, children_left, children_right)
            paths[leaf] = np.unique(np.sort(path_leaf))
        # get rules
        rules = {}
        for key in paths:
            rules[key] = self.get_rule(paths[key], children_left, feature_names, feature, threshold, decision_tree,
                                       tree_)

        return paths, rules

    def find_path_recursive(self, node_idx, path, x, children_left, children_right):
        """
        Given a list of nodes, and target node x, recursively find the nodes that will be visited
        and append to the list path
        """
        path.append(node_idx)
        # reached the goal
        if node_idx == x:
            return True
        left = False
        right = False
        if children_left[node_idx] != _tree.TREE_LEAF:
            left = self.find_path_recursive(children_left[node_idx], path, x, children_left, children_right)
        if children_right[node_idx] != _tree.TREE_LEAF:
            right = self.find_path_recursive(children_right[node_idx], path, x, children_left, children_right)
        # reached leaf in left or right child
        if left or right:
            return True
        path.remove(node_idx)
        return False

    def get_rule(self, path, children_left, column_names, feature, threshold, decision_tree, tree_):
        """
        Given a path (list of nodes from the root to the leaves), list all of decisions,
        along with the classification result (argmax of class).
        Refer to sklearn's tree (cdef class Tree)

        Args:
            path: list of paths
            children_left: sklearn's tree.children_left
            column_names: column names of features
            feature: feature to split on
            threshold: threshold for the split
            decision_tree: decision tree instance
            tree_: decision_tree.tree_ instance

        Returns:

        """
        rule_dict = {}
        class_names = decision_tree.classes_
        for idx, node in enumerate(path):
            # at node, not at leaf
            if idx != len(path) - 1:
                # left-child of this node is the next node in path, then it's <=
                # see sklearn/tree/_tree.pyx, Tree.children_left
                if children_left[node] == path[idx + 1]:
                    rule_dict['{} <= {}'.format(column_names[feature[node]], threshold[node])] = 1
                # right-child handles >
                else:
                    rule_dict['{} > {}'.format(column_names[feature[node]], threshold[node])] = 1
            # at leaf
            else:
                if tree_.n_outputs == 1:
                    value = tree_.value[node][0]
                else:
                    value = tree_.value[node].T[0]
                if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
                    class_name = class_names[np.argmax(value)]
                else:
                    class_name = np.argmax(value)
                rule_dict['class'] = class_name  # most frequent class is the prediction
                rule_dict['value'] = np.sum(value)  # number of supporting examples
        return rule_dict

    def export_text_rule_dt(self, tree_rules: dict, X, y):
        rules = tree_rules
        # adding rule statistics
        for t_idx, t_rules in rules.items():
            for path_rule in t_rules.values():
                _tmp_dfs = []
                for rule_key in path_rule.keys():
                    # skip non-conditions
                    if rule_key in self.non_rule_keys:
                        continue
                    # collect conditions and boolean mask
                    else:
                        if LT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'lt')(float(_rule_threshold)))
                        elif LE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'le')(float(_rule_threshold)))
                        elif GT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'gt')(float(_rule_threshold)))
                        elif GE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'ge')(float(_rule_threshold)))
                        else:
                            raise ValueError('No key found')
                # reduce boolean mask
                mask_res = reduce(and_, _tmp_dfs)
                # these depend on the entire training data, not on the bootstrapped data that the original rf uses
                # path_rule['predict_class'] = y[mask_res].mode()[0]
                path_rule['predict_class'] = y[mask_res].mode()[0]
                y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
                path_rule['condition_length'] = len(_tmp_dfs)
                path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
                path_rule['frequency'] = len(y[mask_res])  # coverage
                path_rule['error_rate'] = 1 - accuracy_score(y[mask_res], y_pred)
                path_rule['accuracy'] = accuracy_score(y[mask_res], y_pred)
                path_rule['precision'] = precision_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                         average=self.metric_averaging)
                path_rule['recall'] = recall_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                   average=self.metric_averaging)
                path_rule['f1_score'] = f1_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                 average=self.metric_averaging)

        return rules

    def asp_dict_from_rules(self, rules: dict):
        """
        Construct intermediate list of dicts that can be processed into ASP strings.
        In this implementation, duplicate patterns are merged by aggregating the support (frequency).

        Args:
            rules: dict of dicts containing rules

        Returns:
            list of dicts
        """
        rule_list = []
        condition_list = []
        print_dicts = []

        # rule and condition instances
        rl_obj_list = []
        lit_obj_list = []

        for t_idx, t in rules.items():
            for node_idx, node_rule in t.items():
                # rule
                rule_txt = ' AND '.join([k for k in node_rule.keys() if k not in self.non_rule_keys])

                if rule_txt in rule_list:
                    rule_idx = rule_list.index(rule_txt)
                    # # add support
                    # # TODO: do not accumulate
                    # for p in print_dicts:
                    #     if p['rule_idx'] == rule_idx:
                    #         p['support'] += node_rule['frequency']
                    continue
                else:
                    rule_list.append(rule_txt)
                    rule_idx = len(rule_list) - 1
                # conditions
                _list_conditions = []
                for k in node_rule.keys():
                    if k not in self.non_rule_keys:
                        if k in condition_list:
                            lit_idx = condition_list.index(k)
                        else:
                            condition_list.append(k)
                            lit_idx = len(condition_list) - 1
                            lit_obj_list.append(Condition(lit_idx, k))
                        _list_conditions.append((rule_idx, lit_idx))
                if self.verbose:
                    print('rule_idx: {} class={}: {}'.format(rule_idx, node_rule['predict_class'], rule_txt))
                    print('conditions: {}'.format(['condition{}'.format(x) for x in _list_conditions]))
                    print('support: {}'.format(node_rule['frequency']))
                    print('=' * 30 + '\n')

                # create new dict
                prn = {
                    'rule_idx': rule_idx,
                    'rule': rule_txt,
                    'conditions': [x for x in _list_conditions],
                    'support': node_rule['frequency_am'],
                    'size': len(_list_conditions),
                    'error_rate': node_rule['error_rate'],
                    'accuracy': node_rule['accuracy'],
                    'precision': node_rule['precision'],
                    'recall': node_rule['recall'],
                    'f1_score': node_rule['f1_score'],
                    'predict_class': node_rule['predict_class']
                }
                print_dicts.append(prn)
                rl_obj_list.append(Rule(idx=rule_idx,
                                         rule_str=rule_txt,
                                         conditions=[Condition(itm_idx_k, condition_list[itm_idx_k])
                                                   for (_, itm_idx_k) in _list_conditions],
                                         support=int(round(node_rule['frequency'] * 100)),
                                         size=len(_list_conditions),
                                         error_rate=int(round(node_rule['error_rate']*100)),
                                         accuracy=int(round(node_rule['accuracy'] * 100)),
                                         precision=int(round(node_rule['precision'] * 100)),
                                         recall=int(round(node_rule['recall'] * 100)),
                                         f1_score=int(round(node_rule['f1_score'] * 100)),
                                         predict_class=node_rule['predict_class']))
        self.rules_ = rl_obj_list
        self.conditions_ = lit_obj_list
        return print_dicts

    def asp_str_from_dicts(self, list_dicts: list):
        print_lines = []
        for rule_dict in list_dicts:
            prn = []
            rule_idx = rule_dict['rule_idx']
            prn.append('rule({}).'.format(rule_idx))
            for x in rule_dict['conditions']:
                prn.append('condition({},{}).'.format(x[0], x[1]))
            prn.append('support({},{}).'.format(rule_idx, int(round(rule_dict['support'] * 100))))
            prn.append('size({},{}).'.format(rule_idx, rule_dict['size']))
            prn.append('accuracy({},{}).'.format(rule_idx, int(round(rule_dict['accuracy'] * 100))))
            prn.append('error_rate({},{}).'.format(rule_idx, int(round(rule_dict['error_rate'] * 100))))
            prn.append('precision({},{}).'.format(rule_idx, int(round(rule_dict['precision'] * 100))))
            prn.append('recall({},{}).'.format(rule_idx, int(round(rule_dict['recall'] * 100))))
            prn.append('f1_score({},{}).'.format(rule_idx, int(round(rule_dict['f1_score'] * 100))))
            prn.append('predict_class({},{}).'.format(rule_idx, rule_dict['predict_class']))
            print_lines.append(' '.join(prn))
        return_str = '\n'.join(print_lines)
        return return_str


class RFRuleExtractor:
    def __init__(self, verbose=False):
        self.feature_names = None
        self.verbose = verbose
        self.non_rule_keys = ['class', 'condition_length', 'error_rate',
                              'precision', 'recall', 'f1_score',
                              'frequency', 'frequency_am', 'predict_class', 'value',
                              'is_tree_max', 'accuracy']
        self.asp_fact_str = None
        self.fitted_ = False
        self.rules_ = None
        self.conditions_ = None
        self.metric_averaging = None

    def fit(self, X, y, model=None, feature_names=None, **params):
        # validate the input model
        if model is None:
            raise ValueError('model parameter is required.')
        if not isinstance(model, RandomForestClassifier):
            raise ValueError('only RandomForestClassifier is supported at the moment.')
        check_is_fitted(model, 'estimators_')

        num_classes = y.nunique()
        self.metric_averaging = 'micro' if num_classes > 2 else 'binary'

        # extract rules for all trees
        rules = {}
        for t_idx, tree in enumerate(model.estimators_):
            try:
                _, rules[t_idx] = self.export_text_rule_tree(tree, X, feature_names)
            except OnlyLeafFoundException:
                print('skipping leaf only tree')
                continue
        rules = self.export_text_rule_rf(rules, X, y)

        # simple rule merging
        printable_dicts = self.asp_dict_from_rules(rules)

        # printable strings
        print_str = self.asp_str_from_dicts(printable_dicts)
        self.asp_fact_str = print_str
        self.fitted_ = True

        return self

    def transform(self, X, y=None, **params):
        return self.asp_fact_str

    def export_text_rule_tree(self, decision_tree, train_data, feature_names=None):
        """
        Given a trained decision_tree instance and trained_data,
        return extracted rules for each decision tree instance
        """
        n_nodes = decision_tree.tree_.node_count
        children_left = decision_tree.tree_.children_left
        children_right = decision_tree.tree_.children_right
        feature = decision_tree.tree_.feature
        threshold = decision_tree.tree_.threshold

        check_is_fitted(decision_tree, 'tree_')
        tree_ = decision_tree.tree_
        if is_classifier(decision_tree):
            class_names = decision_tree.classes_

        # if feature names is specified.
        if feature_names is not None and len(feature_names) != tree_.n_features:
            raise ValueError("feature_names must contain {} elements, got {}".format(tree_.n_features,
                                                                                     len(feature_names)))
        if feature_names is None:
            feature_names = ['feature_{}'.format(i) for i in range(tree_.n_features)]

        # get leave ids
        leave_id = decision_tree.apply(train_data)
        # get path to the leaves
        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path_recursive(0, path_leaf, leaf, children_left, children_right)
            if len(np.unique(path_leaf)) == 1:
                raise OnlyLeafFoundException
            paths[leaf] = np.unique(np.sort(path_leaf))
        # get rules
        rules = {}
        for key in paths:
            rules[key] = self.get_rule(paths[key], children_left, feature_names, feature, threshold, decision_tree,
                                       tree_)

        return paths, rules

    def find_path_recursive(self, node_idx, path, x, children_left, children_right):
        """
        Given a list of nodes, and target node x, recursively find the nodes that will be visited
        and append to the list path
        """
        path.append(node_idx)
        # reached the goal
        if node_idx == x:
            return True
        left = False
        right = False
        if children_left[node_idx] != _tree.TREE_LEAF:
            left = self.find_path_recursive(children_left[node_idx], path, x, children_left, children_right)
        if children_right[node_idx] != _tree.TREE_LEAF:
            right = self.find_path_recursive(children_right[node_idx], path, x, children_left, children_right)
        # reached leaf in left or right child
        if left or right:
            return True
        path.remove(node_idx)
        return False

    def get_rule(self, path, children_left, column_names, feature, threshold, decision_tree, tree_):
        """
        Given a path (list of nodes from the root to the leaves), list all of decisions,
        along with the classification result (argmax of class).
        Refer to sklearn's tree (cdef class Tree)

        Args:
            path: list of paths
            children_left: sklearn's tree.children_left
            column_names: column names of features
            feature: feature to split on
            threshold: threshold for the split
            decision_tree: decision tree instance
            tree_: decision_tree.tree_ instance

        Returns:

        """
        rule_dict = {}
        class_names = decision_tree.classes_
        for idx, node in enumerate(path):
            # at node, not at leaf
            if idx != len(path) - 1:
                # left-child of this node is the next node in path, then it's <=
                # see sklearn/tree/_tree.pyx, Tree.children_left
                if children_left[node] == path[idx + 1]:
                    rule_dict['{} <= {}'.format(column_names[feature[node]], threshold[node])] = 1
                # right-child handles >
                else:
                    rule_dict['{} > {}'.format(column_names[feature[node]], threshold[node])] = 1
            # at leaf
            else:
                if tree_.n_outputs == 1:
                    value = tree_.value[node][0]
                else:
                    value = tree_.value[node].T[0]
                if tree_.n_classes[0] != 1 and tree_.n_outputs == 1:
                    class_name = class_names[np.argmax(value)]
                else:
                    class_name = np.argmax(value)
                rule_dict['class'] = class_name  # most frequent class is the prediction
                rule_dict['value'] = np.sum(value)  # number of supporting examples
        return rule_dict

    def export_text_rule_rf(self, tree_rules: dict, X, y):
        rules = tree_rules
        # adding rule statistics
        for t_idx, t_rules in rules.items():
            for path_rule in t_rules.values():
                _tmp_dfs = []
                for rule_key in path_rule.keys():
                    # skip non-conditions
                    if rule_key in self.non_rule_keys:
                        continue
                    # collect conditions and boolean mask
                    else:
                        if LT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'lt')(float(_rule_threshold)))
                        elif LE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'le')(float(_rule_threshold)))
                        elif GT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'gt')(float(_rule_threshold)))
                        elif GE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'ge')(float(_rule_threshold)))
                        else:
                            raise ValueError('No key found')
                # for a path that has length = 1 (straight to leaf)
                if len(_tmp_dfs) == 0:
                    continue
                # reduce boolean mask
                mask_res = reduce(and_, _tmp_dfs)
                # these depend on the entire training data, not on the bootstrapped data that the original rf uses
                # path_rule['predict_class'] = y[mask_res].mode()[0]
                path_rule['predict_class'] = 1
                y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
                path_rule['condition_length'] = len(_tmp_dfs)
                path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
                path_rule['frequency'] = len(y[mask_res])  # coverage
                path_rule['error_rate'] = 1 - accuracy_score(y[mask_res], y_pred)
                path_rule['accuracy'] = accuracy_score(y[mask_res], y_pred)
                path_rule['precision'] = precision_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                         average=self.metric_averaging)
                path_rule['recall'] = recall_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                   average=self.metric_averaging)
                path_rule['f1_score'] = f1_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                 average=self.metric_averaging)
        return rules

    def asp_dict_from_rules(self, rules: dict):
        """
        Construct intermediate list of dicts that can be processed into ASP strings.
        In this implementation, duplicate patterns are merged by aggregating the support (frequency).

        Args:
            rules: dict of dicts containing rules

        Returns:
            list of dicts
        """
        rule_list = []
        condition_list = []
        print_dicts = []

        # pattern and item instances
        rl_obj_list = []
        lit_obj_list = []

        for t_idx, t in rules.items():
            for node_idx, node_rule in t.items():
                # skip unrelated classes
                if node_rule['class'] != 1:
                    continue
                # rule
                rule_txt = ' AND '.join([k for k in node_rule.keys() if k not in self.non_rule_keys])

                if rule_txt in rule_list:
                    rule_idx = rule_list.index(rule_txt)
                    # # add support
                    # for p in print_dicts:
                    #     # TODO: rethink this, maybe not add the support again
                    #     if p['rule_idx'] == rule_idx:
                    #         p['support'] += node_rule['frequency']
                    continue
                else:
                    rule_list.append(rule_txt)
                    rule_idx = len(rule_list) - 1
                # conditions
                _list_conditions = []
                for k in node_rule.keys():
                    if k not in self.non_rule_keys:
                        if k in condition_list:
                            lit_idx = condition_list.index(k)
                        else:
                            condition_list.append(k)
                            lit_idx = len(condition_list) - 1
                            lit_obj_list.append(Condition(lit_idx, k))
                        _list_conditions.append((rule_idx, lit_idx))
                if self.verbose:
                    print('rule_idx: {} class={}: {}'.format(rule_idx, node_rule['predict_class'], rule_txt))
                    print('conditions: {}'.format(['condition{}'.format(x) for x in _list_conditions]))
                    print('support: {}'.format(node_rule['frequency']))
                    print('=' * 30 + '\n')

                # create new dict
                prn = {
                    'rule_idx': rule_idx,
                    'rule': rule_txt,
                    'conditions': [x for x in _list_conditions],
                    'support': node_rule['frequency_am'],
                    'size': len(_list_conditions),
                    'error_rate': node_rule['error_rate'],
                    'accuracy': node_rule['accuracy'],
                    'precision': node_rule['precision'],
                    'recall': node_rule['recall'],
                    'f1_score': node_rule['f1_score'],
                    'predict_class': node_rule['predict_class']
                }
                print_dicts.append(prn)
                rl_obj_list.append(Rule(idx=rule_idx,
                                         rule_str=rule_txt,
                                         conditions=[Condition(itm_idx_k, condition_list[itm_idx_k])
                                                   for (_, itm_idx_k) in _list_conditions],
                                         support=int(round(node_rule['frequency_am']*100)),
                                         size=len(_list_conditions),
                                         error_rate=int(round(node_rule['error_rate']*100)),
                                         accuracy=int(round(node_rule['accuracy'] * 100)),
                                         precision=int(round(node_rule['precision'] * 100)),
                                         recall=int(round(node_rule['recall'] * 100)),
                                         f1_score=int(round(node_rule['f1_score'] * 100)),
                                         predict_class=node_rule['predict_class']))
        self.rules_ = rl_obj_list
        self.conditions_ = lit_obj_list
        return print_dicts

    def asp_str_from_dicts(self, list_dicts: list):
        print_lines = []
        for rule_dict in list_dicts:
            prn = []
            ptn_idx = rule_dict['rule_idx']
            prn.append('rule({}).'.format(ptn_idx))
            for x in rule_dict['conditions']:
                prn.append('condition({},{}).'.format(x[0], x[1]))
            prn.append('support({},{}).'.format(ptn_idx, int(round(rule_dict['support'] * 100))))
            prn.append('size({},{}).'.format(ptn_idx, rule_dict['size']))
            prn.append('accuracy({},{}).'.format(ptn_idx, int(round(rule_dict['accuracy'] * 100))))
            prn.append('precision({},{}).'.format(ptn_idx, int(round(rule_dict['precision'] * 100))))
            prn.append('recall({},{}).'.format(ptn_idx, int(round(rule_dict['recall'] * 100))))
            prn.append('f1_score({},{}).'.format(ptn_idx, int(round(rule_dict['f1_score'] * 100))))
            prn.append('error_rate({},{}).'.format(ptn_idx, int(round(rule_dict['error_rate'] * 100))))
            prn.append('predict_class({},{}).'.format(ptn_idx, rule_dict['predict_class']))
            print_lines.append(' '.join(prn))
        return_str = '\n'.join(print_lines)
        return return_str


class LGBMTree:
    # mostly copied from shap's tree wrapper
    def __init__(self, tree):
        if 'tree_structure' not in tree or type(tree) != dict:
            raise ValueError('unsupported tree type: {}'.format(type(tree)))
        start = tree['tree_structure']
        num_parents = tree['num_leaves'] - 1
        self.TREE_LEAF = -1

        self.raw_tree_string = tree
        self.tree_index = tree['tree_index']
        self.decision_type = np.empty((2 * num_parents + 1), dtype=np.object)
        self.children_left = np.empty((2 * num_parents + 1), dtype=np.int32)
        self.children_right = np.empty((2 * num_parents + 1), dtype=np.int32)
        self.children_default = np.empty((2 * num_parents + 1), dtype=np.int32)
        self.features = np.empty((2 * num_parents + 1), dtype=np.int32)
        # self.thresholds = np.empty((2 * num_parents + 1), dtype=np.float64)
        self.thresholds = [np.nan] * (2 * num_parents + 1)
        self.values = [-2] * (2 * num_parents + 1)
        self.node_sample_weight = np.empty((2 * num_parents + 1), dtype=np.float64)
        visited, queue = [], [start]
        while queue:
            vertex = queue.pop(0)
            if 'split_index' in vertex.keys():
                vertex_split_index = vertex['split_index']
                if vertex_split_index not in visited:
                    # check whether left child is a leaf
                    if 'split_index' in vertex['left_child'].keys():
                        self.children_left[vertex_split_index] = vertex['left_child']['split_index']
                    else:
                        self.children_left[vertex_split_index] = vertex['left_child']['leaf_index'] + num_parents
                    # check whether right child is a leaf
                    if 'split_index' in vertex['right_child'].keys():
                        self.children_right[vertex_split_index] = vertex['right_child']['split_index']
                    else:
                        self.children_right[vertex_split_index] = vertex['right_child']['leaf_index'] + num_parents
                    # default True split
                    if vertex['default_left']:
                        self.children_default[vertex_split_index] = self.children_left[vertex_split_index]
                    else:
                        self.children_default[vertex_split_index] = self.children_right[vertex_split_index]
                    # other data
                    self.decision_type[vertex_split_index] = vertex['decision_type']
                    self.features[vertex_split_index] = vertex['split_feature']
                    self.thresholds[vertex_split_index] = vertex['threshold']
                    self.values[vertex_split_index] = [vertex['internal_value']]
                    self.node_sample_weight[vertex_split_index] = vertex['internal_count']
                    visited.append(vertex_split_index)
                    queue.append(vertex['left_child'])
                    queue.append(vertex['right_child'])
            else:  # at leaf
                try:  # case where the root is the only node
                    vertex_leaf_index = vertex['leaf_index']
                except KeyError:
                    vertex_leaf_index = 0
                vertex_leaf_node_index = vertex_leaf_index + num_parents

                self.decision_type[vertex_leaf_node_index] = self.TREE_LEAF
                self.children_left[vertex_leaf_node_index] = self.TREE_LEAF
                self.children_right[vertex_leaf_node_index] = self.TREE_LEAF
                self.children_default[vertex_leaf_node_index] = self.TREE_LEAF
                self.features[vertex_leaf_node_index] = self.TREE_LEAF
                self.thresholds[vertex_leaf_node_index] = self.TREE_LEAF
                self.values[vertex_leaf_node_index] = [vertex['leaf_value']]
                try:  # case where the root is the only node
                    self.node_sample_weight[vertex_leaf_node_index] = vertex['leaf_count']
                except KeyError:
                    self.node_sample_weight[vertex_leaf_node_index] = 0
        self.values = np.asarray(self.values)
        # self.values = np.multiply(self.values, scaling)  # scaling is not supported


class LGBMRuleExtractor:
    def __init__(self, verbose=False):
        self.feature_names = None
        self.verbose = verbose
        self.TREE_LEAF = -1
        self.non_rule_keys = ['class', 'condition_length', 'error_rate',
                              'precision', 'recall', 'f1_score',
                              'frequency', 'frequency_am', 'predict_class', 'value',
                              'is_tree_max', 'accuracy']
        self.asp_fact_str = None
        self.fitted_ = False
        self.rules_ = None
        self.conditions_ = None
        self.num_tree_per_iteration = None
        self.metric_averaging = None

    def fit(self, X, y, model=None, feature_names=None, **params):
        if not isinstance(model, lgb.Booster):
            raise ValueError('unsupported model type, expected lgb.Booster but got {}'.format(type(model)))

        if feature_names is None:
            self.feature_names = ['feature_{}'.format(i) for i in range(X.shape[0])]
        else:
            self.feature_names = feature_names

        num_classes = y.nunique()
        self.metric_averaging = 'micro' if num_classes > 2 else 'binary'

        model_dump = model.dump_model()
        self.num_tree_per_iteration = model_dump['num_tree_per_iteration']

        with timer_exec('[LGBM RuleExtractor] LGBMTree init'):
            with Pool(processes=cpu_count()) as pool:
                lgbtrees = pool.map(LGBMTree, model_dump['tree_info'])
        # lgbtrees = [LGBMTree(x) for x in model_dump['tree_info']]

        rules = {}
        with timer_exec('[LGBM RuleExtractor] export text rule tree'):
            with Pool(processes=cpu_count()) as pool:
                _tmp_rules = pool.map(self.export_text_rule_tree, lgbtrees)

        # for t_idx, tree in enumerate(lgbtrees):
        #     _, rules[t_idx] = self.export_text_rule_tree(tree)
        with timer_exec('[LGBM RuleExtractor] enumerate rules'):
            for t_idx, tree in enumerate(_tmp_rules):
                _, rules[t_idx] = _tmp_rules[t_idx]

        with timer_exec('[LGBM RuleExtractor] parallel export text rule'):
            rules = self.export_text_rule_lgb_parallel(rules, X, y)
        # rules = self.export_text_rule_lgb(rules, X, y)

        with timer_exec('[LGBM RuleExtractor] asp dict from rules'):
            printable_dicts = self.asp_dict_from_rules(rules)

        with timer_exec('[LGBM RuleExtractor] print asp'):
            print_str = self.asp_str_from_dicts(printable_dicts)
        self.asp_fact_str = print_str
        self.fitted_ = True
        return self

    def transform(self, X, y=None, **params):
        return self.asp_fact_str

    def export_text_rule_tree(self, tree):
        leaf_idx = np.unique(np.where(tree.children_left == self.TREE_LEAF) +
                             np.where(tree.children_right == self.TREE_LEAF))
        paths = {}
        for leaf in leaf_idx:
            path_leaf = []
            self.find_path_recursive(0, path_leaf, leaf, tree.children_left, tree.children_right)
            paths[leaf] = np.unique(np.sort(path_leaf))

        rules = {}
        for key in paths:
            rule = self.get_rule(paths[key], self.feature_names, tree)
            if rule == {}:  # skip empty rule - single node tree
                continue
            else:
                rules[key] = rule
        return paths, rules

    def find_path_recursive(self, node_idx, path, x, children_left, children_right):
        path.append(node_idx)
        if node_idx == x:
            return True
        left, right = False, False
        if children_left[node_idx] != self.TREE_LEAF:
            left = self.find_path_recursive(children_left[node_idx], path, x, children_left, children_right)
        if children_right[node_idx] != self.TREE_LEAF:
            right = self.find_path_recursive(children_right[node_idx], path, x, children_left, children_right)
        if left or right:
            return True
        path.remove(node_idx)
        return False

    def get_rule(self, path, feature_names, tree):
        # find the max leaf for this tree
        # TODO: this might have to change from leaf_value to leaf_weight or sample_count
        leaf_ids = np.where(tree.decision_type == tree.TREE_LEAF)[0]
        leaf_values = tree.values[leaf_ids]
        leaf_argmax = np.argmax(leaf_values)
        leaf_max = leaf_ids[leaf_argmax]

        if len(path) == 1:  # only root node case
            return {}
        rule_dict = {}
        for idx, node in enumerate(path):
            if idx != len(path) - 1:  # at internal nodes
                # NB. you need complementary splits eg. (<=, >), (>=, <), (==, !=)
                if tree.decision_type[node] == '<=':
                    if tree.children_left[node] == path[idx + 1]:
                        rule_str = '{} <= {}'.format(feature_names[tree.features[node]], tree.thresholds[node])
                    # right-child handles >
                    else:
                        rule_str = '{} > {}'.format(feature_names[tree.features[node]], tree.thresholds[node])
                elif tree.decision_type[node] == '==':
                    if tree.children_left[node] == path[idx + 1]:
                        rule_str = '{} == {}'.format(feature_names[tree.features[node]], tree.thresholds[node])
                    else:
                        rule_str = '{} != {}'.format(feature_names[tree.features[node]], tree.thresholds[node])
                else:
                    raise ValueError('this decision type {} is not supported'.format(tree.decision_type[node]))
                rule_dict[rule_str] = 1
            else:  # at leaf
                # class_name is determined by the tree index, as there are n_class trees in one round
                if self.num_tree_per_iteration > 1:
                    class_name = tree.tree_index % self.num_tree_per_iteration
                else:
                    class_name = 1
                rule_dict['class'] = class_name
                rule_dict['value'] = int(tree.node_sample_weight[node])  # number of supporting examples
                if node == leaf_max:
                    rule_dict['is_tree_max'] = True
                else:
                    rule_dict['is_tree_max'] = False
        return rule_dict

    def export_text_rule_lgb_parallel(self, tree_rules, X, y):
        _tmp_rules = tree_rules
        rules = {}
        # adding rule statistics
        with Pool(cpu_count()) as pool:
            # make same number of x y
            mod_rules = pool.starmap(self.export_text_rule_lgb_parallel_sub,
                                     ((t_rules, X, y) for t_rules in _tmp_rules.values()))
        for t_idx, tree in enumerate(mod_rules):
            rules[t_idx] = mod_rules[t_idx]
        return rules

    def export_text_rule_lgb_parallel_sub(self, t_rules, X, y, classes=None):
        if classes is None:
            classes = [0, 1]
        rules = t_rules
        for path_rule in rules.values():
            _tmp_dfs = []
            for rule_key in path_rule.keys():
                # skip non-conditions
                if rule_key in self.non_rule_keys:
                    continue
                # collect conditions and boolean mask
                else:
                    # TODO: cache result for better performance
                    if LT_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(LT_PATTERN, 1)
                        _tmp_dfs.append(getattr(X[_rule_field], 'lt')(float(_rule_threshold)))
                    elif LE_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(LE_PATTERN, 1)
                        _tmp_dfs.append(getattr(X[_rule_field], 'le')(float(_rule_threshold)))
                    elif GT_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(GT_PATTERN, 1)
                        _tmp_dfs.append(getattr(X[_rule_field], 'gt')(float(_rule_threshold)))
                    elif GE_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(GE_PATTERN, 1)
                        _tmp_dfs.append(getattr(X[_rule_field], 'ge')(float(_rule_threshold)))
                    elif EQ_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(EQ_PATTERN, 1)
                        inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                        # rule threshold in this case can be like '0||1||2'
                        _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                        _tmp_dfs.append(getattr(X[_rule_field], 'isin')(_cat_th))
                    elif NEQ_PATTERN in rule_key:
                        _rule_field, _rule_threshold = rule_key.rsplit(NEQ_PATTERN, 1)
                        inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                        # rule threshold in this case can be like '0||1||2'
                        _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                        # note the bitwise complement ~
                        _tmp_dfs.append(~ getattr(X[_rule_field], 'isin')(_cat_th))
                    else:
                        raise ValueError('No key found')
            # reduce boolean mask
            mask_res = reduce(and_, _tmp_dfs)
            # these depend on the entire training data, not on the bootstrapped data that the original rf uses
            # path_rule['predict_class'] = y[mask_res].mode()[0]
            # y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
            # path_rule['condition_length'] = len(_tmp_dfs)
            # path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
            # path_rule['frequency'] = len(y[mask_res])
            # path_rule['error_rate'] = 1 - accuracy_score(y[mask_res], y_pred)
            # path_rule['precision'] = precision_score(y[mask_res], y_pred,
            #                                         #  pos_label=path_rule['predict_class'],
            #                                          average=self.metric_averaging,
            #                                          zero_division=0)
            # path_rule['recall'] = recall_score(y[mask_res], y_pred,
            #                                 #    pos_label=path_rule['predict_class'],
            #                                    average=self.metric_averaging,
            #                                    zero_division=0)
            # do not subset by mask, use whole dataset
            if y.nunique() != 2:
                raise RuntimeError('only binary classification is supported at this moment')
            path_rule['predict_class'] = 1
            y_pred = np.zeros(y.shape)
            y_pred[mask_res] = 1
            # y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
            path_rule['condition_length'] = len(_tmp_dfs)
            path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
            path_rule['frequency'] = len(y[mask_res])
            path_rule['error_rate'] = 1 - accuracy_score(y, y_pred)
            path_rule['accuracy'] = accuracy_score(y, y_pred)
            path_rule['precision'] = precision_score(y, y_pred,
                                                     #  pos_label=path_rule['predict_class'],
                                                     average=self.metric_averaging,
                                                     zero_division=0)
            path_rule['recall'] = recall_score(y, y_pred,
                                               #    pos_label=path_rule['predict_class'],
                                               average=self.metric_averaging,
                                               zero_division=0)
            path_rule['f1_score'] = f1_score(y, y_pred,
                                             # pos_label=path_rule['predict_class'],
                                             average=self.metric_averaging, zero_division=0)

        return rules

    def export_text_rule_lgb(self, tree_rules, X, y):
        rules = tree_rules
        # adding rule statistics
        for t_idx, t_rules in rules.items():
            for path_rule in t_rules.values():
                _tmp_dfs = []
                for rule_key in path_rule.keys():
                    # skip non-conditions
                    if rule_key in self.non_rule_keys:
                        continue
                    # collect conditions and boolean mask
                    else:
                        if LT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'lt')(float(_rule_threshold)))
                        elif LE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'le')(float(_rule_threshold)))
                        elif GT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GT_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'gt')(float(_rule_threshold)))
                        elif GE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GE_PATTERN, 1)
                            _tmp_dfs.append(getattr(X[_rule_field], 'ge')(float(_rule_threshold)))
                        elif EQ_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(EQ_PATTERN, 1)
                            inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                            # rule threshold in this case can be like '0||1||2'
                            _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                            _tmp_dfs.append(getattr(X[_rule_field], 'isin')(_cat_th))
                        elif NEQ_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(NEQ_PATTERN, 1)
                            inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                            # rule threshold in this case can be like '0||1||2'
                            _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                            # note the bitwise complement ~
                            _tmp_dfs.append(~ getattr(X[_rule_field], 'isin')(_cat_th))
                        else:
                            raise ValueError('No key found')
                # reduce boolean mask
                mask_res = reduce(and_, _tmp_dfs)
                # these depend on the entire training data, not on the bootstrapped data that the original rf uses
                # path_rule['predict_class'] = y[mask_res].mode()[0]
                path_rule['predict_class'] = 1
                y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
                path_rule['condition_length'] = len(_tmp_dfs)
                path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
                path_rule['frequency'] = len(y[mask_res])  # coverage
                path_rule['error_rate'] = 1 - accuracy_score(y[mask_res], y_pred)
                path_rule['accuracy'] = accuracy_score(y[mask_res], y_pred)
                path_rule['precision'] = precision_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                         average=self.metric_averaging)
                path_rule['recall'] = recall_score(y[mask_res], y_pred, pos_label=path_rule['predict_class'],
                                                   average=self.metric_averaging)
                if y.nunique() != 2:
                    raise RuntimeError('only binary classification is supported at this moment')

        return rules

    def asp_dict_from_rules(self, rules: dict):
        """
        Construct intermediate list of dicts that can be processed into ASP strings.
        In this implementation, duplicate patterns are merged by aggregating the support (frequency).

        Args:
            rules: dict of dicts containing rules

        Returns:
            list of dicts
        """
        rule_list = []
        condition_list = []
        print_dicts = []

        # pattern and item instances
        rl_obj_list = []
        lit_obj_list = []

        for t_idx, t in rules.items():
            for node_idx, node_rule in t.items():
                # if not node_rule['is_tree_max']:
                #     continue  # skip non_max case

                # pattern
                rule_txt = ' AND '.join([k for k in node_rule.keys() if k not in self.non_rule_keys])

                if rule_txt in rule_list:
                    # rule_idx = rule_list.index(rule_txt)
                    # # add support
                    # for p in print_dicts:
                    #     # TODO: do not accumulate
                    #     if p['rule_idx'] == rule_idx:
                    #         p['support'] += node_rule['frequency']
                    continue
                else:
                    rule_list.append(rule_txt)
                    rule_idx = len(rule_list) - 1
                # items
                _list_conditions = []
                for k in node_rule.keys():
                    if k not in self.non_rule_keys:
                        if k in condition_list:
                            lit_idx = condition_list.index(k)
                        else:
                            condition_list.append(k)
                            lit_idx = len(condition_list) - 1
                            lit_obj_list.append(Condition(lit_idx, k))
                        _list_conditions.append((rule_idx, lit_idx))
                if self.verbose:
                    print('rule_id {} class={}: {}'.format(rule_idx, node_rule['predict_class'], rule_txt))
                    print('conditions: {}'.format(['condition{}'.format(x) for x in _list_conditions]))
                    print('support: {}'.format(node_rule['frequency']))
                    print('=' * 30 + '\n')

                # create new dict
                prn = {
                    'rule_idx': rule_idx,
                    'rule': rule_txt,
                    'conditions': [x for x in _list_conditions],
                    'support': node_rule['frequency_am'],
                    'size': len(_list_conditions),
                    'error_rate': node_rule['error_rate'],
                    'accuracy': node_rule['accuracy'],
                    'precision': node_rule['precision'],
                    'recall': node_rule['recall'],
                    'f1_score': node_rule['f1_score'],
                    'predict_class': node_rule['predict_class']
                }
                print_dicts.append(prn)
                rl_obj_list.append(Rule(idx=rule_idx,
                                        rule_str=rule_txt,
                                        conditions=[Condition(itm_idx_k, condition_list[itm_idx_k])
                                                   for (_, itm_idx_k) in _list_conditions],
                                        support=int(round(node_rule['frequency_am'] * 100)),
                                        size=len(_list_conditions),
                                        accuracy=int(round(node_rule['accuracy'] * 100)),
                                        error_rate=int(round(node_rule['error_rate'] * 100)),
                                        precision=int(round(node_rule['precision'] * 100)),
                                        recall=int(round(node_rule['recall'] * 100)),
                                        f1_score=int(round(node_rule['f1_score'] * 100)),
                                        predict_class=node_rule['predict_class']))
        self.rules_ = rl_obj_list
        self.conditions_ = lit_obj_list
        return print_dicts

    def asp_str_from_dicts(self, list_dicts: list):
        print_lines = []
        for rule_dict in list_dicts:
            prn = []
            rule_idx = rule_dict['rule_idx']
            prn.append('rule({}).'.format(rule_idx))
            for x in rule_dict['conditions']:
                prn.append('condition({},{}).'.format(x[0], x[1]))
            prn.append('support({},{}).'.format(rule_idx, int(round(rule_dict['support'] * 100))))
            prn.append('size({},{}).'.format(rule_idx, rule_dict['size']))
            prn.append('accuracy({},{}).'.format(rule_idx, int(round(rule_dict['accuracy'] * 100))))
            prn.append('error_rate({},{}).'.format(rule_idx, int(round(rule_dict['error_rate'] * 100))))
            prn.append('precision({},{}).'.format(rule_idx, int(round(rule_dict['precision'] * 100))))
            prn.append('recall({},{}).'.format(rule_idx, int(round(rule_dict['recall'] * 100))))
            prn.append('f1_score({},{}).'.format(rule_idx, int(round(rule_dict['f1_score'] * 100))))
            prn.append('predict_class({},{}).'.format(rule_idx, rule_dict['predict_class']))
            print_lines.append(' '.join(prn))
        return_str = '\n'.join(print_lines)
        return return_str


class LGBMLocalRuleExtractor(LGBMRuleExtractor):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        self.non_rule_keys.append('leaf_idx')

    def fit(self, X, y, model=None, feature_names=None, **params):
        if not isinstance(model, lgb.Booster):
            raise ValueError('unsupported model type, expected lgb.Booster but got {}'.format(type(model)))

        if feature_names is None:
            self.feature_names = ['feature_{}'.format(i) for i in range(X.shape[0])]
        else:
            self.feature_names = feature_names

        num_classes = y.nunique()
        self.metric_averaging = 'micro' if num_classes > 2 else 'binary'

        model_dump = model.dump_model()
        self.num_tree_per_iteration = model_dump['num_tree_per_iteration']

        with timer_exec('[LGBM Local RuleExtractor] LGBMTree init'):
            with Pool(processes=cpu_count()) as pool:
                lgbtrees = pool.map(LGBMTree, model_dump['tree_info'])
        # lgbtrees = [LGBMTree(x) for x in model_dump['tree_info']]

        rules = {}
        with timer_exec('[LGBM Local RuleExtractor] export text rule tree'):
            with Pool(processes=cpu_count()) as pool:
                _tmp_rules = pool.map(self.export_text_rule_tree, lgbtrees)

        # for t_idx, tree in enumerate(lgbtrees):
        #     _, rules[t_idx] = self.export_text_rule_tree(tree)
        with timer_exec('[LGBM Local RuleExtractor] enumerate rules'):
            for t_idx, tree in enumerate(_tmp_rules):
                _, rules[t_idx] = _tmp_rules[t_idx]

        with timer_exec('[LGBM Local RuleExtractor] parallel export text rule'):
            rules = self.export_text_rule_lgb_parallel(rules, X, y)
        # rules = self.export_text_rule_lgb(rules, X, y)

        with timer_exec('[LGBM Local RuleExtractor] asp dict from rules'):
            self.asp_dict_from_rules(rules)

        # with timer_exec('[LGBM Local RuleExtractor] print asp'):
        #     print_str = self.asp_str_from_dicts(printable_dicts)
        # self.asp_fact_str = print_str
        self.fitted_ = True
        return self

    def export_text_rule_lgb_parallel(self, tree_rules, X, y):
        if y.nunique() != 2:
            raise RuntimeError('only binary classification is supported at this moment')
        # {tree_idx: {leaf_idx: {condition_1: 1, condition_2: 1, ...},
        #             leaf_idx: {...}},
        #  tree_idx: {leaf_idx: {...},...},
        #  ...}
        _tmp_rules = tree_rules
        rules = {}
        classes = y.unique()
        # adding rule statistics
        with Pool(cpu_count()) as pool:
            # make same number of x y
            mod_rules = pool.starmap(self.export_text_rule_lgb_parallel_sub,
                                     ((t_rules, X, y, classes) for t_rules in _tmp_rules.values()))
        for t_idx, tree in enumerate(mod_rules):
            rules[t_idx] = mod_rules[t_idx]
        return rules

    def export_text_rule_lgb_parallel_sub(self, t_rules, X, y, classes=None):
        if classes is None:
            classes = [0, 1]

        # {leaf_idx: {condition_1: 1, condition_2: 1, ...},
        #  leaf_idx: {}, ...}
        ret_rules = dict()
        cond_cache = dict()

        for cls in classes:
            for leaf_idx, t_path_rule in t_rules.items():
                path_rule = deepcopy(t_path_rule)
                _tmp_dfs = []
                for rule_key in path_rule.keys():
                    # skip non-conditions
                    if rule_key in self.non_rule_keys:
                        continue
                    # collect conditions and boolean mask
                    else:
                        if rule_key in cond_cache:
                            _tmp_mask = cond_cache[rule_key]
                        elif LT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LT_PATTERN, 1)
                            _tmp_mask = getattr(X[_rule_field], 'lt')(float(_rule_threshold))
                        elif LE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(LE_PATTERN, 1)
                            _tmp_mask = getattr(X[_rule_field], 'le')(float(_rule_threshold))
                        elif GT_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GT_PATTERN, 1)
                            _tmp_mask = getattr(X[_rule_field], 'gt')(float(_rule_threshold))
                        elif GE_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(GE_PATTERN, 1)
                            _tmp_mask = getattr(X[_rule_field], 'ge')(float(_rule_threshold))
                        elif EQ_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(EQ_PATTERN, 1)
                            inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                            # rule threshold in this case can be like '0||1||2'
                            _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                            _tmp_mask = getattr(X[_rule_field], 'isin')(_cat_th)
                        elif NEQ_PATTERN in rule_key:
                            _rule_field, _rule_threshold = rule_key.rsplit(NEQ_PATTERN, 1)
                            inverse_map = dict(enumerate(X[_rule_field].cat.categories))
                            # rule threshold in this case can be like '0||1||2'
                            _cat_th = [inverse_map[int(x)] for x in _rule_threshold.split('||', -1)]
                            # note the bitwise complement ~
                            _tmp_mask = ~ getattr(X[_rule_field], 'isin')(_cat_th)
                        else:
                            raise ValueError('No key found')
                        _tmp_dfs.append(_tmp_mask)
                # reduce boolean mask
                mask_res = reduce(and_, _tmp_dfs)

                path_rule['leaf_idx'] = leaf_idx
                path_rule['predict_class'] = cls
                y_pred = np.zeros(y.shape)
                y_pred[mask_res] = cls
                # y_pred = [path_rule['predict_class'] for _ in range(len(y[mask_res]))]
                path_rule['condition_length'] = len(_tmp_dfs)
                path_rule['frequency_am'] = len(y[mask_res]) / len(y)  # anti-monotonic
                path_rule['frequency'] = len(y[mask_res])
                path_rule['error_rate'] = 1 - accuracy_score(y, y_pred)
                path_rule['accuracy'] = accuracy_score(y, y_pred)
                path_rule['precision'] = precision_score(y, y_pred,
                                                         #  pos_label=path_rule['predict_class'],
                                                         average=self.metric_averaging,
                                                         zero_division=0)
                path_rule['recall'] = recall_score(y, y_pred,
                                                   #    pos_label=path_rule['predict_class'],
                                                   average=self.metric_averaging,
                                                   zero_division=0)
                path_rule['f1_score'] = f1_score(y, y_pred,
                                                 # pos_label=path_rule['predict_class'],
                                                 average=self.metric_averaging, zero_division=0)
                ret_rules['{}_{}'.format(leaf_idx, cls)] = path_rule

        # return {leaf_idx+cls: {accuracy: , recall: ,...}}
        # e.g. {6_1: {accuracy: ..., recall: ...,}, 7_1: {...}, ...}
        return ret_rules

    def asp_dict_from_rules(self, rules: dict):
        """
        Construct intermediate list of dicts that can be processed into ASP strings.
        In this implementation, duplicate patterns are merged by aggregating the support (frequency).

        Args:
            rules: dict of dicts containing rules

        Returns:
            list of dicts
        """
        rule_list = []

        condition_list = []
        print_dicts = []

        # pattern and item instances
        rl_obj_list = []
        lit_obj_list = []

        path_rules = {}

        # {tree_idx: {leaf_idx_class: {}, leaf_idx_class: {}}, tree_idx: {}, ...}
        for t_idx, t in rules.items():
            for node_idx, node_rule in t.items():
                # if not node_rule['is_tree_max']:
                #     continue  # skip non_max case

                # pattern
                rule_txt = ' AND '.join([k for k in node_rule.keys() if k not in self.non_rule_keys])
                rule_list.append(rule_txt)
                rule_idx = len(rule_list) - 1

                # items
                _list_conditions = []
                for k in node_rule.keys():
                    if k not in self.non_rule_keys:
                        if k in condition_list:
                            lit_idx = condition_list.index(k)
                        else:
                            condition_list.append(k)
                            lit_idx = len(condition_list) - 1
                            lit_obj_list.append(Condition(lit_idx, k))
                        _list_conditions.append((rule_idx, lit_idx))
                if self.verbose:
                    print('rule_id {} class={}: {}'.format(rule_idx, node_rule['predict_class'], rule_txt))
                    print('conditions: {}'.format(['condition{}'.format(x) for x in _list_conditions]))
                    print('support: {}'.format(node_rule['frequency']))
                    print('=' * 30 + '\n')

                # print_dicts.append(prn)
                rule = Rule(idx=rule_idx,
                            rule_str=rule_txt,
                            conditions=[Condition(itm_idx_k, condition_list[itm_idx_k])
                                       for (_, itm_idx_k) in _list_conditions],
                            support=int(round(node_rule['frequency_am'] * 100)),
                            size=len(_list_conditions),
                            accuracy=int(round(node_rule['accuracy'] * 100)),
                            error_rate=int(round(node_rule['error_rate'] * 100)),
                            precision=int(round(node_rule['precision'] * 100)),
                            recall=int(round(node_rule['recall'] * 100)),
                            f1_score=int(round(node_rule['f1_score'] * 100)),
                            predict_class=node_rule['predict_class'])
                rl_obj_list.append(rule)

                # create new dict
                prn = {
                    'rule_idx': rule_idx,
                    'rule': rule,
                    'conditions': [x for x in _list_conditions],
                    'support': node_rule['frequency_am'],
                    'size': len(_list_conditions),
                    'error_rate': node_rule['error_rate'],
                    'accuracy': node_rule['accuracy'],
                    'precision': node_rule['precision'],
                    'recall': node_rule['recall'],
                    'f1_score': node_rule['f1_score'],
                    'predict_class': node_rule['predict_class']
                }

                # node_idx already contains leaf_idx and class
                tree_rule_idx = '{}_{}'.format(t_idx, node_idx)
                path_rules[tree_rule_idx] = prn
        self.all_path_rule = path_rules
        self.rules_ = rl_obj_list
        self.conditions_ = lit_obj_list
        return None

    def asp_str_from_dicts(self, list_dicts: list):
        print_lines = []
        for rule_dict in list_dicts:
            prn = []
            rule_idx = rule_dict['rule_idx']
            prn.append('rule({}).'.format(rule_idx))
            for x in rule_dict['conditions']:
                prn.append('condition({},{}).'.format(x[0], x[1]))
            prn.append('support({},{}).'.format(rule_idx, int(round(rule_dict['support'] * 100))))
            prn.append('size({},{}).'.format(rule_idx, rule_dict['size']))
            prn.append('accuracy({},{}).'.format(rule_idx, int(round(rule_dict['accuracy'] * 100))))
            prn.append('error_rate({},{}).'.format(rule_idx, int(round(rule_dict['error_rate'] * 100))))
            prn.append('precision({},{}).'.format(rule_idx, int(round(rule_dict['precision'] * 100))))
            prn.append('recall({},{}).'.format(rule_idx, int(round(rule_dict['recall'] * 100))))
            prn.append('f1_score({},{}).'.format(rule_idx, int(round(rule_dict['f1_score'] * 100))))
            prn.append('predict_class({},{}).'.format(rule_idx, rule_dict['predict_class']))
            print_lines.append(' '.join(prn))
        return_str = '\n'.join(print_lines)
        return return_str
