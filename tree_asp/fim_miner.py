import pandas as pd
import numpy as np
import optuna
import json
import fim
import os
import subprocess

from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from utils import timer_exec, load_data
from mdlp.mdlp import MDLPDiscretizer
from timeit import default_timer as timer
from clasp_parser import generate_answers
from classifier import RuleClassifier
from rule import Rule
from typing import List


class TransactionItem:
    def __init__(self, attr, value):
        self.attr = attr if type(attr) is str else str(attr)
        self.value = value if type(value) is str else str(value)

    def __eq__(self, other):
        return hash((self.attr, self.value)) == hash((other.attr, other.value))

    def __str__(self):
        return '{}={}'.format(self.attr, self.value)

    def __repr__(self):
        return 'TransactionItem({},{})'.format(self.attr, self.value)


class Transaction:
    def __init__(self, row: pd.Series, target_label=None):
        self.target_label = target_label
        self.items = []
        self.string_items = []
        self.class_item = None
        self.idx = None

        for attr, value in row.iteritems():
            itm = TransactionItem(attr, value)
            if attr == self.target_label:
                self.class_item = itm
            self.items.append(itm)
            self.string_items.append("{}={}".format(itm.attr, itm.value))


class TransactionDatabase:
    def __init__(self, dataset: pd.DataFrame, target_column=None):
        self.target_column = target_column
        self.class_labels = set()
        self.string_dataset = []
        self.transaction_dataset = []
        self.dataframe = dataset

        for idx, row in dataset.iterrows():
            trn = Transaction(row, target_label=target_column)
            # trn.idx = idx  # this is pandas index not transaction index if shuffled
            self.transaction_dataset.append(trn)
            self.class_labels.add(str(trn.class_item))
            self.string_dataset.append(trn.string_items)


class AssociationRule:
    def __init__(self, rule_idx, head, body, support, confidence):
        self.rule_idx = rule_idx
        self.head = head
        self.body = body
        self.support = support
        self.confidence = confidence
        self.rule_length = len(body) + 1  # count head as 1

    def __gt__(self, other):
        if self.confidence > other.confidence:
            return True
        elif self.confidence == other.confidence and self.support > other.support:
            return True
        elif (self.confidence == other.confidence and self.support == other.support and
              self.rule_length < other.rule_length):
            return True
        elif (self.confidence == other.confidence and self.support == other.support and
              self.rule_length == other.rule_length and self.rule_idx < other.rule_idx):
            return True
        else:
            return False

    def __lt__(self, other):
        return not self > other

    def __str__(self):
        return '{} <= {}'.format(self.head, self.body)


class AssociationRuleClassifier:
    def __init__(self, association_rules: List[AssociationRule], default_class=None, mode='first'):
        self.rules = association_rules
        self.default_class = default_class
        self.mode = mode

    def fit(self, X, y):
        if not self.default_class:
            self.default_class = y.mode()[0]
        return self

    def predict(self, X: pd.DataFrame):
        db_size = X.shape[0]
        predictions = np.zeros(db_size)

        # 2 modes - 1. first rule that fires 2. majority voting on the fired rules (ignore default rules)
        # first rule fired mask (true for fired)
        # get first false
        # apply second rule
        # previous false and this one true
        # get second false
        # apply third rule
        # first+second false and this one true
        if self.mode == 'first':
            covered_idx = np.array([False] * db_size)
            literal_cache = {}
            for ar in self.rules:
                body_mask = np.array([True] * db_size)
                # body loop
                for literal in ar.body:
                    # if we only care about discretized datasets we don't need to worry about interval evaluation
                    # just need to check value equality
                    if literal in literal_cache:
                        literal_mask = literal_cache[literal]
                    else:
                        attr, val = literal.split('=')
                        literal_mask = np.array([X[attr].values.astype(str) == val])
                        literal_mask = literal_mask.reshape(db_size)
                        literal_cache[literal] = literal_mask
                    body_mask &= literal_mask
                # body_cover_idx = np.where(body_mask)[0]  # index covered by this rule
                # only predict those with newly covered
                # before | after | result
                # True   | True  | False
                # True   | False | False
                # False  | True  | True
                # False  | False | False
                before_cover = deepcopy(covered_idx)
                change = ~before_cover & body_mask
                predictions[change] = ar.head.split('=')[1]
                covered_idx |= body_mask
                if all(covered_idx):
                    break  # finish prediction
            return predictions
        elif self.mode == 'vote':
            # for each rule construct body mask
            literal_cache = {}
            rule_predictions = [[] for l in range(db_size)]
            for ar in self.rules:
                body_mask = np.array([True] * db_size)
                # body loop
                for literal in ar.body:
                    # if we only care about discretized datasets we don't need to worry about interval evaluation
                    # just need to check value equality
                    if literal in literal_cache:
                        literal_mask = literal_cache[literal]
                    else:
                        attr, val = literal.split('=')
                        literal_mask = np.array([X[attr].values.astype(str) == val])
                        literal_mask = literal_mask.reshape(db_size)
                        literal_cache[literal] = literal_mask
                    body_mask &= literal_mask
                body_cover_idx = np.where(body_mask)[0]  # index covered by this rule
                head_val = ar.head.split('=')[1]
                for bc_idx in body_cover_idx:
                    rule_predictions[bc_idx].append(head_val)
            # majority voting
            for d_idx, head_list in enumerate(rule_predictions):
                uniq, cnt = np.unique(head_list, return_counts=True)
                max_idx = np.argmax(cnt)
                predictions[d_idx] = uniq[max_idx]
            return predictions
        else:
            raise ValueError('Unknown mode type: {}'.format(self.mode))


def calculate_rule_statistics(tdb: TransactionDatabase, rules):
    db_size = len(tdb.string_dataset)
    literal_cache = {}
    return_list = []
    # rule loop
    for idx, (head, body, sup, conf) in enumerate(rules):  # head=str, body=('str','str'), sup=float, conf=float
        # body loop
        body_mask = np.array([True] * db_size)
        for literal in body:
            # if we only care about discretized datasets we don't need to worry about interval evaluation
            # just need to check value equality
            if literal in literal_cache:
                literal_mask = literal_cache[literal]
            else:
                attr, val = literal.split('=')
                literal_mask = np.array([tdb.dataframe[attr].values.astype(str) == val])
                literal_mask = literal_mask.reshape(db_size)
                literal_cache[literal] = literal_mask
            body_mask &= literal_mask
        body_match_count = tdb.dataframe.loc[body_mask,:].shape[0]
        # head
        if head in literal_cache:
            head_mask = literal_cache[head]
        else:
            attr, val = head.split('=')
            head_mask = np.array([tdb.dataframe[attr].values.astype(str) == val])
            head_mask = head_mask.reshape(db_size)
            literal_cache[head] = head_mask
        # head_match_count = tdb.dataframe.loc[head_mask,:].size
        # both body and head match
        head_and_body_match_count = tdb.dataframe.loc[(head_mask & body_mask),:].shape[0]

        support = body_match_count / db_size
        if body_match_count == 0:
            confidence = 0
        else:
            confidence = head_and_body_match_count / body_match_count
        print('{} :- {} org_sup:{} new_sup:{} org_conf:{} new_conf:{}'.format(
            head, body, sup, support, conf, confidence
        ))
        ar = AssociationRule(rule_idx=idx, head=head, body=body, support=support, confidence=confidence)
        return_list.append(ar)
    return return_list


def get_best_n_rules(tdb: TransactionDatabase, best_n=50, max_rule_count=2000, max_iter=100, timeout=60):
    candidate_rules = []
    min_support = 0.1
    min_confidence = 0.5
    step_confidence = 0.05
    max_rule_length = len(tdb.dataframe.columns)
    zmax = 4  # max rule length
    zmin = 1  # min rule length
    start = timer()
    for iter in range(max_iter):
        appear = {'label=1': 'h', 'label=0': 'h', None: 'b'}
        candidate_rules = fim.arules(tdb.string_dataset, supp=min_support, conf=min_confidence, mode="o",
                                     appear=appear, report='sc', zmax=zmax, zmin=zmin)
        end = timer()
        rule_count = len(candidate_rules)
        # increase maxlen
        if zmax < max_rule_length:
            zmax += 1
        # decrease min confidence
        if min_confidence > step_confidence:
            min_confidence -= step_confidence
        if rule_count > max_rule_count:
            break  # sufficient rules mined
        if (end - start) > timeout:
            break  # return due to timeout
    # sort rules, originally there are at maximum max_rule_count rules
    ars = []
    for r_idx, rule in enumerate(candidate_rules):
        ar = AssociationRule(rule_idx=r_idx, head=rule[0], body=rule[1], support=rule[2], confidence=rule[3])
        ars.append(ar)
    ars.sort(reverse=True)
    return ars[:best_n]


def get_best_n_rules_per_class(tdb: TransactionDatabase, best_n=50, max_rule_count=2000, max_iter=100, timeout=60):
    best_rules = []
    for cls in tdb.class_labels:
        candidate_rules = []
        min_support = 0.2
        min_confidence = 0.5
        step_confidence=0.05
        max_rule_length = len(tdb.dataframe.columns)
        zmax = 4  # max rule length
        zmin = 1  # min rule length
        start = timer()
        for iter in range(max_iter):
            appear = {'label=1': 'h', 'label=0': 'h', None: 'b'}
            for k in tdb.class_labels:
                if k == cls:
                    appear[str(k)] = 'ignore'
            print(appear)
            candidate_rules = fim.arules(tdb.string_dataset, supp=min_support, conf=min_confidence, mode="o",
                                         appear=appear, report='sc', zmax=zmax, zmin=zmin)
            end = timer()
            rule_count = len(candidate_rules)
            # increase maxlen
            if zmax < max_rule_length:
                zmax += 1
            # decrease min confidence
            if min_confidence > step_confidence:
                min_confidence -= step_confidence
            if rule_count > max_rule_count:
                break  # sufficient rules mined
            if (end - start) > timeout:
                break  # return due to timeout
        # sort rules, originally there are at maximum max_rule_count rules
        ars = []
        for r_idx, rule in enumerate(candidate_rules):
            ar = AssociationRule(rule_idx=r_idx, head=rule[0], body=rule[1], support=rule[2], confidence=rule[3])
            ars.append(ar)
        ars.sort(reverse=True)
        best_rules += ars[:best_n]
    return best_rules


def run_experiment(dataset_name):
    X, y = load_data(dataset_name)
    classes = y.unique()
    numerical_features = list(X.select_dtypes(include=['float', 'int']).columns)
    if len(numerical_features) > 0:
        mdlp = MDLPDiscretizer(features=numerical_features)
        Xz = mdlp.fit_transform(X, y)  # np.ndarray
        X = pd.DataFrame(data=Xz, columns=X.columns)
        # one hot encoder cannot handle intervals so treat them as strings
        for col in numerical_features:
            X.loc[:, col] = X.loc[:, col].astype('str').astype('category')
    categorical_features = list(X.columns[X.dtypes == 'category'])
    feat = X.columns

    num_classes = y.nunique()
    metric_averaging = 'micro' if num_classes > 2 else 'binary'
    _values, _counts = np.unique(y, return_counts=True)
    _max_idx = np.argmax(_counts)
    pos_label = _values[_max_idx]

    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for f_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        experiment_tag = 'lgb_{}_{}'.format(dataset_name, f_idx)
        print(experiment_tag)
        start = timer()

        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        tdb = TransactionDatabase(pd.concat([x_train, y_train], axis=1), target_column='label')
        ars = get_best_n_rules_per_class(tdb, best_n=10)
        # for ar in ars:
        #     print('{} <= {} {} {}'.format(ar.head, ar.body, ar.support, ar.confidence))

        db_size = len(tdb.string_dataset)
        literal_cache = {}
        literal_list = []
        return_list = []
        # rule loop
        for ar in ars:  # head=str, body=('str','str'), sup=float, conf=float
            # comment line
            return_list.append('%% rule_idx: {}, {} <= {} {} {}'.format(ar.rule_idx, ar.head, ar.body,
                                                                        ar.support, ar.confidence))
            # body loop
            body_mask = np.array([True] * db_size)
            for literal in ar.body:
                # if we only care about discretized datasets we don't need to worry about interval evaluation
                # just need to check value equality
                if literal in literal_cache:
                    literal_mask = literal_cache[literal]
                else:
                    attr, val = literal.split('=')
                    literal_mask = np.array([tdb.dataframe[attr].values.astype(str) == val])
                    literal_mask = literal_mask.reshape(db_size)
                    literal_list.append(literal)
                    literal_cache[literal] = literal_mask
                body_mask &= literal_mask
                return_list.append('literal({},{}).'.format(ar.rule_idx, literal_list.index(literal)))
            cover_idx = np.where(body_mask)[0]  # covers/2
            for data_idx in cover_idx:
                # rule covers instance
                return_list.append('covers({},{}).'.format(ar.rule_idx, data_idx))
                # prediction from this rule on the instance
                return_list.append('predicted_class({},{},{}).'.format(ar.rule_idx, data_idx, ar.head.split('=')[-1]))
                # truth label of this instance
                return_list.append('truth_class({},{},{}).'.format(ar.rule_idx, data_idx, y_train.iloc[data_idx]))
            # cba-like default class, which is the mode of uncovered instances
            uncovered_idx = np.where(~body_mask)[0]
            n_values, n_counts = np.unique(tdb.dataframe['label'].iloc[uncovered_idx], return_counts=True)
            max_idx = np.argmax(n_counts)
            default_class = n_values[max_idx]
            return_list.append('default_class({},{}).'.format(ar.rule_idx, default_class))
            # head class
            return_list.append('consequent_class({},{}).'.format(ar.rule_idx, ar.head.split('=')[-1]))
            # size of the rule (number of literals in the body)
            return_list.append('size({},{}).'.format(ar.rule_idx, len(ar.body)))
            # prep for metric evaluation
            pred = np.zeros_like(tdb.dataframe['label'].values)
            pred[cover_idx] = ar.head.split('=')[-1]
            pred[uncovered_idx] = default_class
            # accuracy of this rule
            acc = int(round(accuracy_score(y_train, pred) * 100))
            # precision of this rule
            prc = int(round(precision_score(y_train, pred, pos_label=pos_label, average=metric_averaging) * 100))
            return_list.append('accuracy({},{}).'.format(ar.rule_idx, acc))
            return_list.append('precision({},{}).'.format(ar.rule_idx, prc))
            return_list.append('support({},{}).'.format(ar.rule_idx, int(round(ar.support * 100))))
            return_list.append('confidence({},{}).'.format(ar.rule_idx, int(round(ar.confidence * 100))))
        mine_end = timer()
        print('association rule mining completed {} seconds'.format(round(mine_end - start)))

        exp_dir = './tmp/mining'

        tmp_pattern_file = os.path.join(exp_dir, '{}_pattern_out.txt'.format(experiment_tag))
        tmp_class_file = os.path.join(exp_dir, '{}_n_class.lp'.format(experiment_tag))

        with open(tmp_pattern_file, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(return_list))

        with open(tmp_class_file, 'w', encoding='utf-8') as outfile:
            outfile.write('class(0..{}).'.format(int(y_train.nunique() - 1)))

        asprin_encoding = './asp_encoding/mining_efficient.lp'

        asprin_start = timer()
        print('asprin start')
        try:
            # o = subprocess.run(['asprin', asprin_preference[asprin_pref], asprin_enc[encoding],
            #                     tmp_class_file, tmp_pattern_file, '0', '--parallel-mode=16'
            #                     ], capture_output=True, timeout=3600)
            o = subprocess.run(['asprin', asprin_encoding,
                                tmp_class_file, tmp_pattern_file, '20', '--parallel-mode=8'
                                ], capture_output=True)
            asprin_completed = True
        except subprocess.TimeoutExpired:
            o = None
            asprin_completed = False
        asprin_end = timer()
        print('asprin completed {} seconds | {} from start'.format(round(asprin_end - asprin_start),
                                                                   round(asprin_end - start)))

        if asprin_completed:
            answers, clasp_info = generate_answers(o.stdout.decode())
        else:
            answers, clasp_info = None, None
        end = timer()
        log_json = os.path.join(exp_dir, 'output.json')
        log_json_quali = os.path.join(exp_dir, 'output_quali.json')

        if asprin_completed and clasp_info is not None:
            py_rule_start = timer()
            print('py rule evaluation start')
            scores = []
            for ans_idx, ans_set in enumerate(answers):
                if not ans_set.is_optimal:
                    continue
                rule_index = []
                # selected
                selected_rules = []
                for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                    pat_idx = ans[-1][0]
                    rule_index.append(pat_idx)
                    s_rule = [ar for ar in ars if ar.rule_idx == pat_idx]
                    selected_rules += s_rule
                # break
                rule_classifier = AssociationRuleClassifier(selected_rules, mode='first')
                rule_classifier.fit(x_train, y_train)
                rule_pred = rule_classifier.predict(x_valid)
                rule_pred_metrics = {'accuracy': accuracy_score(y_valid, rule_pred),
                                     'precision': precision_score(y_valid, rule_pred, average=metric_averaging),
                                     'recall': recall_score(y_valid, rule_pred, average=metric_averaging),
                                     'f1': f1_score(y_valid, rule_pred, average=metric_averaging),
                                     'auc': roc_auc_score(y_valid, rule_pred)}
                scores.append((ans_idx, rule_pred_metrics))
            py_rule_end = timer()
            print('py rule evaluation completed {} seconds | {} from start'.format(round(py_rule_end - py_rule_start),
                                                                                   round(py_rule_end - start)))

            out_dict = {
                # experiment
                'dataset': dataset_name,
                'num_class': num_classes,
                'asprin_completed': asprin_completed,
                # clasp
                'models': clasp_info.stats['Models'],
                'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
                # 'optimal': int(clasp_info.stats['Optimal']),
                'clasp_time': clasp_info.stats['Time'],
                'clasp_cpu_time': clasp_info.stats['CPU Time'],
                # timer
                'py_total_time': end - start,
                'py_mine_time': mine_end - start,
                'py_asprin_time': asprin_end - asprin_start,
                'py_rule_time': py_rule_end - py_rule_start,
                # metrics
                'fold': f_idx,
                # 'vanilla_metrics': vanilla_metrics,
                # 'rule_metrics': rule_pred_metrics,
                'rule_metrics': scores,
            }
        else:
            out_dict = {
                # experiment
                'dataset': dataset_name,
                'num_class': num_classes,
                'asprin_completed': asprin_completed,
                # # clasp
                # 'models': int(clasp_info.stats['Models']),
                # 'optimum': True if clasp_info.stats['Optimum'] == 'yes' else False,
                # 'optimal': int(clasp_info.stats['Optimal']),
                # 'clasp_time': clasp_info.stats['Time'],
                # 'clasp_cpu_time': clasp_info.stats['CPU Time'],
                # timer
                'py_total_time': end - start,
                'py_mine_time': mine_end - start,
                'py_asprin_time': asprin_end - asprin_start,
                'py_rule_time': 0,
                # metrics
                'fold': f_idx,
                # 'vanilla_metrics': vanilla_metrics,
                # 'rule_metrics': rule_pred_metrics,
            }
        with open(log_json, 'a', encoding='utf-8') as out_log_json:
            out_log_json.write(json.dumps(out_dict) + '\n')

        out_quali = deepcopy(out_dict)
        out_quali['rules'] = []
        if asprin_completed:
            for ans_idx, ans_set in enumerate(answers):
                _tmp_rules = []
                if not ans_set.is_optimal:
                    continue
                rule_index = []
                # selected
                selected_rules = []
                for ans in ans_set.answer:  # list(tuple(str, tuple(int)))
                    pat_idx = ans[-1][0]
                    rule_index.append(pat_idx)
                    s_rule = [ar for ar in ars if ar.rule_idx == pat_idx]
                    selected_rules += s_rule
                _tmp_rules = [str(ar) for ar in selected_rules]
                out_quali['rules'].append((ans_idx, _tmp_rules))
        with open(log_json_quali, 'a', encoding='utf-8') as out_log_quali:
            out_log_quali.write(json.dumps(out_quali)+'\n')


if __name__ == '__main__':
    # datasets = ['autism',
    #             'breast', 'cars',
    #             'credit_australia', 'credit_taiwan', 'heart', 'ionosphere',
    #             'kidney', 'krvskp', 'voting', 'eeg', 'census', 'airline']
    datasets = ['synthetic_1']
    for data in datasets:
        print('='*20, data, '='*20)
        run_experiment(data)
