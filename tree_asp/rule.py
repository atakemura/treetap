class Rule:
    def __init__(self, idx, rule_str, conditions, support, size, accuracy, error_rate, precision, recall, predict_class):
        self.idx = idx
        self.rule_str = rule_str
        self.items = conditions
        self.support = support
        self.size = size
        self.accuracy = accuracy
        self.error_rate = error_rate
        self.precision = precision
        self.recall = recall
        self.predict_class = predict_class

    def __str__(self):
        return 'rule_idx={}. rule={}. items={}. size={}. predict_class={}.' \
               'accuracy={}. error_rate={}. precision={}. recall={}.'.format(
            self.idx, self.rule_str, self.items, self.size, self.predict_class,
            self.accuracy, self.error_rate, self.precision, self.recall)

    def __repr__(self):
        return 'Rule({}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
            self.idx, self.rule_str, self.items, self.support, self.size, self.accuracy,
            self.error_rate, self.precision, self.recall, self.predict_class)

    def __eq__(self, other):
        return (self.idx == other.idx) and (self.rule_str == other.rule_str)


class Condition:
    def __init__(self, idx, condition_str):
        self.idx = idx
        self.condition_str = condition_str

    def __str__(self):
        return 'condition_idx={}. condition={}.'.format(self.idx, self.condition_str)

    def __repr__(self):
        return 'Condition({}, {})'.format(self.idx, self.condition_str)

    def __eq__(self, other):
        return (self.idx == other.idx) and (self.condition_str == other.condition_str)
