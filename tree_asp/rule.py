class Rule:
    def __init__(self, idx, rule_str, literals, support, size, accuracy, error_rate, precision, recall, predict_class):
        self.idx = idx
        self.rule_str = rule_str
        self.items = literals
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


class Literal:
    def __init__(self, idx, literal_str):
        self.idx = idx
        self.literal_str = literal_str

    def __str__(self):
        return 'literal_idx={}. literal={}.'.format(self.idx, self.literal_str)

    def __repr__(self):
        return 'Literal({}, {})'.format(self.idx, self.literal_str)

    def __eq__(self, other):
        return (self.idx == other.idx) and (self.literal_str == other.literal_str)
