class Pattern:
    def __init__(self, idx, pattern_str, items, support, size, error_rate, mode_class):
        self.idx = idx
        self.pattern_str = pattern_str
        self.items = items
        self.support = support
        self.size = size
        self.error_rate = error_rate
        self.mode_class = mode_class

    def __str__(self):
        return 'pattern_idx={}. pattern={}. items={}. size={}. mode_class={}. error_rate={}.'.format(
            self.idx, self.pattern_str, self.items, self.size, self.mode_class, self.error_rate)

    def __repr__(self):
        return 'Pattern({}, {}, {}, {}, {}, {}, {})'.format(
            self.idx, self.pattern_str, self.items, self.support, self.size, self.error_rate, self.mode_class)

    def __eq__(self, other):
        return (self.idx == other.idx) and (self.pattern_str == other.pattern_str)


class Item:
    def __init__(self, idx, item_str):
        self.idx = idx
        self.item_str = item_str

    def __str__(self):
        return 'item_idx={}. item={}.'.format(self.idx, self.item_str)

    def __repr__(self):
        return 'Item({}, {})'.format(self.idx, self.item_str)

    def __eq__(self, other):
        return (self.idx == other.idx) and (self.item_str == other.item_str)
