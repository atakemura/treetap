import re


class AnswerSet:
    def __init__(self, answer_id: int, answer_string: str, optimization=None, is_optimal: bool = False):
        self.answer_id = answer_id
        self.answer = list(self._parse_answer_string(answer_string))
        self.optimization = optimization
        self.is_optimal = is_optimal

    def _parse_answer_string(self, answer_string: str, discard_quotes: bool = False,
                             parse_int: bool = True, parse_args: bool = True):
        # adapted from clyngor/clyngor/answers.py :: naive_parsing_of_answer_set
        answer_pattern = re.compile(r'([a-z_][a-zA-Z0-9_]*|[0-9]+|"[^"]*")(\([^)]+\))?')

        for match in answer_pattern.finditer(answer_string):
            pred, args = match.groups()
            assert args is None or (args.startswith('(') and args.endswith(')'))
            if args and parse_args:
                args = args[1:-1]  # remove surrounding parens
                if discard_quotes:
                    raise NotImplementedError
                    # args = utils.remove_arguments_quotes(args)
                args = tuple(
                    (int(arg) if parse_int and
                                 (arg[1:] if arg.startswith('-') else arg).isnumeric() else arg)
                    for arg in args.split(',')
                ) if parse_args else args
            elif args:  # args should not be parsed
                args = args
            pred = int(pred) if pred.isnumeric() else pred  # handle
            yield pred, args or ()


class ClaspInfo:
    def __init__(self, stats_string: dict = None, info_string: tuple = None):
        self.stats = self._parse_stats_string(stats_string)
        self.info = self._parse_info_string(info_string)

    def _parse_stats_string(self, stats_string):
        # Usually like {'Models': '3', 'Optimum': 'yes', ...}
        # Keys: Models, Optimum, Optimal, Time, CPU Time, Calls
        return stats_string

    def _parse_info_string(self, info_string):
        # Usually like ('asprin version 3.1.0', 'Reading from xyz ...', 'Solving...')
        return info_string

    def __str__(self):
        info = '\n'.join(self.info) if self.info else ''
        stats = '\n'.join('{}: {}'.format(k, v) for k, v in self.stats.items()) if self.stats else ''
        return '\n'.join(['Clasp INFO '.ljust(80, '='), info, 'Clasp STATS '.ljust(80, '='), stats])
