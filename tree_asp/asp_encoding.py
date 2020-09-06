import clingo


class BaseSolver:
    def __init__(self):
        self.models = []
        self.encoding = ''

    def solve(self, instance=None):
        ctl = clingo.Control()
        ctl.add('base', [], instance)
        ctl.add('solve', [], self.encoding)
        ctl.ground([('base', []), ('solve', [])])

        # solver configuration
        ctl.configuration.solve.models = 0

        ctl.solve(on_model=self.on_model)

    def on_model(self, m):
        self.models.append(m.symbols(shown=True))


class SkylineSolver(BaseSolver):
    def __init__(self, n_class=2):
        super(SkylineSolver, self).__init__()
        self.n_class = n_class - 1
        self.encoding = 'class(0..{}).'.format(self.n_class) + """
% we would like to pick 1 pattern for each predict_class
1 { selected(I) :  predict_class(I, K), valid(I) } 1 :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

% skyline condition
greater_in_size_and_geq_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
                                            size(I,Si), size(J, Sj), Si <  Sj, X <= Y.
geq_in_size_and_greater_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
                                            size(I,Si), size(J, Sj), Si <= Sj, X <  Y.

same_class(J) :- selected(I), predict_class(I,X), predict_class(J,Y), X = Y, I != J.

dominated :- valid(J), greater_in_size_and_geq_in_frequency(J), same_class(J).
dominated :- valid(J), geq_in_size_and_greater_in_frequency(J), same_class(J).

% cannot be dominated
:- dominated.

%#maximize{ S@2 : support(I,S), selected(I) }.
%#minimize{ E@1 : error_rate(I,E), selected(I) }.

#show selected/1.
        """


class MaximalSolver(BaseSolver):
    def __init__(self, n_class=2):
        super(MaximalSolver, self).__init__()
        self.n_class = n_class - 1
        self.encoding = 'class(0..{}).\n'.format(self.n_class) + """
% we would like to pick 1 pattern for each predict_class
1 { selected(I) :  predict_class(I, K), valid(I) } 1 :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

% % not_subset(J) = I is not a subset of J
not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J), I != J.
% I != J is not necessary here, but I guess it should propagate better
% % not not_subset(I,J) = I is a subset of a valid itemset J and they have they same support => I is not closed
dominated :- selected(I), pattern(J), not not_subset(J), I != J.

% cannot be dominated
:- dominated.

#maximize{ S@2 : support(I,S), selected(I) }.
#minimize{ E@1 : error_rate(I,E), selected(I) }.

#show selected/1.
        """


class ClosedSolver(BaseSolver):
    def __init__(self, n_class=2):
        super(ClosedSolver, self).__init__()
        self.n_class = n_class - 1
        self.encoding = 'class(0..{}).\n'.format(self.n_class) + """
% we would like to pick 1 pattern for each predict_class
1 { selected(I) :  predict_class(I, K), valid(I) } 1 :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

same_class(J) :- selected(I), predict_class(I,X), predict_class(J,Y), X = Y, I != J.

% % not_subset(J) = I is not a subset of J
not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J).
dominated :- selected(I), pattern(J), support(I,X), support(J,X), not not_subset(J), I != J, same_class(J).

% cannot be dominated
:- dominated.

#maximize{ S@2 : support(I,S), selected(I) }.
#minimize{ E@1 : error_rate(I,E), selected(I) }.

#show selected/1."""
