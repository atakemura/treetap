% class(0..2).
% you need to add class(0..K). by yourself.

% we would like to pick 1 pattern for each mode_class
1 { selected(I) :  mode_class(I, K), valid(I) } :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

% % not_subset(J) = I is not a subset of J
% I != J is not necessary here, but I guess it should propagate better
% not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J), I != J.
not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J), I != J,
    not same_class(I, J).

% % not not_subset(I,J) = I is a subset of a valid itemset J
% and they have the same support => I is not closed
dominated :- selected(I), pattern(J), not not_subset(J), I != J.

same_class(I,J) :- mode_class(I,X), mode_class(J,Y), X = Y, I != J.

% cannot be dominated
:- dominated.

#show selected/1.
