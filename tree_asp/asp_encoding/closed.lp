% class(0..2).
% you need to add class(0..K). by yourself.

% we would like to pick 1 pattern for each mode_class
1 { selected(I) :  mode_class(I, K), valid(I) } :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

% same_class(J) :- selected(I), mode_class(I,X), mode_class(J,Y), X = Y, I != J.
same_class(I,J) :- mode_class(I,X), mode_class(J,Y), X = Y, I != J.

% % not_subset(J) = I is not a subset of J
% not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J).
% dominated :- selected(I), pattern(J), support(I,X), support(J,X), not not_subset(J), I != J, same_class(J).

% % not_subset(J) = I is not a subset of J
not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J), not same_class(I,J).
dominated :- selected(I), pattern(J), support(I,X), support(J,X), not not_subset(J), I != J.

% cannot be dominated
:- dominated.

% #maximize{ S@2 : support(I,S), selected(I) }.
% #minimize{ E@1 : error_rate(I,E), selected(I) }.

#show selected/1.
