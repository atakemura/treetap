%here should be all individual constraints
% each constraint should say when a constraint is invalid
1 { selected(I) : pattern(I) } 1.

% % not_subset(J) = I is not a subset of J
not_subset(J) :- selected(I), item(I,Vi), not item(J,Vi), pattern(J), I != J.
% I != J is not necessary here, but I guess it should propagate better
% % not not_subset(I,J) = I is a subset of a valid itemset J and they have they same support => I is not closed
dominated :- selected(I), pattern(J), not not_subset(J), I != J.

:- dominated.

#show selected/1.