1 { selected(I) : valid(I) } 1.
% each constraint should say when a constraint is invalid
% here should be all individual constraints

valid(I) :- pattern(I), not invalid(I).

greater_in_size_and_geq_in_frequency(J) :- selected(I), support(I,X), support(J,Y), size(I,Si), size(J, Sj), Si <  Sj, X <= Y.
geq_in_size_and_greater_in_frequency(J) :- selected(I), support(I,X), support(J,Y), size(I,Si), size(J, Sj), Si <= Sj, X <  Y.

dominated :- valid(J), greater_in_size_and_geq_in_frequency(J).
dominated :- valid(J), geq_in_size_and_greater_in_frequency(J).

:- dominated.

#show selected/1.