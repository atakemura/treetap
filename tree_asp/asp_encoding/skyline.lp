% class(0..2).
% you need to add class(0..K). by yourself.

% we would like to pick 1 pattern for each mode_class
1 { selected(I) :  mode_class(I, K), valid(I) } :- class(K).

% pattern is not invalid
valid(I) :- pattern(I), not invalid(I).

% skyline condition
% greater_in_size_and_geq_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
%                                             size(I,Si), size(J, Sj), Si <  Sj, X <= Y.
% geq_in_size_and_greater_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
%                                             size(I,Si), size(J, Sj), Si <= Sj, X <  Y.

greater_in_size_and_geq_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
    size(I,Si), size(J, Sj), Si <  Sj, X <= Y, same_class(I,J).
geq_in_size_and_greater_in_frequency(J) :- selected(I), support(I,X), support(J,Y),
    size(I,Si), size(J, Sj), Si <= Sj, X <  Y, same_class(I,J).

% same_class(J) :- selected(I), mode_class(I,X), mode_class(J,Y), X = Y, I != J.
same_class(I,J) :- mode_class(I,X), mode_class(J,Y), X = Y, I != J.

% dominated :- valid(J), greater_in_size_and_geq_in_frequency(J), same_class(J).
% dominated :- valid(J), geq_in_size_and_greater_in_frequency(J), same_class(J).

% put the same class in skyline condition
dominated :- valid(J), greater_in_size_and_geq_in_frequency(J).
dominated :- valid(J), geq_in_size_and_greater_in_frequency(J).

% cannot be dominated
:- dominated.

% #maximize{ S@2 : support(I,S), selected(I) }.
% #minimize{ E@1 : error_rate(I,E), selected(I) }.

#show selected/1.
