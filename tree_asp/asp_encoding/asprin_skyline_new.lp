% class(0..2).
% you need to add class(0..K). by yourself.

% we would like to pick 1 pattern for each mode_class
1 { selected(I) :  predict_class(I, K), valid(I) } :- class(K).

% pattern is not invalid
valid(I) :- rule(I), not invalid(I).

% rule length not more than 10
invalid(I) :- size(I,S), S > 10, rule(I).

% inaccurate rules
invalid(I) :- error_rate(I,E), E > 50, rule(I).

% low coverage rules
invalid(I) :- support(I,Sp), Sp < 10, rule(I).

% total rule length not more than 30
:- #sum{S:size(I,S),selected(I)} > 30.

% total support not less than 10
%:- #sum{Sp: support(I,Sp), selected(I)} < 10.

% more concise rule that has the same effectiveness
% at least 2 conditions
% J is better if...
ge_sup_leq_size_geq_acc_geq_prec(J) :- selected(I), size(I,Si), size(J,Sj),
valid(J),
support(I,Spi), support(J,Spj), accuracy(I,Ai), accuracy(J,Aj),
precision(I,Pi),precision(J,Pj),
Spi < Spj, Si >= Sj, Ai <= Aj, Pi <= Pj.

geq_sup_le_size_geq_acc_geq_prec(J) :- selected(I), size(I,Si), size(J,Sj),
valid(J),
support(I,Spi), support(J,Spj), accuracy(I,Ai), accuracy(J,Aj),
precision(I,Pi),precision(J,Pj),
Spi <= Spj, Si > Sj, Ai <= Aj, Pi <= Pj.

geq_sup_leq_size_ge_acc_geq_prec(J) :- selected(I), size(I,Si), size(J,Sj),
valid(J),
support(I,Spi), support(J,Spj), accuracy(I,Ai), accuracy(J,Aj),
precision(I,Pi),precision(J,Pj),
Spi <= Spj, Si >= Sj, Ai < Aj, Pi <= Pj.

geq_sup_leq_size_geq_acc_ge_prec(J) :- selected(I), size(I,Si), size(J,Sj),
valid(J),
support(I,Spi), support(J,Spj), accuracy(I,Ai), accuracy(J,Aj),
precision(I,Pi),precision(J,Pj),
Spi <= Spj, Si >= Sj, Ai <= Aj, Pi < Pj.

dominated :- valid(J), ge_sup_leq_size_geq_acc_geq_prec(J).
dominated :- valid(J), geq_sup_le_size_geq_acc_geq_prec(J).
dominated :- valid(J), geq_sup_leq_size_ge_acc_geq_prec(J).
dominated :- valid(J), geq_sup_leq_size_geq_acc_ge_prec(J).

% cannot be dominated
:- dominated.

#preference(more_accurate, more(weight)) { A :: selected(I), accuracy(I,A) }.
#preference(more_coverage, more(weight)) { S :: selected(I), support(I,S) }.
#preference(less_rule, less(cardinality)) {selected(I)}.

#preference(all, pareto) { **more_accurate ; **more_coverage; **less_rule}.

#optimize(all).

#show selected/1.
