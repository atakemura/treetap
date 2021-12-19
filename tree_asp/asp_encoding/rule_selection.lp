class(0..1).
% you need to add class(0..K). by yourself.

1 { selected(I) : mode_class(I,K) } :- class(K).

%overlap_count(I,J,O) :- O = #count { Vi: item(I,Vi), item(J,Vi)}, selected(I), selected(J), I<J.

%#minimize{ 1@4, Vi, I, J : item(I,Vi), item(J,Vi), selected(I), selected(J), I<J}.
%#minimize{ 1@5, I : selected(I) }.

% count number of selected atoms
%rule_count(C) :- C = #count { I : selected(I) }.

% minimize the number of rules
% minimize the overlap

% % count
%#minimize{ C@5 : rule_count(C) }.
% % overlap
%#minimize{ O : overlap_count(I,J,O) }.
% % coverage
%#maximize{ S,I : support(I,S), selected(I) }.
% % accuracy
%#minimize{ E,I : error_rate(I,E), selected(I) }.
% % size
%#minimize{ L,I : size(I,L), selected(I) }.

#show selected/1.

#preference(p1,less(cardinality)){selected(I)}.

% requires = predicate on O
%#preference(p2,less(weight)){O::overlap_count(I,J,O)}.
% does not require = predicate
#preference(p2,less(weight)){ 1, Vi, I, J :: item(I,Vi), item(J,Vi), selected(I), selected(J), I<J }.

#preference(p3,more(weight)){S::support(I,S),selected(I)}.

#preference(p4,less(weight)){E::error_rate(I,E),selected(I)}.

#preference(p5,less(weight)){L::size(I,L),selected(I)}.

% pareto fair all
%#preference(all,pareto){**p1;**p2;**p3;**p4;**p5}.

% prefer short rule
#preference(all,lexico){5::**p1;3::**p2;2::**p3;4::**p4;1::**p5}.

% partial preference
%#preference(a1,pareto){**p3;**p4;**p5}.
%#preference(a2,pareto){**p1;**p2}.
%#preference(all,lexico){2::**a2;1::**a1}.

#optimize(all).