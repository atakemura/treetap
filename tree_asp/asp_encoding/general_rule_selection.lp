% class(0..1).
% you need to add class(0..K). by yourself.

1 { selected(I) : mode_class(I,K) } :- class(K).

overlap_count(I,J,O) :- O = #count { Vi: item(I,Vi), item(J,Vi)}, selected(I), selected(J), I<J.

#minimize{ 1@4, Vi, I, J : item(I,Vi), item(J,Vi), selected(I), selected(J), I<J}.
#minimize{ 1@5, I : selected(I) }.

% count number of selected atoms
%rule_count(C) :- C = #count { I : selected(I) }.

% minimize the number of rules
% minimize the overlap

% % count
%#minimize{ C@5 : rule_count(C) }.
% % overlap
#minimize{ O@4 : overlap_count(I,J,O) }.
% % coverage
#maximize{ S@3 : support(I,S), selected(I) }.
% % accuracy
#minimize{ E@2 : error_rate(I,E), selected(I) }.
% % size
#minimize{ L@1 : size(I,L), selected(I) }.

#show selected/1.