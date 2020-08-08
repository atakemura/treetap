% class(0..1).
% you need to add class(0..K). by yourself.

1 { selected(I) : mode_class(I, K) } 5 :- class(K).

% same_class(I,J) :- mode_class(I,X), mode_class(J,Y), X = Y, I < J.

% overlap(I,J,O) :- O = #count{ 1 : selected(I), selected(J), item(I, Vi), item(J, Vi), I != J, same_class(I, J) }.

overlap(I,J) :- selected(I), selected(J), item(I,Vi), item(J,Vi), I < J.%, same_class(I,J).
overlap_count(I,J,O) :- O = #count { Vi: item(I,Vi), item(J,Vi)}, overlap(I,J).

% count number of selected atoms
rule_count(C) :- C = #count { I : selected(I) }.

:- selected(I), error_rate(I,E), E > 5.

% % count
% % #minimize{ C@5 : C = #count{I, selected(I)} }.
% % overlap
% #minimize{ O@4 : overlap_count(I,J,O), selected(I), selected(J) }.
% % coverage
% #maximize{ S@3 : support(I,S), selected(I) }.
% % accuracy
% #minimize{ E@2 : error_rate(I,E), selected(I) }.
% % size
% #minimize{ L@1 : size(I,L), selected(I) }.

#show selected/1.