class(0..1).

1 {selected(I) : consequent_class(I,K) } :- class(K).


% class(K) /1
% predicted_class(I,D,K) /3
% literal(I,L) /2
% consequent_class(I,K) /2
% truth_class(D,D,K) /3
% covers(I,D) /2
% accuracy(I,A) /2
% precision(I,P) /2

same_class(I,J) :- selected(I),
                   consequent_class(I,Ki), consequent_class(J,Ki),
                   I != J.

same_body(J) :- selected(I),selected(J),
                C=#count{Li:literal(I,Li),literal(J,Li)},
                size(I,C),size(J,C),I!=J.
same_body_diff_conseq(J) :- selected(I), same_body(J), not same_class(I,J), I!=J.

invalid :- selected(I), selected(J), same_body_diff_conseq(J), I!=J.
:- invalid.

rule_count(C) :- C=#count{I:selected(I)}.

% #minimize { 1,I :   selected(I) }.
#minimize { C : rule_count(C)}.

#minimize { S,I :   selected(I), size(I,S) }.

#maximize { A,P : selected(I), accuracy(I,A), precision(I,P)}.

% #minimize { 1,D :   covers(I,D), covers(J,D),
%                     selected(I), selected(J),
%                     predicted_class(I,D,Ki), predicted_class(J,D,Ki),
%                     I < J }.

% #minimize { 1,D :   covers(I,D), covers(J,D),
%                     selected(I), selected(J),
%                     predicted_class(I,D,Ki), predicted_class(J,D,Kj),
%                     I < J, Ki != Kj }.

% #minimize { 1,D :   selected(I),
%                     covers(I,D), 
%                     predicted_class(I,D,Ki),
%                     truth_class(I,D,Kj),
%                     Ki!=Kj}.

% #maximize { 1,I :   selected(I),
%                     covers(I,D),
%                     predicted_class(I,D,Ki),
%                     truth_class(I,D,Kj),
%                     Ki=Kj}.


%this does not work
% func(N) :-
%     F1 = #count{I:selected(I)},
%     F2 = #sum{I,S:selected(I),size(I,S)},
%     F3 = #sum{1,D:covers(I,D),covers(J,D),selected(I),selected(J),
%                     predicted_class(I,D,Ki),predicted_class(J,D,Ki),
%                     I < J},
%     F4 = #sum{1,D:covers(I,D),covers(J,D),selected(I),selected(J),
%                     predicted_class(I,D,Ki),predicted_class(J,D,Kj),
%                     I < J, Ki != Kj},
%     F5 = #sum{ 1,D : selected(I), covers(I,D), predicted_class(I,D,Ki),
%                     truth_class(I,D,Kj),Ki!=Kj},
%     F6 = #sum{ 1,I : selected(I),covers(I,D),predicted_class(I,D,Ki),
%                     truth_class(I,D,Kj),Ki=Kj},
%     N = F1 + F2 + F3 + F4 + F5 - F6.

% #minimize{N:func(N)}.

#show selected/1.