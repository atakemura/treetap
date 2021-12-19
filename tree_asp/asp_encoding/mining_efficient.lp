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

% % J is preferred
% % confidence(I) < confidence(J)
% cba_preferred_gr_conf(J) :- selected(I), confidence(I,Ci), confidence(J,Cj), same_class(I,J),
%                     I!=J, Ci < Cj.
% % confidence(I) = confidence(J) and support(I) < support(J)
% cba_preferred_eq_conf_gr_sup(J) :- selected(I), confidence(I,Ci), confidence(J,Cj),
%                     support(I,Si), support(J,Sj), same_class(I,J),
%                     I!=J, Ci = Cj, Si < Sj.
% % % confidence, support =, index(I) > index(J)
% % cba_preferred_eq_conf_sup_gr_idx(J) :- selected(I), confidence(I,Ci), confidence(J,Cj), 
% %                     support(I,Si), support(J,Sj), same_class(I,J),
% %                     I!=J, Ci = Cj, Si = Sj, I > J.

% cba_dominated :- cba_preferred_gr_conf(J).
% cba_dominated :- cba_preferred_eq_conf_gr_sup(J).
% % cba_dominated :- cba_preferred_eq_conf_sup_gr_idx(J).

% :- cba_dominated.

% cannot have the same body and different consequent
same_body(J) :- selected(J),
                C=#count{Li:literal(I,Li),literal(J,Li)},
                size(I,C),size(J,C),I!=J.
same_body_diff_conseq(J) :- selected(I), same_body(J), not same_class(I,J), I!=J.
{same_body_diff_conseq(I):selected(I)}1:- same_body_diff_conseq(I).

% {same_body_diff_conseq(I):selected(I)} 1.
% invalid :- selected(I), selected(J), same_body_diff_conseq(J), I!=J.

% no overlap
% shared_item(J) :-   selected(J),
%                     literal(I,Li), literal(J,Li), 
%                     same_class(I,J), I!=J.
% invalid(J) :- shared_item(J)

% #const size_limit = 10.
% #const rule_count_limit = 5.
% invalid(I) :- #sum{ S,J:size(J,S),consequent_class(J,K),class(K),I=J} > size_limit, 
%                 selected(I).

% :- invalid(I).

% class(0..1).

% generate step n_class * 2^n solutions
2 {selected(I) : consequent_class(I,K) } 5 :- class(K).

% cardinality
#preference(f1, less(cardinality)){ selected(I) }.

% overall size
#preference(f2, less(weight)){ S,I :: size(I,S), selected(I) }.

% intraclass overlap
#preference(f3, less(cardinality)) { covers(I,D), covers(J,D),
                            selected(I), selected(J),
                            predicted_class(I,D,Ki), predicted_class(J,D,Ki),
                            I < J}.

% interclass overlap
#preference(f4, less(cardinality)) { covers(I,D), covers(J,D),
                            selected(I), selected(J),
                            predicted_class(I,D,Ki), predicted_class(J,D,Kj),
                            I < J, Ki != Kj }.

% % incorrect cover
#preference(f5, less(cardinality)) { selected(I), covers(I,D), 
                                    predicted_class(I,D,Ki),
                                    truth_class(I,D,Kj),
                                    Ki!=Kj}.

% % correct cover
#preference(f6, more(cardinality)) { selected(I),covers(I,D),
                                    predicted_class(I,D,Ki),truth_class(I,D,Ki)}.

% precision
#preference(f7, more(weight)) { P,I :: selected(I), precision(I,P) }.

% accuracy
#preference(f8, more(weight)) { A,I :: selected(I), accuracy(I,A)}.

% #preference(all, pareto) {**f1;**f2;**f3;**f4;**f6;**f7}.

% #preference(f9, more(weight)){ A,P :: selected(I), accuracy(I,A), precision(I,P) }.

% #preference(all, pareto) {**f1}.
% #preference(all, pareto) {**f1;**f2}.
% #preference(all, pareto) {**f1;**f2;**f7;**f8}.
#preference(all, pareto) {**f1;**f2;**f3;**f4;**f5;**f6;**f7;**f8}.
#optimize(all).

#show selected/1.