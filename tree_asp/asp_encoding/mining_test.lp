% R: rule
% C: class

class(0..1).

2 {selected(I) : predict_class(I,K) } :- class(K).

predict_class(1,0).
predict_class(2,0).
predict_class(3,1).
predict_class(4,1).

truth_class(1,0).
truth_class(2,0).
truth_class(3,1).
truth_class(4,0).

size(1,1).
size(2,1).
size(3,2).
size(4,2).

% example of 6 data points
covers(1,1).
covers(1,2).
covers(1,3).
covers(1,4).
covers(2,1).
covers(2,2).

covers(3,4).
covers(3,5).
covers(3,6).
covers(4,4).
covers(4,5).
covers(4,6).

% this should be 4
%f1(F1) :- F1 = #sum{ 1,I : selected(I) }.

% this should be 6
%f2(F2) :- F2 = #sum { S,I : selected(I), size(I,S) }.

% this should be 5, 2 in class(0) and 3 in class(1)
%f3(F3) :- F3 = #sum { 1,Di : covers(I,Di), covers(J,Di),
%                             selected(I), selected(J),
%                             predict_class(I,Ki), predict_class(J,Ki),
%                             I < J }.

% this should be 1
%f4(F4) :- F4 = #sum { 1,Di : covers(I,Di), covers(J,Di),
%                             selected(I), selected(J),
%                             predict_class(I,Ki), predict_class(J,Kj),
%                             I < J, Ki != Kj }.

% f5 is absent, as the aggregate lower bound condition is a constraint

% f6 incorrect coverage, should be 0
%f6(F6) :- F6 = #sum { 1,Di : selected(I), covers(I,Di), predict_class(I,Ki),truth_class(I,Kj),Ki!=Kj}.

% f7 correct coverage, should be 3
%f7(F7) :- F7 = #sum { 1,I : selected(I),covers(I,Di),predict_class(I,Ki),truth_class(I,Kj),Ki=Kj}.

#show selected/1.
%#show f1/1.
%#show f2/1.
%#show f3/1.
%#show f4/1.
%#show f6/1.
%#show f7/1.

#preference(f1, less(weight)){ 1,I :: selected(I) }.
#preference(f2, less(weight)){ S,I :: selected(I), size(I,S) }.
#preference(f3, less(weight)) { 1,Di :: covers(I,Di), covers(J,Di),
                             selected(I), selected(J),
                             predict_class(I,Ki), predict_class(J,Ki),
                             I < J }.
#preference(f4, less(weight)) { 1,Di :: covers(I,Di), covers(J,Di),
                             selected(I), selected(J),
                             predict_class(I,Ki), predict_class(J,Kj),
                             I < J, Ki != Kj }.

#preference(f6, less(weight)) { 1,Di :: selected(I), covers(I,Di), predict_class(I,Ki),truth_class(I,Kj),Ki!=Kj}.

#preference(f7, more(weight)) { 1,I :: selected(I),covers(I,Di),predict_class(I,Ki),truth_class(I,Kj),Ki=Kj}.


%#preference(f1, less(weight)) { F1 :: f1(F1) }.
%#preference(f2, less(weight)) { F2 :: f2(F2) }.
%#preference(f3, less(weight)) { F3 :: f3(F3) }.
%#preference(f4, less(weight)) { F4 :: f4(F4) }.
%#preference(f6, less(weight)) { F6 :: f6(F6) }.
%#preference(f7, more(weight)) { F7 :: f7(F7) }.
#preference(all, pareto) {**f1;**f2;**f3;**f4;**f6;**f7}.
#optimize(all).
