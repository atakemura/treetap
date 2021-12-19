% R: rule
% C: class

class(0..1).

2 {selected(I) : predict_class(I,D,K) } :- class(K).

f1(F1) :- F1 = #sum{ 1,I : selected(I) }.

f2(F2) :- F2 = #sum { S,I : selected(I), size(I,S) }.

f3(F3) :- F3 = #sum { 1,Di : covers(I,Di), covers(J,Di),
                             selected(I), selected(J),
                             predict_class(I,D,Ki), predict_class(J,D,Ki),
                             I < J }.

f4(F4) :- F4 = #sum { 1,Di : covers(I,Di), covers(J,Di),
                             selected(I), selected(J),
                             predict_class(I,D,Ki), predict_class(J,D,Kj),
                             I < J, Ki != Kj }.


f6(F6) :- F6 = #sum { 1,Di : selected(I), covers(I,Di), predict_class(I,D,Ki),truth_class(I,D,Kj),Ki!=Kj}.

f7(F7) :- F7 = #sum { 1,I : selected(I),covers(I,Di),predict_class(I,D,Ki),truth_class(I,D,Kj),Ki=Kj}.

#show selected/1.
#show f1/1.
#show f2/1.
#show f3/1.
#show f4/1.
#show f6/1.
#show f7/1.

#preference(f1, less(weight)) { F1 :: f1(F1) }.
#preference(f2, less(weight)) { F2 :: f2(F2) }.
#preference(f3, less(weight)) { F3 :: f3(F3) }.
#preference(f4, less(weight)) { F4 :: f4(F4) }.
#preference(f6, less(weight)) { F6 :: f6(F6) }.
#preference(f7, more(weight)) { F7 :: f7(F7) }.
#preference(all, pareto) {**f1;**f2;**f3;**f4;**f6;**f7}.
#optimize(all).
