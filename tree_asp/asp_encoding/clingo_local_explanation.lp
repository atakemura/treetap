%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% List of Atoms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% rule(I)            % I is index, I-th rule
%% condition(I,C)     % C is index, C-th condition
%% size(I,S)          % S is the number of conditions (items) in the rule
%% accuracy(I,A)      % A is the accuracy of this rule
%% error_rate(I,E)    % E is the error rate (1-accuracy) of this rule
%% precision(I,P)     % P is the precision of this rule
%% recall(I,R)        % R is the recall of this rule
%% f1_score(I,F)      % F is the F1 score of this rule
%% predict_class(I,X) % X is the predicted class (head) of this rule
%% class(K)           % K is the target class
%% selected(I)        % I is the selected rule
%% valid(I)           % I is a valid rule (not invalid)
%% invalid(I)         % True when I is invalid
%% rule_overlap(I,J,Cn) % Cn is the number of conditions shared between rules I and J

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% we would like to pick minimum 1 and maximum 3 for each
1 { selected(I) :  valid(I) } 3.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% User Defined Local Constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rule is not invalid
valid(I) :- rule(I), not invalid(I).

% rule length not more than 10
%invalid(I) :- size(I,S), S > 10, rule(I).

% inaccurate rules
%invalid(I) :- error_rate(I,E), E > 50, rule(I).

% low precision rules
%invalid(I) :- precision(I,P), P < 10, rule(I).

% low recall rules
%invalid(I) :- recall(I,R), R < 10, rule(I).

% low coverage rules
%invalid(I) :- support(I,Sp), Sp < 10, rule(I).

% total rule length not more than 30
% :- #sum{S,I : size(I,S), selected(I)} > 30.

% total number of rules selected should not be more than 10
% :- #count{I : selected(I)} > 10.

% total support not less than 10
%:- #sum{Sp: support(I,Sp), selected(I)} < 10.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dominance Relation Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% more concise rule that has the same or better effectiveness
% and share at least one condition

% number of shared conditions between rules
rule_overlap(I,J,Cn) :- selected(I), selected(J), I!=J,
    Cn = #count{Ci : Ci=Cj, condition(I,Ci), condition(J,Cj)}.

% I is dominated by J
ge_f1_geq_sup(J) :- selected(I), valid(J),
    f1_score(I,Fi), f1_score(J,Fj), support(I,Si), support(J,Sj),
    Fi < Fj, Si = Sj.

geq_f1_ge_sup(J) :- selected(I), valid(J),
    f1_score(I,Fi), f1_score(J,Fj), support(I,Si), support(J,Sj),
    Fi <= Fj, Si < Sj.

dominated :- valid(J), ge_f1_geq_sup(J).
dominated :- valid(J), geq_f1_ge_sup(J).

% cannot be dominated
:- dominated.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Optimization Over Answer Sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

selected_rules(SR) :- SR = #count { I : selected(I) }, SR != 0.

%#maximize { F/SR,I  : selected(I), f1_score(I,F), selected_rules(SR) }.
#maximize { P/SR,I  : selected(I), precision(I,P), selected_rules(SR) }.
#minimize { Cn,I : selected(I), selected(J), rule_overlap(I,J,Cn) }.

#show selected/1.