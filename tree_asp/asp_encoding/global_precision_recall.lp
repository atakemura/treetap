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
% class(0..2).
% You need to add class(0..K). by yourself if you're running this script
% manually. If this is ran automatically there's a separate file with just
% class(1). in it.

% we would like to pick at least 1 pattern for each predict_class
1 { selected(I) :  predict_class(I, K), valid(I) } 5 :- class(K).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% User Defined Local Constraints and Selection Criteria
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rule is not invalid
valid(I) :- rule(I), not invalid(I).

% rule length not more than 10
invalid(I) :- size(I,S), S > 10, rule(I).

% inaccurate rules
invalid(I) :- error_rate(I,E), E > 70, rule(I).

% low precision rules
invalid(I) :- precision(I,P), P < 2, rule(I).

% low recall rules
invalid(I) :- recall(I,R), R < 2, rule(I).

% low coverage rules
invalid(I) :- support(I,Sp), Sp < 2, rule(I).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Dominance Relation Definitions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% I is dominated by J
gt_prc_geq_rec(J) :- selected(I), valid(J),
    precision(I,Pi), precision(J,Pj), recall(I,Ri), recall(J,Rj),
    Pi < Pj, Ri <= Rj.

geq_prc_gt_rec(J) :- selected(I), valid(J),
    precision(I,Pi), precision(J,Pj), recall(I,Ri), recall(J,Rj),
    Pi <= Pj, Ri < Rj.

dominated :- valid(J), gt_prc_geq_rec(J).
dominated :- valid(J), geq_prc_gt_rec(J).

% cannot be dominated
:- dominated.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Optimization Over Answer Sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of shared conditions between rules
rule_overlap(I,J,Cn) :- selected(I), selected(J), I!=J,
    Cn = #count{Ci : Ci=Cj, condition(I,Ci), condition(J,Cj)}.

selected_rules(SR) :- SR = #count { I : selected(I) }, SR != 0.

#maximize { Pi/(S*SR)@3,I : selected(I), size(I,S), precision(I,Pi), selected_rules(SR) }.
#maximize { R/S@2,I : selected(I), size(I,S), recall(I,R) }.
#minimize { Cn : selected(I), selected(J), rule_overlap(I,J,Cn) }.

#show selected/1.
