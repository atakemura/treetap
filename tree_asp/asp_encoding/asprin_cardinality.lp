#preference(p1, less(weight)) { O :: overlap_count(I,J,O), selected(I), selected(J) }.
#preference(p2, less(weight)) { L :: size(I,L), selected(I) }.
#preference(p3, less(weight)) { E :: error_rate(I,E), selected(I) }.
#preference(p4, more(weight)) { S :: support(I,S), selected(I) }.
#preference(p5, less(cardinality)) { selected(I) }.
#preference(all, lexico) { 4::**p1; 1::**p2; 2::**p3; 3::**p4; 5::**p5}.
#optimize(all).

% Models       : 13
%   Optimum    : yes
%   Optimal    : 1
% Calls        : 16
% Time         : 10449.540s (Solving: 10439.65s 1st Model: 1399.82s Unsat: 9039.26s)
% CPU Time     : 166834.825s
% Threads      : 16