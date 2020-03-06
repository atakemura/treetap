#preference(smaller_rule, less(weight)) { S :: selected(I), size(I,S) }.
#preference(more_accurate, less(weight)) { E :: selected(I), error_rate(I,E) }.
#preference(more_coverage, more(weight)) { C :: selected(I), support(I, C)}.
#preference(less_patterns, less(cardinality)) { selected(I) }.
#preference(all, lexico) { 1::**smaller_rule; 2::**less_patterns; 3::**more_accurate; 4::**more_coverage }.
#optimize(all).
