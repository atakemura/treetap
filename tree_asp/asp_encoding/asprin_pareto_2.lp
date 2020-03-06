#preference(smaller_rule, less(weight)) { S :: selected(I), size(I,S) }.
#preference(more_accurate, less(weight)) { E :: selected(I), error_rate(I,E) }.
#preference(more_coverage, more(weight)) { C :: selected(I), support(I, C)}.
#preference(less_patterns, less(cardinality)) { selected(I) }.
#preference(all, pareto) { **smaller_rule; **less_patterns; **more_accurate; **more_coverage }.
#optimize(all).
