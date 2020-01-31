#preference(smaller_rule, less(weight)) { S :: selected(I), size(I,S) }.
#preference(more_accurate, less(weight)) { E :: selected(I), error_rate(I,E) }.
#preference(all, pareto) { **smaller_rule; **more_accurate }.
#optimize(all).
