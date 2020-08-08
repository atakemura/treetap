#preference(more_accurate, less(weight)) { E :: selected(I), error_rate(I,E) }.
#preference(more_coverage, more(weight)) { C :: selected(I), support(I,C) }.
#preference(all, pareto) { **more_accurate; **more_coverage }.
#optimize(all).