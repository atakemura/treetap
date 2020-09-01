from answers import AnswerSet, ClaspInfo


def generate_answers(output: str):
    answer = None  # is used to generate a model only when we are sur there is (no) optimization
    answer_number = None
    optimization = None
    optimum_found = False
    stats = None
    info = None
    answer_sets = []
    clasp_info = None

    for ptype, pload in parse_clasp_output(output, yield_info=True, yield_opti=True, yield_stats=True):
        if ptype == 'answer_number':
            if answer and answer_number:
                answer_sets.append(AnswerSet(answer_id=answer_number, answer_string=answer,
                                             optimization=optimization, is_optimal=optimum_found))
            # keep the answer number and reset the rest
            answer_number, answer, optimization, optimum_found = pload, None, None, False
        elif ptype == 'answer':  # yield previously found answer
            answer = pload
        elif ptype == 'optimum found':
            optimum_found = pload
        elif ptype == 'optimization':
            optimization = pload
        elif ptype == 'statistics':
            stats = pload
        elif ptype == 'info':
            info = pload  # info comes at the end
            clasp_info = ClaspInfo(stats_string=stats, info_string=info)
        else:
            raise ValueError('Unexpected payload type: {} in {}'.format(ptype, (ptype, pload)))
    if answer is not None and answer_number is not None:  # most likely at the end
        answer_sets.append(AnswerSet(answer_id=answer_number, answer_string=answer,
                                     optimization=optimization, is_optimal=optimum_found))

    return answer_sets, clasp_info


def parse_clasp_output(output: iter or str,
                       yield_stats: bool = False,
                       yield_opti: bool = False,
                       yield_info: bool = False,
                       yield_prgs: bool = False):
    """
    Yield pairs (payload type, payload) where type is 'info', 'statistics', 'optimization', 'progression', or 'answer'
    and payload the raw information.

    If you set yield_{stats,opti,info,prgs}, you will get corresponding information in tuples.
    In any case, tuple ('answer', termset) will be returned with termset a string containing the raw data.

    Based on clyngor library(https://github.com/Aluriak/clyngor), more specifically from this file:
    https://github.com/Aluriak/clyngor/blob/fe9b5c1050eb97d61a9c84bfb801de8365ef5672/clyngor/parsing.py

    Args:
        output: iterable of lines or full clasp output to parse
        yield_stats: yields final statistics as a mapping {field: value} under type 'statistics'
        yield_opti: yields line sometimes following an answer set, beginning with 'Optimization: ' or 'OPTIMUM FOUND'.
        yield_info: yields all lines not included in other types, including the first lines not related to first answer
                    under type 'info' as a tuple of lines
        yield_prgs: yields lines of Progressions that are sometimes following an answer set in multithreading contexts,
                    under type 'progression' as a string.

    Returns:
        (payload_type: str, payload: str)

    """
    # These may change depending on the clingo/clasp version so watch out for any changes.
    # TextOutput https://github.com/potassco/clasp/blob/master/src/clasp_output.cpp#L700
    # printUnsat https://github.com/potassco/clasp/blob/master/src/clasp_output.cpp#L980
    FLAG_ANSWER = 'Answer: '
    FLAG_OPT = 'Optimization: '
    FLAG_OPT_FOUND = 'OPTIMUM FOUND'
    FLAG_OPT_FOUND_ASPRIN = 'OPTIMUM FOUND *'
    FLAG_PROGRESS = 'Progression :'

    output = iter(output.splitlines() if isinstance(output, str) else output)

    # get the first lines
    line = next(output)
    infos = []
    while not line.startswith(FLAG_ANSWER):
        infos.append(line)
        try:
            line = next(output)
        except StopIteration:
            return

    # first answer begins
    while True:
        if line.startswith(FLAG_ANSWER):
            yield 'answer_number', int(line[len(FLAG_ANSWER):])
            yield 'answer', next(output)
        elif line.startswith(FLAG_OPT) and yield_opti:
            yield 'optimization', tuple(map(int, line[len(FLAG_OPT):].strip().split()))
        elif (line.startswith(FLAG_OPT_FOUND) or line.startswith(FLAG_OPT_FOUND_ASPRIN)) and yield_opti:
            yield 'optimum found', True
        elif line.startswith(FLAG_PROGRESS) and yield_prgs:
            yield 'progression', line[len(FLAG_PROGRESS):].strip()
        elif not line.strip():  # empty line: statistics are beginning
            if not yield_stats:
                break  # stats are the last part of the output
            stats = {}
            for line in output:
                sep = line.find(':')
                key, value = line[:sep], line[sep+1:]
                stats[key.strip()] = value.strip()
            yield 'statistics', stats
            break
        else:  # should not happen
            infos.append(line)
        try:
            line = next(output)
        except StopIteration:
            break

    if yield_info:
        yield 'info', tuple(infos)

