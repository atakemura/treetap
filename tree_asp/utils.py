from contextlib import contextmanager
from datetime import datetime
from timeit import default_timer as timer


@contextmanager
def timer_exec(name):
    start = timer()
    yield
    print('{} time elapsed {} seconds'.format(name, round(timer() - start)), flush=True)


def time_print(msg):
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '\t', msg, flush=True)
