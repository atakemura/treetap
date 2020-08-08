from contextlib import contextmanager
from timeit import default_timer as timer

@contextmanager
def timer_exec(name):
    start = timer()
    yield
    print('{} time elapsed {} seconds'.format(name, round(timer() - start)))
