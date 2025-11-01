import functools
import os

def profile(func):
    import cProfile
    profiler = cProfile.Profile()
    @functools.wraps(func)
    def profiled(*args, **kwargs):
        try:
            profiler.enable()
            retval = func(*args, *kwargs)
            profiler.disable()
        except Exception as e:
            profiler.disable()
            raise e
        finally:
            profiler.dump_stats(file=f"prof/{func.__name__}.prof")

        return retval
    return profiled

