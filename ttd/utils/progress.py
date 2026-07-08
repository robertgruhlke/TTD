"""Progress-bar helper for the backward training loop."""

import contextlib

from tqdm import tqdm


@contextlib.contextmanager
def train_progress_bar(total, desc=None):
    """train()/train_adaptive_basis() unconditionally print 'n = <n>' once per backward
    timestep; show a tqdm progress bar instead by shadowing print() in that module.

    The import below is intentionally local: ttd.utils is imported partway through
    ttd.solvers.backward_iteration's own module-level import chain (via
    ttd.bases.spline -> ttd.utils.timing), so importing ttd.solvers.backward_iteration
    at module level here would create a circular import.
    """
    import ttd.solvers.backward_iteration as backward_iteration_module

    pbar = tqdm(total=total, desc=desc, leave=False)

    def _print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("n = "):
            pbar.update(1)
        else:
            tqdm.write(" ".join(str(a) for a in args))

    backward_iteration_module.print = _print
    try:
        yield
    finally:
        del backward_iteration_module.print
        pbar.close()
