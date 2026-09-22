"""Skip re-validation of values that the writer already validated.

Every span and task validates itself in ``__post_init__``. That is right for caller input. It is
wasted work when a reader rebuilds hundreds of thousands of objects from a control database whose
rows passed the same checks when they were written. The reader opens :func:`trusted_construction`
around hydration, and the validators return early while it is open.

The flag is a :class:`~contextvars.ContextVar`, so it is scoped to the current thread or task and
cannot leak into caller code that runs after the reader returns.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


_TRUSTED: ContextVar[bool] = ContextVar("timenet_trusted_construction", default=False)


def is_trusted() -> bool:
    """Return whether the current context is rebuilding values that were validated before.

    Returns:
        ``True`` inside :func:`trusted_construction`, ``False`` otherwise.
    """
    return _TRUSTED.get()


@contextmanager
def trusted_construction() -> Iterator[None]:
    """Mark the values constructed inside this block as already validated.

    Only readers rebuilding a stored artifact should open this. Values built from caller input must
    keep their checks.

    Yields:
        Nothing.
    """
    token = _TRUSTED.set(True)
    try:
        yield
    finally:
        _TRUSTED.reset(token)
