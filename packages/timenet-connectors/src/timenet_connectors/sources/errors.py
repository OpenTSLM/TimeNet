"""Attach source locations to connector errors raised during decoding."""

from collections.abc import Generator
from contextlib import contextmanager

from timenet.errors import TimeNetConnectorError


@contextmanager
def error_context(where: str) -> Generator[None]:
    """Add a source location to the connector errors raised inside the block.

    The error keeps its type and its traceback. Nested contexts give messages such as
    ``file: row: message``.

    Args:
        where: The place in the source, such as ``"train.jsonl line 4"`` or ``"exams.csv row 17"``.

    Yields:
        Nothing. The block runs inside the context.

    Raises:
        TimeNetConnectorError: The error from the block, with ``where`` before its message.
    """
    try:
        yield
    except TimeNetConnectorError as error:
        message = str(error.args[0]) if error.args else str(error)
        error.args = (f"{where}: {message}", *error.args[1:])
        raise
