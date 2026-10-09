"""Explicit execution and lifetime ownership for inspection readers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from typing import Any, TypeVar, cast

from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.viewer.inspection import ViewerInspection


_Result = TypeVar("_Result")


def clone_reader(reader: TimeFReader) -> TimeFReader:
    """Return an independent reader over the same pinned storage handles."""
    return ViewerInspection(reader).independent_reader()


class ReaderWorker:
    """Own a cloned reader and execute synchronous operations on its sole thread."""

    def __init__(self, reader: TimeFReader, name: str) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        try:
            self._reader, self._thread = self.executor.submit(lambda: (clone_reader(reader), get_ident())).result()
        except BaseException:
            self.executor.shutdown(wait=True)
            raise
        self._inspection = ViewerInspection(self._reader)

    def run(self, operation: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        """Run an inspection operation from an already scheduled job.

        Returns:
            The synchronous operation's result.

        Raises:
            TimeFValidationError: If called outside the reader's owning thread.
        """
        if get_ident() != self._thread:
            raise TimeFValidationError("inspection must run on its owning worker")
        return operation(self._inspection, *args, **kwargs)

    async def call(self, operation: Callable[..., _Result], *args: Any, **kwargs: Any) -> _Result:
        """Return the result of an operation explicitly dispatched to its owning thread."""

        def invoke() -> _Result:
            return self.run(operation, *args, **kwargs)

        return cast("_Result", await asyncio.wrap_future(self.executor.submit(invoke)))

    def close(self) -> None:
        """Close the reader on its owning thread after queued jobs have stopped."""
        try:
            self.executor.submit(self._reader.close).result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
