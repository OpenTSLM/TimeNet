"""Explicit execution and lifetime ownership for inspection readers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from typing import Concatenate, ParamSpec, TypeVar, cast

from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.reader.inspection import ReaderInspection


_Result = TypeVar("_Result")
_Params = ParamSpec("_Params")


class ReaderWorker:
    """Own a cloned reader and execute synchronous operations on its sole thread."""

    def __init__(self, reader: TimeFReader, name: str) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        try:
            self._reader, self._thread = self._executor.submit(lambda: (reader.clone(), get_ident())).result()
        except BaseException:
            self._executor.shutdown(wait=True)
            raise
        self._inspection = self._reader.inspect()

    def run(
        self,
        operation: Callable[Concatenate[ReaderInspection, _Params], _Result],
        *args: _Params.args,
        **kwargs: _Params.kwargs,
    ) -> _Result:
        """Run an inspection operation on the owning thread.

        Returns:
            The synchronous operation's result.

        Raises:
            TimeFValidationError: If called outside the reader's owning thread.
        """
        if get_ident() != self._thread:
            raise TimeFValidationError("inspection must run on its owning worker")
        return operation(self._inspection, *args, **kwargs)

    async def call(
        self,
        operation: Callable[Concatenate[ReaderInspection, _Params], _Result],
        *args: _Params.args,
        **kwargs: _Params.kwargs,
    ) -> _Result:
        """Return the result of an operation explicitly dispatched to its owning thread."""

        def invoke() -> _Result:
            return self.run(operation, *args, **kwargs)

        return cast("_Result", await asyncio.wrap_future(self._executor.submit(invoke)))

    def close(self) -> None:
        """Close the reader on its owning thread after queued requests have finished."""
        try:
            self._executor.submit(self._reader.close).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
