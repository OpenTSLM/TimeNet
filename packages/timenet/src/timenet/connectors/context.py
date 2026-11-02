"""The parent releases of one connector build, which the engine opens before the connector runs."""

from contextlib import ExitStack
from dataclasses import dataclass, field

from timenet.reader import TimeFReader
from timenet.types.reference_keys import ParentAlias
from timenet.types.references import DatasetRef


@dataclass
class BuildContext:
    """The declared parent releases of one connector build, opened by parent alias.

    The engine opens the parents from the output registry before the download starts, and records
    each of them in the composed dataset after :meth:`~timenet.connectors.BaseConnector.compose`
    returns. The context only holds the readers and closes them.
    """

    references: dict[ParentAlias, DatasetRef] = field(default_factory=dict)
    parents: dict[ParentAlias, TimeFReader] = field(default_factory=dict)

    def close(self) -> None:
        """Release the parent readers once the writer has consumed them.

        The reader opened last closes first. A reader that fails to close does not keep the others
        open: each of them still closes, and this method then raises the error.
        """
        stack = ExitStack()
        for parent in self.parents.values():
            stack.callback(parent.close)
        stack.close()

    def __enter__(self) -> "BuildContext":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
