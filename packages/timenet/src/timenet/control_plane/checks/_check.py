"""One named check: what it defends, the query that finds a breach, and what to do about it.

A check is a query that must return no rows. A row is a breach of the invariant. The name and the
remedy turn a failed load into a message that says which invariant broke and what the caller must
do next.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final


SAMPLE_ROWS: Final = 3
"""How many offending rows a failure message names."""


@dataclass(frozen=True)
class Check:
    """One invariant the dropped key constraints used to enforce, as a query that must return no rows."""

    name: str
    """A stable identifier for this check.

    It stays the same when the message or the query changes, so a caller can refer to one check
    across versions.
    """
    invariant: str
    """What the check defends, in one sentence."""
    sql: str
    """The query. It must return no rows, and its first column must identify the offending row."""
    remedy: str
    """What the caller must do when the check finds a row."""

    @property
    def count_query(self) -> str:
        """The query that counts the offending rows.

        Returns:
            A query with one row and one column. The load runs this one, not :attr:`sql`, because a
            count does not materialize a breach of three million rows.
        """
        return f"SELECT count(*) FROM ({self.sql}) AS offending"  # noqa: S608 - self.sql is this package's own text

    @property
    def sample_query(self) -> str:
        """The query that names the first offending rows.

        Returns:
            A query of at most :data:`SAMPLE_ROWS` rows. The load runs this one only after
            :attr:`count_query` finds a breach.
        """
        return f"SELECT * FROM ({self.sql}) AS offending LIMIT {SAMPLE_ROWS}"  # noqa: S608 - own text

    def failure(self, total: int, sample: Sequence[object]) -> str:
        """Return the message a failed check raises.

        Args:
            total: How many rows the check found.
            sample: The identifier of up to :data:`SAMPLE_ROWS` of those rows.

        Returns:
            The message. It names the invariant, the count, some offending rows, and the remedy.
        """
        listed = ", ".join(str(value) for value in sample)
        return (
            f"{self.name}: {self.invariant} Found {total} row(s), for example {listed}. "
            f"{self.remedy} The version was not published."
        )
