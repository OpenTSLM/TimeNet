"""What the writer checks before it commits, as queries that must return no rows.

These are not tests. They run inside the load transaction, against the loaded tables, between the
last insert and ``COMMIT``. They stand in for the primary and foreign keys the shipped database does
not declare.

That trade is measured. On ECG-QA (1.35M control rows), plain tables are 19.7 MB and the same
content with keys and indexes is 189.0 MB. The write takes 3.9 s against 11.9 s. One process writes
a version once, and the version is immutable afterwards. A constraint carried in the file therefore
re-checks, for the rest of the dataset's life, something that cannot change. One check per invariant
here costs the load one bulk anti-join, and costs every later reader nothing.

A failed check aborts the transaction, so a build that does not hold together publishes nothing and
leaves no file behind. Each check counts every offending row rather than stopping at the first, and
the message that it raises names the first few. A resolved id that names nothing is therefore stored
as null and caught here, rather than failing its insert.

Each module below holds one family of invariant, and a family says what it guarantees rather than
which table it reads:

- :mod:`~timenet.control_plane.checks.identity`: each row is named one time, and each surrogate id
  is dense.
- :mod:`~timenet.control_plane.checks.references`: every id that names another row finds one.
- :mod:`~timenet.control_plane.checks.shape`: a row does not contradict itself.
- :mod:`~timenet.control_plane.checks.payload`: a typed task holds what its class declares.
- :mod:`~timenet.control_plane.checks.values`: the control plane and the values plane agree.

Most of the checks are generated, because the repetitive ones follow from the table maps in
:mod:`timenet.control_plane.schema` and from the declaration in
:mod:`timenet.control_plane.payload`. A new table or a new task field therefore brings its checks
with it.
"""

from typing import Final

from timenet.control_plane.checks._check import SAMPLE_ROWS, Check
from timenet.control_plane.checks.identity import IDENTITY
from timenet.control_plane.checks.payload import PAYLOAD
from timenet.control_plane.checks.references import REFERENCES
from timenet.control_plane.checks.shape import SHAPE
from timenet.control_plane.checks.values import VALUES


# RUF067 asks an __init__ to re-export only. The suite is one line, and it reads best beside the
# families that it joins, so it stays here.
VALIDATIONS: Final[tuple[Check, ...]] = (*IDENTITY, *REFERENCES, *SHAPE, *PAYLOAD, *VALUES)  # noqa: RUF067
"""Every invariant the dropped key constraints used to enforce, in the order the writer runs them.

The order goes from the cheap and general to the narrow: a row must be identified before a
reference to it means anything, and a task payload is worth checking only after the rows that hold
it are sound. The writer stops at the first family member that finds a row.
"""

__all__ = ["IDENTITY", "PAYLOAD", "REFERENCES", "SAMPLE_ROWS", "SHAPE", "VALIDATIONS", "VALUES", "Check"]
