"""The one exception type the evaluation suite raises."""

from __future__ import annotations


class EvaluationError(Exception):
    """Any failure the evaluation suite reports itself.

    Every raise names the values behind the failure, not only that something went wrong. A reader
    of the traceback should not have to re-run to find out which path or which package was at
    fault.

    The suite raises this and stops. It does not catch a precondition failure and carry on with a
    partial run: a report that looks complete and is not is worse than no report.
    """
