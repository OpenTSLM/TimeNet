"""The :class:`Access` enum: how a user obtains a dataset's data.

Access (what you must do to get the bytes) is a separate axis from the license (how you may
redistribute them). A dataset can be openly licensed yet credentialed, or the reverse.
"""

from enum import StrEnum, unique


@unique
class Access(StrEnum):
    """How a user obtains the source data."""

    OPEN = "open"
    """Anyone can fetch the data; no account or agreement is needed."""
    CREDENTIALED = "credentialed"
    """A credentialed account and a signed agreement are needed, for example a PhysioNet DUA."""
    RESTRICTED = "restricted"
    """Access is granted case by case, for example an AWS-IAM grant or a direct request."""
