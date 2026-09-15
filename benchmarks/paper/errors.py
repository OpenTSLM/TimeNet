"""Errors the paper harness raises.

All of them derive from :class:`timenet.errors.TimeNetError`, so a caller that already handles
TimeNet errors handles these too.
"""

from timenet.errors import TimeNetError


class PaperHarnessError(TimeNetError):
    """Base class for every error the paper measurement harness raises."""


class CellRecordError(PaperHarnessError, ValueError):
    """A cell record is malformed, or its fields contradict each other."""


class ProvenanceError(PaperHarnessError):
    """A provenance gate refused the campaign.

    The aggregator raises this instead of warning. A campaign that silently mixes two states
    produces a plausible table that is wrong.
    """


class RenderError(PaperHarnessError):
    """A generator cannot render what the aggregate holds."""


class CacheControlError(PaperHarnessError):
    """A cache was not in the state the measurement declared.

    Raised when the page-cache drop failed, when the canary says it did not work, or when a lane's
    derived cache contradicts the state it declared.
    """


class StorageMeasurementError(PaperHarnessError, ValueError):
    """A storage tree cannot be measured under the storage rule."""


class ItemRegistryError(PaperHarnessError, ValueError):
    """The item registry is inconsistent, or an enumerator broke its own contract."""


class LaneError(PaperHarnessError):
    """A lane cannot open its representation, or cannot deliver an item."""


class ObservationError(PaperHarnessError, ValueError):
    """One timed observation is malformed, or the runner cannot fold a set of them."""
