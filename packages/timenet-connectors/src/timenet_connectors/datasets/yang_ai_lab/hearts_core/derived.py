"""Create child-owned signals while checking them against frozen benchmark inputs."""

from collections.abc import Callable

import numpy as np
import pyarrow as pa

from timenet.dataset import RegularAxis, Signal
from timenet.errors import TimeFFormatError


def derived_signal(
    expected: Signal, original: Signal, loader: Callable[[], pa.Array], *, parent: str, atol: float = 1e-8
) -> Signal:
    """Copy the benchmark signal structure with values computed from an original signal.

    Returns:
        A child-owned signal with a lazy, checked transformation and explicit provenance.
    """

    def checked() -> pa.Array:
        values = loader()
        reference = expected.to_arrow()
        if len(values) != len(reference) or not np.allclose(
            values.to_numpy(), reference.to_numpy(), rtol=1e-7, atol=atol, equal_nan=False
        ):
            raise TimeFFormatError(
                f"{expected.id}: values derived from {parent}/{original.id} do not reproduce the pinned HEARTS input"
            )
        return values

    return Signal.from_loader(
        spec=expected.spec,
        time_axis=expected.time_axis,
        id=expected.id,
        name=expected.name,
        loader=checked,
        n_values=expected.n_values,
        time_offsets_loader=expected.time_offsets_loader,
        source_id=original.id,
        annotations=expected.annotations,
        metadata={"parent_dataset": parent, "parent_signal": original.id},
    )


def timestamp_positions(original: Signal, timestamps: np.ndarray, *, origin_us: int) -> np.ndarray:
    """Locate exact source timestamps, rejecting absent samples instead of interpolating.

    Returns:
        Source row positions in the requested order.

    Raises:
        TimeFFormatError: If any requested timestamp is absent.
    """
    source = sample_offsets(original) + origin_us
    positions = np.searchsorted(source, timestamps)
    if np.any(positions >= len(source)) or not np.array_equal(source[positions], timestamps):
        raise TimeFFormatError(f"{original.id}: HEARTS requests timestamps absent from the original recording")
    return positions


def sample_offsets(signal: Signal) -> np.ndarray:
    """Return integer offsets for either a regular or an explicit-time signal."""
    axis = signal.time_axis
    if isinstance(axis, RegularAxis):
        return np.fromiter(
            (axis.time_offset_us(index) for index in range(signal.n_values)), dtype=np.int64, count=signal.n_values
        )
    return signal.time_offsets_us()
