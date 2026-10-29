"""What the authors' preprocessing fixes: the published scale, the trace length, and where the padding goes.

The authors built the HDF5 files with something equivalent to ``generate_h5.py --scale 2 --use_all_leads
--new_freq 400 --new_len 4096`` from their ``ecg-preprocessing`` repository (its README, "CODE-15%"). Its
``preprocess_ecg`` multiplies by the scale, then pads a shorter trace with zeros, ``(4096 - n) // 2`` before
and the rest after, or cuts a longer one around its centre. This module undoes the scale and the padding, and
redoes both to rebuild the authors' model input exactly.
"""

from typing import NamedTuple

import numpy as np

from timenet.errors import TimeFFormatError


SCALE = 2
"""The published values are the millivolt values times 2. The author confirms the factor in
``ecg-age-prediction`` issue 5; the READMEs' "scale 1e-4V" sentence is wrong."""
TRACE_SAMPLES = 4096
"""Every published trace's length: ``--new_len 4096``."""
# Halving a float32 only lowers its exponent, so it is exact, and SCALE * (value * HALF) == value.
_HALF = np.float32(1 / SCALE)


class Padding(NamedTuple):
    """The zeros ``preprocess_ecg`` added before and after a trace."""

    start: int
    end: int

    @property
    def n_values(self) -> int:
        """Return the number of samples the trace holds without its padding."""
        return TRACE_SAMPLES - self.start - self.end


def centred_padding(leading_zeros: int, trailing_zeros: int) -> Padding:
    """Return the most padding ``preprocess_ecg`` could have added, given a trace's runs of all-lead zeros.

    The authors pad ``(4096 - n) // 2`` samples before a trace of ``n`` samples and the rest after, so the
    padding after is the padding before or one more. Of the zeros at either end, only those the rule can
    explain are padding; any others are zeros of the published trace, and stay. A trace with no zeros at its
    end, such as one cut from a longer recording, has no padding.

    Args:
        leading_zeros: Samples, from the first, that are zero in every lead.
        trailing_zeros: Samples, from the last, that are zero in every lead.

    Returns:
        The padding before and after the trace.
    """
    start = min(leading_zeros, trailing_zeros)
    end = start + 1 if trailing_zeros > start else start
    return Padding(start=start, end=end)


def to_millivolts(trace: np.ndarray, padding: Padding) -> np.ndarray:
    """Return a published trace's samples without its padding, in millivolts.

    Args:
        trace: One published ``(4096, 12)`` float32 trace.
        padding: Its padding.

    Returns:
        The ``(n, 12)`` float32 samples, halved, which is exact.
    """
    return trace[padding.start : TRACE_SAMPLES - padding.end] * _HALF


def authors_input(millivolts: np.ndarray, padding: Padding) -> np.ndarray:
    """Rebuild the authors' model input from a trace's millivolt samples and its padding.

    This is the published trace, bit for bit: the samples times :data:`SCALE`, between the zeros.

    Args:
        millivolts: The ``(n, 12)`` float32 samples, in the published lead order.
        padding: The trace's padding, from its Record's metadata.

    Returns:
        The ``(4096, 12)`` float32 trace.

    Raises:
        TimeFFormatError: If the samples and the padding don't add up to 4096 samples.
    """
    if millivolts.shape[0] != padding.n_values:
        raise TimeFFormatError(
            f"a trace of {millivolts.shape[0]} samples with padding {tuple(padding)} isn't {TRACE_SAMPLES} samples"
        )
    trace = np.zeros((TRACE_SAMPLES, millivolts.shape[1]), dtype=np.float32)
    trace[padding.start : TRACE_SAMPLES - padding.end] = millivolts * np.float32(SCALE)
    return trace
