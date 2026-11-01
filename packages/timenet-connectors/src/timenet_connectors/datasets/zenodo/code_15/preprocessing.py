"""What the authors' preprocessing fixes: the published scale, the trace length, and where the padding goes.

The authors built the HDF5 files with something equivalent to ``generate_h5.py --scale 2 --use_all_leads
--new_freq 400 --new_len 4096`` from their ``ecg-preprocessing`` repository (its README, "CODE-15%"). Its
``preprocess_ecg`` multiplies by the scale, then pads a shorter trace with zeros, ``(4096 - n) // 2`` before
and the rest after, or cuts a longer one around its centre.

The conversion only undoes this, with :func:`to_millivolts`: a Record's leads are in millivolts and leave the
padding out, so they hold the recording, not the network's fixed-size input. The padding is kept in the
Record's metadata. A cut trace can't be uncut, and stays 4096 samples.

:func:`authors_input` redoes both, but the conversion never calls it. It is for a user who needs the authors'
exact model input, such as to run their published models on a Record, and the tests use it to show the undo
loses nothing: on the pinned release, every one of the 345,109 traces rebuilds bit for bit.
"""

from typing import NamedTuple

import numpy as np

from timenet.errors import TimeFFormatError


SCALE = 2
"""The published values are the millivolt values times 2. The author confirms the factor in
``ecg-age-prediction`` issue 5; the READMEs' "scale 1e-4V" sentence is wrong.

Measured against four of the authors' raw exams, the factor is 1.953125 (= 2 x 1000/1024), not 2: see
VERIFICATION.md #19. Which side carries the 1000/1024 isn't known yet, so the factor stays the author's."""
TRACE_SAMPLES = 4096
"""Every published trace's length: ``--new_len 4096``."""
# Halving a float32 at least twice the smallest normal only lowers its exponent, so SCALE * (value * _HALF)
# == value. The release's smallest nonzero magnitude is 1.9e-36, far above that bound of 2.4e-38.
_HALF = np.float32(1 / SCALE)


class Padding(NamedTuple):
    """How many zero samples ``preprocess_ecg`` added before and after a trace.

    Both are counts of samples, not positions: the recording starts at sample ``n_before`` of the published
    trace and ends ``n_after`` samples before its end.
    """

    n_before: int
    n_after: int

    @property
    def n_values(self) -> int:
        """Return the number of samples the trace holds without its padding.

        That is the published trace's 4096 samples, less the zeros before and the zeros after: a trace of 2,934
        samples between 581 zeros on each side has 4096 - 581 - 581 = 2,934.
        """
        return TRACE_SAMPLES - self.n_before - self.n_after

    @property
    def values(self) -> slice:
        """Return the samples of the published trace that hold the recording, between the zeros."""
        return slice(self.n_before, TRACE_SAMPLES - self.n_after)


def centred_padding(leading_zeros: int, trailing_zeros: int) -> Padding:
    """Return the most padding ``preprocess_ecg`` could have added, given a trace's runs of all-lead zeros.

    The authors pad ``(4096 - n) // 2`` samples before a trace of ``n`` samples and the rest after, so the
    padding after is the padding before or one more. Of the zeros at either end, only those the rule can
    explain are padding; any others are zeros of the published trace, and stay. A trace with no zeros at its
    end, such as one cut from a longer recording, has no padding.

    This is an inference, not the authors' record: the raw lengths aren't published. A trace whose recording
    itself starts and ends with all-lead zeros loses them as padding, and one with real zeros at its end alone
    loses one. On the pinned release, 344,932 of the 345,109 traces have the layouts of the known recording
    lengths: 581 zeros each side, which four of the authors' raw exams confirm (2,200 samples at 300 Hz
    resample to 2,934), 48, or none. The other 177 may be misread. Either way, only exact zeros are dropped,
    and the Record's metadata keeps the counts, so :func:`authors_input` still rebuilds the published trace.

    Args:
        leading_zeros: Samples, from the first, that are zero in every lead.
        trailing_zeros: Samples, from the last, that are zero in every lead.

    Returns:
        The padding before and after the trace.
    """
    n_before = min(leading_zeros, trailing_zeros)
    # The padding after is the padding before, or one more if the trailing zeros allow it.
    n_after = min(trailing_zeros, n_before + 1)
    return Padding(n_before=n_before, n_after=n_after)


def to_millivolts(trace: np.ndarray, padding: Padding) -> np.ndarray:
    """Return a published trace's samples without its padding, in millivolts.

    Args:
        trace: One published ``(4096, 12)`` float32 trace.
        padding: Its padding.

    Returns:
        The ``(n, 12)`` float32 samples, halved, which is exact.
    """
    return trace[padding.values] * _HALF


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
    trace[padding.values] = millivolts * np.float32(SCALE)
    return trace
