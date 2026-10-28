"""The authors' 10 s windows and their labels, as ``prepare_data.make_windows`` defines them.

``make_windows`` resamples the recording into 10 s bins anchored at its first sample
(``resample("10s", origin="start")``), so after a gap the bins stay on the first segment's grid. It keeps a
bin that holds exactly 1000 samples, no NaN, and at least one labelled sample. A bin that a gap cuts holds
fewer samples, so no kept window crosses a gap.

A window's class in a scheme is the most common class among its labelled samples. Unlabelled samples
don't count, so a window with one labelled sample takes that sample's class. A tie goes to the class first
in Python string order, since ``Series.mode`` sorts the modes it returns and ``make_windows`` takes the
first. Every window keeps its 1000 samples; nothing is dropped or relabelled to break the tie.

``make_windows`` also drops a window with a NaN acceleration value. The loader never meets one: a NaN
fails the build when its signal is read.
"""

from dataclasses import dataclass

import numpy as np

from timenet_connectors.datasets.oxwearables.capture24.release import WINDOW_SAMPLES, WINDOW_US
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import NO_LABEL, LabelDictionary, LabelScheme


@dataclass(frozen=True)
class Windows:
    """A recording's kept windows."""

    start_us: np.ndarray
    """Each window's first sample's time from the recording's first sample, int64 microseconds."""
    classes: dict[LabelScheme, np.ndarray]
    """Each window's class per scheme, as an index into ``LabelDictionary.vocabulary(scheme)``."""

    def __len__(self) -> int:
        return len(self.start_us)


def make_windows(
    offsets_us: np.ndarray, labels: np.ndarray, dictionary: LabelDictionary, schemes: tuple[LabelScheme, ...]
) -> Windows:
    """Return the windows ``make_windows`` keeps, with each one's class in each scheme.

    Args:
        offsets_us: Each sample's time from the first sample.
        labels: Each sample's raw label code, or ``NO_LABEL``.
        dictionary: The label dictionary the codes index.
        schemes: The schemes to label the windows in.

    Returns:
        The kept windows, in time order.
    """
    bins = offsets_us // WINDOW_US
    starts = np.flatnonzero(np.diff(bins, prepend=bins[0] - 1))
    sizes = np.diff(starts, append=len(bins))
    full = starts[sizes == WINDOW_SAMPLES]
    samples = full[:, None] + np.arange(WINDOW_SAMPLES)
    window_labels = labels[samples]
    labelled = (window_labels != NO_LABEL).any(axis=1)
    kept, window_labels = full[labelled], window_labels[labelled]
    classes = {}
    for scheme in schemes:
        vocabulary_size = len(dictionary.vocabulary(scheme))
        # One extra slot counts the unlabelled samples, which then drop out.
        codes = np.append(dictionary.class_codes(scheme), vocabulary_size)[window_labels]
        rows = np.arange(len(kept))[:, None] * (vocabulary_size + 1)
        counts = np.bincount((rows + codes).ravel(), minlength=len(kept) * (vocabulary_size + 1))
        counts = counts.reshape(len(kept), vocabulary_size + 1)[:, :vocabulary_size]
        # argmax takes the first of equal counts: the class first in string order, as Series.mode does.
        classes[scheme] = counts.argmax(axis=1).astype(np.int16)
    return Windows(start_us=offsets_us[kept], classes=classes)
