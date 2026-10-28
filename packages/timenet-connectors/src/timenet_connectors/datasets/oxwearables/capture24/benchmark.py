"""What the authors' benchmark fixes: the window, the scored label schemes, and the split.

These come from the authors' code, ``prepare_data.py`` and ``benchmark.py``, and from the paper, not from
the archive. ``windows`` applies the window, and ``tasks`` the schemes and the split.
"""

from timenet.types import US_PER_S, Split
from timenet_connectors.datasets.oxwearables.capture24.release import SAMPLE_PERIOD_US
from timenet_connectors.datasets.oxwearables.capture24.vocabulary import LabelScheme


WINDOW_US = 10 * US_PER_S
"""The authors' window: 10 s, ``prepare_data.py --winsec 10``."""
WINDOW_SAMPLES = WINDOW_US // SAMPLE_PERIOD_US

BENCHMARK_SCHEMES = (LabelScheme.WILLETTS_SPECIFIC_2018, LabelScheme.WALMSLEY_2020)
"""The schemes the authors' ``benchmark.py`` scores, in its default ``--annots`` order. Each becomes a task per window.

So every window has two tasks: one in ``WillettsSpecific2018`` (10 classes) and one in ``Walmsley2020``
(4 classes). The other four schemes (``WillettsMET2018``, ``DohertySpecific2018``, ``Willetts2018`` and
``Doherty2018``) have no tasks; the dataset still holds their vocabulary and mapping as annotations.
"""

FIRST_TEST_PARTICIPANT = 101
"""The first test participant: the paper's test set is P101 to P151, and the authors' ``benchmark.py`` tests
``range(101, 152)``. The README of the authors' repository says P102, which neither follows.
"""


def split_of(participant_number: int) -> Split:
    """Return a participant's benchmark split: P001 to P100 train, P101 to P151 test.

    Returns:
        ``Split.TEST`` from :data:`FIRST_TEST_PARTICIPANT` on, else ``Split.TRAIN``.
    """
    return Split.TEST if participant_number >= FIRST_TEST_PARTICIPANT else Split.TRAIN
