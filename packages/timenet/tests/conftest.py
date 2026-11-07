"""Shared pytest configuration for the timenet test suite.

Registers Hypothesis profiles so the example count can be tuned per environment via the
``HYPOTHESIS_PROFILE`` environment variable (``dev`` for a fast edit loop, ``ci`` for the full run).
"""

import os
from pathlib import Path
import sys

from hypothesis import HealthCheck, settings


# Make the test helpers of this directory (e.g. ``_demo_child``) importable by name under pytest's
# importlib import mode, which does not add test dirs to ``sys.path``.
sys.path.insert(0, str(Path(__file__).parent))

settings.register_profile("dev", max_examples=10)
settings.register_profile("ci", max_examples=100, print_blob=True)
settings.register_profile(
    "nightly",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
