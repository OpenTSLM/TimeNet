"""Dataset connectors for TimeNet.

Concrete connectors live under ``timenet_connectors.datasets.<org>.<name>``. Each one exposes a
module-level ``CONNECTOR``. There is no central registry. TimeNet finds them lazily by dataset id
(see :mod:`timenet_connectors.discovery`). Reusable bases, for example for the HuggingFace Hub, live
under ``timenet_connectors.bases``. The connector contract is :class:`timenet.connectors.BaseConnector`.

The module-level helpers :func:`build` and :func:`load` (from :mod:`timenet_connectors.api`) are the
producer-side shortcuts. Use them to curate and consume a dataset from local code.
"""

from timenet_connectors.api import build as build, load as load
