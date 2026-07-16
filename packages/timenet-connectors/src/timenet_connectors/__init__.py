"""Dataset connectors for TimeNet.

Concrete connectors live under ``timenet_connectors.datasets.<org>.<name>`` and each exposes a
module-level ``CONNECTOR``; they are found lazily by dataset id (see :mod:`timenet_connectors.discovery`),
so there is no central registry. Reusable bases (e.g. for the HuggingFace Hub) live under
``timenet_connectors.bases``. The connector contract itself is :class:`timenet.connectors.BaseConnector`.

The module-level helpers :func:`build` and :func:`load` (from :mod:`timenet_connectors.api`) are the
producer-side shortcuts for curating and consuming a dataset from local code.
"""

from timenet_connectors.api import build as build, load as load
