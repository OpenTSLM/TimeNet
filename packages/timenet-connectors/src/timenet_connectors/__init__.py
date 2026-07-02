"""Dataset connectors for TimeNet.

Concrete connectors live under ``timenet_connectors.datasets.<org>.<name>`` and each exposes a
module-level ``CONNECTOR``; they are found lazily by dataset id (see :mod:`timenet_connectors.discovery`),
so there is no central registry. Reusable bases (e.g. for the HuggingFace Hub) live under
``timenet_connectors.bases``. The connector contract itself is :class:`timenet.connectors.BaseConnector`.
"""
