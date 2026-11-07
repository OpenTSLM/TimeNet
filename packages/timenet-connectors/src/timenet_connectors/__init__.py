"""Dataset connectors for TimeNet.

Concrete connectors live under ``timenet_connectors.datasets.<org>.<name>``. Each one exposes a
module-level ``CONNECTOR``. There is no central registry. TimeNet finds them lazily by dataset id
(see :mod:`timenet_connectors.discovery`). Reusable bases, for example for the HuggingFace Hub, live
under ``timenet_connectors.bases``. The connector contract is :class:`timenet.connectors.BaseConnector`.

To build a dataset from Python, call ``timenet.build("org/name")``, and read it with
``TimeNet().load("org/name")``.
"""
