"""The ``timenet-build`` producer CLI and the built-in builder backend.

The SDK loads :mod:`timenet_connectors.builder.backend` through the ``timenet.builders`` entry point.
This package imports none of its modules, so that loading the backend does not load the CLI and its
libraries. The ``timenet-build`` console script runs :func:`timenet_connectors.builder.cli.main`.
"""
