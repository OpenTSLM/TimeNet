"""The CLI flags that set the default list of sources.

Each flag overrides one variable of :mod:`timenet.config`, and a flag that is not given is ``None``.
Pass the values to :func:`~timenet.client.default_sources` or :func:`~timenet.client.default_store`.
"""

import typer


LOCAL_REGISTRY = typer.Option(
    None,
    "--local-registry",
    metavar="PATH",
    help="A local registry: the first source that a lookup checks, and the place where downloads and builds land "
    "(default: $TIMENET_LOCAL_REGISTRY, else ~/.timenet/registry).",
)
REMOTE_REGISTRY = typer.Option(
    None,
    "--remote-registry",
    metavar="URL|PATH",
    help="A remote registry, as a URL or a local path: the second source that a lookup checks, if the local "
    "registry does not have the dataset (default: $TIMENET_REMOTE_REGISTRY, else timenet://).",
)
AUTO_BUILD = typer.Option(
    None,
    "--auto-build/--no-auto-build",
    help="Whether installed connectors build a dataset that neither the local nor the remote registry has "
    "(default: $TIMENET_AUTO_BUILD, else true).",
)
