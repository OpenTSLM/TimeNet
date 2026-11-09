"""CLI flags for registry locations and automatic builds.

Each flag overrides the matching ``TIMENET_*`` environment variable.
Omitted flags have the value ``None``.
"""

import typer


LOCAL_REGISTRY = typer.Option(
    None,
    "--local-registry",
    metavar="PATH",
    help="The local registry, where downloads and builds land and which a lookup checks first (default: "
    "$TIMENET_LOCAL_REGISTRY, else ~/.timenet/registry).",
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
