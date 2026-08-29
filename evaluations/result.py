"""One object carrying everything a run produced."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import platform
import sys

from pydantic import BaseModel

from evaluations.errors import EvaluationError
from evaluations.harness import Artifact, Measurement
from timenet.provenance import build_env


class Environment(BaseModel):
    """What produced a result.

    Timings from two machines are not comparable, so a result without its environment cannot be
    read later.
    """

    # Interpreter version.
    python: str
    # Operating system and machine, for example darwin-arm64.
    platform: str
    # Logical processors visible to the run.
    cpu_count: int
    # Installed distributions and their versions.
    packages: dict[str, str]


class EvaluationResult(BaseModel):
    """Everything one run produced. This is the deliverable."""

    # Identifies the run, and names its output directory.
    run_id: str
    # When the run began.
    started_at: datetime
    # The directory the release was read from.
    source_path: Path
    # What produced the result.
    environment: Environment
    # One per format, carrying its size on disk.
    artifacts: tuple[Artifact, ...]
    # Two per format: what its write cost, and its median read.
    measurements: tuple[Measurement, ...]


def collect_environment() -> Environment:
    """Record the interpreter, the machine, and every installed package.

    Reuses ``timenet.provenance.build_env``, which already collects this for the manifest.

    An empty package list is treated as a failure rather than written out. A result that
    satisfies its own schema while carrying no provenance is worse than no result, because
    nothing downstream can tell it apart from a good one.

    Returns:
        The environment this run is executing in.

    Raises:
        EvaluationError: If ``build_env`` returns no packages.
    """
    collected = build_env()
    packages = collected.get("packages", {})

    if not packages:
        raise EvaluationError(
            f"timenet.provenance.build_env returned no packages, so the result would carry no "
            f"provenance; it returned keys {sorted(collected)}"
        )

    return Environment(
        python=sys.version.split()[0],
        platform=f"{sys.platform}-{platform.machine()}",
        cpu_count=os.cpu_count() or 0,
        packages=dict(sorted(packages.items())),
    )
