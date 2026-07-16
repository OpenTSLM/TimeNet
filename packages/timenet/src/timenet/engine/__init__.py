"""The curation engine: drive a connector through ``download -> convert -> derive_schema -> store``."""

from timenet.engine.engine import run_pipeline


__all__ = ["run_pipeline"]
