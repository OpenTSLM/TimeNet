"""Signed SWOT RiverSP observations and coastal forecast tasks."""

from .connector import CONNECTOR, SwotRiverSlopeConnector
from .model_inputs import model_inputs


__all__ = ["CONNECTOR", "SwotRiverSlopeConnector", "model_inputs"]
