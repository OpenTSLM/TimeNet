"""The sleep-edfx connector package.

This package exposes no ``CONNECTOR`` yet. Discovery instantiates each connector it finds and
reads a dataset id from its card. This connector has no card and no ``convert`` body.
"""

from timenet_connectors.datasets.physionet.sleep_edfx.connector import (
    SleepEdfxConnector as SleepEdfxConnector,
    SleepEdfxRecording as SleepEdfxRecording,
    SleepEdfxSource as SleepEdfxSource,
)
