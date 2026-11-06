"""The sleep-edfx connector package: the connector implementation and the dataset card."""

from timenet_connectors.datasets.physionet.sleep_edfx.connector import (
    CONNECTOR as CONNECTOR,
    SleepEdfxConnector as SleepEdfxConnector,
)
from timenet_connectors.datasets.physionet.sleep_edfx.source import (
    SleepEdfxRecording as SleepEdfxRecording,
    SleepEdfxSource as SleepEdfxSource,
)
