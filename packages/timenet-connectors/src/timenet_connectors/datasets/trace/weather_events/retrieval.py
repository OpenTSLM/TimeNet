"""One row of a retrieval file as published: a window's generated description and its event report."""

from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from timenet.errors import TimeFValidationError
from timenet_connectors.datasets.trace.weather_events._fields import EventLabelField, Text, nest
from timenet_connectors.datasets.trace.weather_events.archive import TraceFileRef
from timenet_connectors.datasets.trace.weather_events.vocabulary import Channel


class ChannelSentences(BaseModel):
    """One generated sentence per Channel, describing that channel over the window."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE channel sentences")

    # One field per Channel, named by its JSON key, as WindowReadings names the values.
    temperature: Text
    precipitation: Text
    relative_humidity: Text
    visibility: Text
    wind_u: Text
    wind_v: Text
    sky_code: Text

    def sentence(self, channel: Channel) -> str:
        """Return the sentence about one channel."""
        return getattr(self, channel.value)


class GeneratedWindowDescription(BaseModel):
    """ChatGPT-written sentences about a window, one per channel; used as the retrieval prompt.

    The retrieval files publish it under the window JSON's keys, but every value is text the authors
    had ChatGPT write from the window (paper Appendix B.4). Its ``DATE`` and ``location`` are
    sentences about the window's dates and place, not timestamps or a County.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE description")

    # Sentences, despite the keys they share with the window JSON.
    DATE: Text
    location: Text
    labels: Text
    """The weather the window shows, as a list of words, such as ``Clear, Hot, Cloudy``."""
    channels: ChannelSentences

    @model_validator(mode="before")
    @classmethod
    def _group_channels(cls, data: Any) -> Any:
        return nest(data, tuple(Channel), "channels")

    @property
    def text(self) -> str:
        """Return the description as the authors' ``generate_dsp`` composes it for their text encoder."""
        # Spacing as published: generate_dsp joins the channel sentences with " \n ".
        channels = " \n ".join(self.channels.sentence(channel) for channel in Channel)
        return (
            f"Weather time series location: {self.location} Time range: {self.DATE} "
            f"The weather is {self.labels}. {channels}"
        )


class EventReport(BaseModel):
    """One event window's report, as the retrieval files publish it."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE event report")

    event_id: StrictInt
    # RetrievalIndex.from_files checks the label against the window's, which rules out NO_EVENT.
    event_type: EventLabelField
    narrative: Text

    @property
    def text(self) -> str:
        """Return the report as the authors' ``generate_er`` writes it for their text encoder."""
        return f"The weather event is {self.event_type.label}. {self.narrative}"


class RetrievalRow(BaseModel):
    """One row of a retrieval file: a window's generated description and, for an event window, its report.

    Every row names a window of the JSON files by its id. The file's ``timeseries`` column copies that
    window's series, which its Record already holds, so it is not read.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="TRACE retrieval row")

    window_id: StrictInt = Field(alias="id")
    description: GeneratedWindowDescription
    event_report: EventReport | None = Field(alias="events")
    """The NOAA report of the event that followed the window; ``None`` for a window without one."""

    @property
    def key(self) -> str:
        """Return the window's key in the JSON files, which name windows by their id as text."""
        return str(self.window_id)


# The columns RetrievalRow reads, and the rows per read, so neither the checks nor the task stream
# hold a whole file's descriptions.
_RETRIEVAL_COLUMNS = ["id", "description", "events"]
_RETRIEVAL_BATCH_ROWS = 1024


def read_retrieval_rows(file: TraceFileRef) -> Iterator[RetrievalRow]:
    """Read and validate a Stage 2 parquet file one batch at a time.

    Yields:
        Each row, in file order.

    Raises:
        TimeFValidationError: If a row is invalid, naming the file and the row's id.
    """
    with pq.ParquetFile(file.path) as parquet:
        for batch in parquet.iter_batches(batch_size=_RETRIEVAL_BATCH_ROWS, columns=_RETRIEVAL_COLUMNS):
            for published in batch.to_pylist():
                try:
                    row = RetrievalRow.model_validate(published)
                # A pydantic ValidationError is a ValueError.
                except ValueError as exc:
                    raise TimeFValidationError(f"{file.path} (id {published['id']!r}): {exc}") from exc
                yield row
