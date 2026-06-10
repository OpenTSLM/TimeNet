# TimeNet 2.0 — System Design

## What TimeNet is

TimeNet is a Python library and CLI for registering, fetching, and exploring time-series datasets in a unified format called TimeF.

TimeNet is not for: training, inference, model definitions, evaluation metrics.

---

## Architectural overview

![TimeNet 2.0 architecture diagram](assets/architecture.svg)

- **`BaseConnector`**: the only contract a new data source must satisfy.
- **`TimeFDataset`**: the in-memory model a connector populates during `convert()`.
- **`TimeFWriter`**: serializes a populated `TimeFDataset` to disk. (Deferred — to be designed.)

---
