# timenet-connectors

Dataset-specific connectors for [TimeNet](../timenet). A connector holds the logic to fetch a
dataset's raw artifacts and convert them into the shared TimeF format.

This package currently ships a placeholder `BaseConnector`. Concrete dataset integrations (and the
Ray Data processor) land here.

## Install

```bash
pip install timenet-connectors
```

## Usage

```python
from timenet_connectors import BaseConnector
```
