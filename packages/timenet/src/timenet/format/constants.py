"""On-disk filenames and layout templates for the TimeF format."""

MANIFEST_FILE = "manifest.json"
SAMPLES_FILE = "samples.parquet"
ANNOTATIONS_FILE = "annotations.parquet"
INDEX_FILE = "time_series_index.parquet"

SHARD_DIR = "time_series"
SHARD_TEMPLATE = "time_series/shard-{:05d}.parquet"

TASKS_DIR = "tasks"
TASK_PART_TEMPLATE = "tasks/task={task_type}/part-0.parquet"

DEFAULT_SHARD_TARGET_BYTES = 128 * 2**20
DEFAULT_ROW_GROUP_TARGET_BYTES = 4 * 2**20
DEFAULT_CHUNK_MAX_BYTES = 1 * 2**20
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3
