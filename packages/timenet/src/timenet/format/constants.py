"""On-disk filenames and layout templates for the TimeF format."""

MANIFEST_FILE = "manifest.json"
SAMPLES_FILE = "samples.parquet"
ANNOTATIONS_FILE = "annotations.parquet"
INDEX_FILE = "time_series_index.parquet"

SHARD_DIR = "time_series"
SHARD_TEMPLATE = "time_series/shard-{:08d}.parquet"

TASKS_DIR = "tasks"
TASK_PART_TEMPLATE = "tasks/task={task_type}/part-{:08d}.parquet"

# Control-plane tables, each sharded into numbered parts under their own directory. Eight digits keeps
# the parts lexically sortable up to 100M shards, far past any table's realistic size.
SAMPLES_TEMPLATE = "samples/part-{:08d}.parquet"
ANNOTATIONS_TEMPLATE = "annotations/part-{:08d}.parquet"
INDEX_TEMPLATE = "time_series_index/part-{:08d}.parquet"

# The column each prunable control table is sorted by at write time and pruned/bisected on at read
# time. Naming the contract keeps the writer's sort and the reader's lookup from drifting.
INDEX_SORT_KEY = "sample_id"
ANNOTATIONS_SORT_KEY = "id"

DEFAULT_SHARD_TARGET_BYTES = 128 * 2**20
# Target size for one control-table part, measured on the in-memory Arrow table (not on disk). Kept
# separate from the values-plane target so the two can be tuned independently.
DEFAULT_CONTROL_SHARD_TARGET_BYTES = 128 * 2**20
DEFAULT_ROW_GROUP_TARGET_BYTES = 4 * 2**20
DEFAULT_CHUNK_MAX_BYTES = 1 * 2**20
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3
