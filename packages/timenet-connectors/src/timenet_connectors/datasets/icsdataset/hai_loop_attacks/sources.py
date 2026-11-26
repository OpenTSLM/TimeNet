"""Where the examples table comes from: one Hugging Face dataset revision, pinned by commit and SHA-256.

These constants are the only place that names the table. To move the pin, change them here.
"""

EXAMPLES_REPO = "night-shift-data/hai-loop-attacks"
"""The Hugging Face dataset repo that publishes the table under CC BY-SA 4.0."""
EXAMPLES_REVISION = "531d912fc8651c8ef2068afe0906fa15705a449c"
"""The commit of :data:`EXAMPLES_REPO` that holds the table."""
EXAMPLES_FILENAME = "examples.csv.gz"
"""The table of loop examples: one row per example, no series values."""
EXAMPLES_SHA256 = "f11104defca497e2bd1429d84fd6c09308efbfd114211d4efa65a2152e356c6b"
"""SHA-256 of the table file as published, gzip bytes included."""
