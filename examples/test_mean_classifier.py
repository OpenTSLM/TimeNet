"""End-to-end TimeNet example: build a dataset, load it, and train a classifier.

The task is deliberately simple: each record in ``timenet/test-mean`` is one noisy signal, labeled by
whether its mean is above or below zero. The script runs the whole loop top to bottom: build the
dataset from its connector into the local registry, load it back through the SDK, turn the records into
an ``(X, y)`` training pair, and train a scikit-learn classifier.

scikit-learn is not a TimeNet dependency. Run it with uv, which pulls scikit-learn in for this one
command without adding it to your environment. From the repo root::

    uv run --with scikit-learn examples/test_mean_classifier.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from timenet.client import TimeNet
import timenet_connectors


# Build the connector's dataset into the local registry (idempotent), then load it back through the SDK.
timenet_connectors.build("timenet/test-mean")
dataset = TimeNet().load("timenet/test-mean")

# A quick tour of what we loaded: identity, counts, specs, and a record preview.
dataset.describe()
# timenet/test-mean @ 1.0.0
#   name     Test Mean
#   license  CC-BY-4.0
#   domains  general
#   tags     demo, synthetic, classification
#
# counts
#   records      1000
#   series       signal=1000
#   annotations  0
#   tasks        classification=1000
#
# specs
#   spec    name    value          rate   dtype
#   signal  Signal  dimensionless  hertz  float
#
# records (first 5 of 1000)
#   record_id   view  signals  length  tasks  annotations
#   record-0    full  1         64      1      0
#   record-1    full  1         64      1      0
#   record-10   full  1         64      1      0
#   record-100  full  1         64      1      0
#   record-101  full  1         64      1      0

# Pair each record's values with its label. Materialization is deferred by default (Arrow); ask for
# output="numpy" here since scikit-learn needs it. test-mean has one task type, so the task is inferred.
x, y = dataset.to_features_and_targets(output="numpy")
# x: np.ndarray (1000, 64) float32   -- one 64-point signal per row
# y: np.ndarray (1000,) object       -- "above_zero" / "below_zero" per row
# (the default output="arrow" instead returns X as a FixedSizeListArray<float>[64] and y as a string array)

# Train a classifier on a held-out split and report accuracy.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
accuracy = accuracy_score(y_test, model.predict(x_test))

print(f"loaded {len(dataset.records)} records")  # loaded 1000 records
print(f"test accuracy: {accuracy:.3f}")  # test accuracy: 1.000
