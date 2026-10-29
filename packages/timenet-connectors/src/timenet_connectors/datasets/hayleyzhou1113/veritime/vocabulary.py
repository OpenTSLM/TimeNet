"""The class names of VeriTime's classification tasks, in the order their questions offer them."""

from collections.abc import Mapping

from timenet_connectors.datasets.hayleyzhou1113.veritime.release import SourceTask


CLASS_VOCABULARIES: Mapping[SourceTask, tuple[str, ...]] = {
    # The authors' extract_label.py takes Yes or No from the ChatTS answer.
    SourceTask.ANOMALY_DETECTION: ("Yes", "No"),
    # Each knowledge question lists its classes as options a), b), and so on, in this order.
    SourceTask.CTU: ("Desktop", "Laptop"),
    SourceTask.ECG: ("normal sinus rhythm", "atrial fibrillation", "other cardiac rhythms", "noise"),
    SourceTask.EMG: ("Healthy", "Myopathy", "Neuropathy"),
    SourceTask.RCW: ("Right Whale Present", "No Right Whale"),
}
"""Each classification task's labels, as the release spells them. Scenario attribution offers its own
options in each question and inferential calculation answers a count, so neither has one."""
