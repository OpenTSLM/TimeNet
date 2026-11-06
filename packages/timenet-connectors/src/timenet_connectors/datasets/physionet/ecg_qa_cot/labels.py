"""The fixed values of ECG-QA CoT: the PTB-XL findings that the clinical context states, and the kept keys."""

# The PTB-XL signal-quality annotations, each with the phrase that the clinical context states if a
# record has it.
QUALITY_ISSUES = (
    ("baseline_drift", "baseline drift noted"),
    ("static_noise", "static noise present"),
    ("burst_noise", "burst noise present"),
    ("electrodes_problems", "electrode artifacts present"),
)
# The PTB-XL findings outside signal quality, each with the phrase that the clinical context states
# if a record has it.
OTHER_FINDINGS = (
    ("extra_beats", "extra beats detected during recording"),
    ("pacemaker", "pacemaker present"),
)
# The imported PTB-XL annotations that a record keeps. The connector clears every other one.
KEPT_ANNOTATIONS = frozenset({"subject_ids", "sex", "age"})
# The sexes that an imported PTB-XL record can have. The connector refuses any other value.
PTBXL_SEXES = frozenset({"male", "female"})
