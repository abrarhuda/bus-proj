"""Data paths, explicit boolean normalization, and reproducible stratified splits."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType",
    "Weekend", "Revenue",
]


def normalize_boolean(values):
    """Return booleans, raising on missing or unrecognized representations."""
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "yes": True, "no": False}
    result = []
    for value in values:
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"Invalid boolean representation: {value!r}")
        result.append(mapping[key])
    return np.asarray(result, dtype=bool)


def stratified_split(y, seed=42, train_fraction=.6, validation_fraction=.2):
    """Return a row-index -> split mapping with complete, disjoint coverage."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    labels = np.empty(len(y), dtype=object)
    labels[:] = "test"
    for value in np.unique(y):
        indices = np.flatnonzero(y == value)
        rng.shuffle(indices)
        n_train = int(np.floor(len(indices) * train_fraction))
        n_valid = int(np.floor(len(indices) * validation_fraction))
        labels[indices[:n_train]] = "train"
        labels[indices[n_train:n_train + n_valid]] = "validation"
    return labels
