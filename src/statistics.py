"""Small NumPy-based descriptive and uncertainty calculations."""
import numpy as np


def wilson_interval(successes, total, z=1.96):
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must describe a valid count")
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_difference(a, b, repetitions=2000, seed=42):
    """Bootstrap mean(a)-mean(b), resampling within each group."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both groups require observations")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(
            b, len(b), replace=True).mean()
    return float(a.mean() - b.mean()), tuple(np.quantile(draws, [.025, .975]))
