"""Transparent NumPy preprocessing, logistic regression, metrics, and ranking."""
from dataclasses import dataclass
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1 + exp_x)
    return out


def logistic_loss_gradient(X, y, beta, l2=0.01):
    X, y, beta = np.asarray(X, float), np.asarray(y, float), np.asarray(beta, float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scores = X @ beta
        loss = np.mean(np.logaddexp(0, scores) - y * scores)
        loss += l2 / 2 * np.sum(beta[1:] ** 2)
        gradient = X.T @ (sigmoid(scores) - y) / len(y)
        gradient[1:] += l2 * beta[1:]
    return float(loss), gradient


def fit_logistic(X, y, learning_rate=.05, l2=.01, max_iter=10000,
                 tolerance=1e-7):
    """Batch gradient descent; the intercept is not regularized."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or len(X) != len(y) or not np.all(np.isfinite(X)):
        raise ValueError("X and y must have matching finite numeric values")
    beta = np.zeros(X.shape[1])
    losses = []
    for iteration in range(1, max_iter + 1):
        loss, gradient = logistic_loss_gradient(X, y, beta, l2)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("Nonfinite logistic loss or gradient")
        losses.append(loss)
        beta -= learning_rate * gradient
        if np.linalg.norm(gradient) < tolerance:
            return {"beta": beta, "losses": losses, "iterations": iteration,
                    "converged": True, "reason": "gradient_tolerance"}
    return {"beta": beta, "losses": losses, "iterations": max_iter,
            "converged": False, "reason": "iteration_limit"}


def predict_proba(X, beta):
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return sigmoid(np.asarray(X, float) @ np.asarray(beta, float))


def log_loss(y, probabilities):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(probabilities, float), 1e-15, 1 - 1e-15)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("y and probabilities must have equal nonzero length")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def brier_score(y, probabilities):
    return float(np.mean((np.asarray(probabilities) - np.asarray(y)) ** 2))


def confusion_metrics(y, probabilities, threshold=.5):
    y = np.asarray(y, int)
    predicted = np.asarray(probabilities) >= threshold
    if len(y) == 0 or len(y) != len(predicted):
        raise ValueError("y and probabilities must have equal nonzero length")
    tp = int(np.sum((y == 1) & predicted))
    tn = int(np.sum((y == 0) & ~predicted))
    fp = int(np.sum((y == 0) & predicted))
    fn = int(np.sum((y == 1) & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(y)}


def roc_curve(y, scores):
    """ROC points with tied scores grouped before cumulative counting."""
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must have equal nonzero length")
    positives, negatives = np.sum(y == 1), np.sum(y == 0)
    if positives == 0 or negatives == 0:
        return np.array([0., 1.]), np.array([0., 1.]), np.nan
    order = np.argsort(-scores, kind="mergesort")
    ys, ss = y[order], scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[distinct]
    fp = (distinct + 1) - tp
    tpr = np.r_[0., tp / positives, 1.]
    fpr = np.r_[0., fp / negatives, 1.]
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(tpr, fpr))
    return fpr, tpr, auc


def average_precision(y, scores):
    """Recall-increment weighted precision, with ties evaluated as blocks."""
    y, scores = np.asarray(y, int), np.asarray(scores, float)
    positives = int(np.sum(y == 1))
    if len(y) != len(scores) or len(y) == 0:
        raise ValueError("y and scores must have equal nonzero length")
    if positives == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    ys, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    precision = tp / (ends + 1)
    recall = tp / positives
    previous = np.r_[0., recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def classification_metrics(y, probabilities, threshold=.5):
    result = confusion_metrics(y, probabilities, threshold)
    _, _, auc = roc_curve(y, probabilities)
    result.update({"log_loss": log_loss(y, probabilities),
                   "brier_score": brier_score(y, probabilities),
                   "roc_auc": auc,
                   "average_precision": average_precision(y, probabilities),
                   "threshold": threshold})
    return result


def best_f1_threshold(y, probabilities):
    scores = np.unique(np.asarray(probabilities, float))
    candidates = [(classification_metrics(y, probabilities, value)["f1"], value)
                  for value in scores]
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


@dataclass
class TrainingPreprocessor:
    """Fit transformations on training rows, then reuse the fixed layout."""
    numeric: tuple
    categorical: tuple
    include_page_values: bool = False

    def fit(self, frame):
        numeric = list(self.numeric)
        if self.include_page_values and "PageValues" not in numeric:
            numeric.append("PageValues")
        self.numeric_features = numeric
        self.medians = {}
        self.means = {}
        self.scales = {}
        self.categories = {}
        self.groups = {}
        for column in numeric:
            values = np.asarray(frame[column], dtype=float)
            finite = values[np.isfinite(values)]
            self.medians[column] = float(np.median(finite))
            transformed = np.log1p(np.maximum(finite, 0)) if column in {
                "Administrative", "Administrative_Duration", "Informational",
                "Informational_Duration", "ProductRelated", "ProductRelated_Duration"
            } else finite
            self.means[column] = float(np.mean(transformed))
            scale = float(np.std(transformed))
            self.scales[column] = scale if scale > 0 else 1.0
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            levels = sorted(values.unique().tolist())
            self.categories[column] = levels
        self.feature_names = ["intercept"]
        for column in numeric:
            self.feature_names.append(column)
        for column in self.categorical:
            levels = self.categories[column][1:]
            start = len(self.feature_names)
            self.feature_names.extend(f"{column}={level}" for level in levels)
            self.groups[column] = list(range(start, len(self.feature_names)))
        return self

    def transform(self, frame):
        if not hasattr(self, "feature_names"):
            raise RuntimeError("Preprocessor must be fit on training data first")
        columns = [np.ones(len(frame))]
        for column in self.numeric_features:
            values = pd_numeric(frame[column], self.medians[column])
            if column in {"Administrative", "Administrative_Duration", "Informational",
                          "Informational_Duration", "ProductRelated", "ProductRelated_Duration"}:
                values = np.log1p(np.maximum(values, 0))
            columns.append((values - self.means[column]) / self.scales[column])
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            reference = self.categories[column][0]
            columns.extend((values == level).to_numpy(float) for level in self.categories[column][1:])
        return np.column_stack(columns)


def pd_numeric(values, median):
    values = np.asarray(values, dtype=float)
    values = values.copy()
    values[~np.isfinite(values)] = median
    return values
