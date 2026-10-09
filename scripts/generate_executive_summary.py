"""Generate the business-facing report from saved analysis artifacts."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT

TABLES = ROOT / "reports" / "tables"
REPORT = ROOT / "reports" / "executive_summary.md"


def metric(metrics, model, split, threshold=None):
    result = metrics[(metrics.model == model) & (metrics.split == split)]
    if threshold is not None:
        result = result[result.threshold.round(6) == round(threshold, 6)]
    if result.empty:
        raise ValueError(f"Missing metrics row for {model}/{split}")
    return result.iloc[0]


def generate():
    required = [
        "split_summary.csv", "visitor_conversion_training.csv",
        "visitor_bootstrap_difference.csv", "model_metrics.csv",
        "selected_threshold.csv", "validation_grouped_permutation_importance.csv",
        "pagevalues_sensitivity_metrics.csv",
    ]
    missing = [name for name in required if not (TABLES / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run notebooks 01-03 first; missing: {missing}")

    splits = pd.read_csv(TABLES / "split_summary.csv")
    bootstrap = pd.read_csv(TABLES / "visitor_bootstrap_difference.csv").iloc[0]
    metrics = pd.read_csv(TABLES / "model_metrics.csv")
    threshold = pd.read_csv(TABLES / "selected_threshold.csv").iloc[0]
    importance = pd.read_csv(TABLES / "validation_grouped_permutation_importance.csv")
    sensitivity = pd.read_csv(TABLES / "pagevalues_sensitivity_metrics.csv")
    primary = metric(metrics, "primary_logistic", "test", threshold.threshold)
    baseline = metric(metrics, "training_prevalence", "test")
    pagevalues = sensitivity[sensitivity.split == "test"].iloc[0]
    top = importance.head(4)
    sessions = int(splits.sessions.sum())
    purchases = int(splits.purchases.sum())
    recommendations = [
        "| New visitors converted at 25.0% versus 14.0% for returning visitors. | Test onboarding or messaging tailored to new visitors with randomized assignment. | Conversion rate; bounce or abandonment guardrails. | Historical visitor type is observational and does not establish causation. |",
        "| ProductRelated and ProductRelated_Duration ranked highly in validation permutation importance. | Test product-information layout, navigation, or discovery improvements. | Conversion rate plus usability guardrails. | Session summaries are retrospective; association is not causation. |",
        "| Month and ExitRates ranked highly in this fitted model. | Test seasonal content or checkout-friction hypotheses by predeclared segment. | Conversion rate and segment guardrails. | Month is a time label and ExitRates is a session summary, not a causal intervention. |",
    ]
    report = f"""# Executive summary

## Scope and question

This project analyzes **{sessions:,} summarized online-shopping sessions** from the UCI Online Shoppers Purchasing Intention Dataset. There were **{purchases:,} purchase sessions**, an observed conversion rate of **{purchases / sessions:.2%}**. The question is which session characteristics are most strongly associated with purchase completion. `Revenue=True` means a purchase occurrence, not dollar revenue.

## Data quality and design

The audit found 0 missing values, 0 invalid boolean values, 0 invalid count/duration measurements, and 125 exact duplicate rows. Exact duplicates were retained because identical session summaries can belong to different users. The seed-42 split contains 7,397 training, 2,465 validation, and 2,468 test rows; validation and test outcomes were reserved for their documented roles.

See the [`data-quality audit`](figures/00_data_quality_audit.png) and
[`split-balance chart`](figures/00_split_balance.png) for visual checks of
the source quality and partition composition.

## Findings

1. **Visitor type shows a descriptive difference.** New visitors converted at 25.02% (251/1,003; 95% Wilson interval 22.44%-27.80%), versus 13.98% for returning visitors (886/6,338; interval 13.15%-14.85%). The bootstrap difference was **{bootstrap.difference_percentage_points:.2f} percentage points**, with a 95% interval of **{bootstrap.bootstrap_ci_low_percentage_points:.2f}-{bootstrap.bootstrap_ci_high_percentage_points:.2f} points**. This is association, not a causal visitor-status effect. See [`visitor_conversion_training.csv`](tables/visitor_conversion_training.csv) and [`02_visitor_conversion_training.png`](figures/02_visitor_conversion_training.png).

2. **The primary model provides ranking signal beyond prevalence.** The PageValues-excluded model achieved test average precision **{primary.average_precision:.3f}** and log loss **{primary.log_loss:.3f}**, versus **{baseline.average_precision:.3f}** and **{baseline.log_loss:.3f}** for the training-prevalence baseline. At the validation-selected F1 threshold of **{threshold.threshold:.4f}**, test F1 was **{primary.f1:.3f}**. This is retrospective performance, not deployment evidence. See [`model_metrics.csv`](tables/model_metrics.csv) and [`09_test_roc_curve.png`](figures/09_test_roc_curve.png).

   The [`test model comparison`](figures/13_test_model_comparison.png) shows
   the locked primary and baseline metrics side by side.

3. **The strongest validation permutation groups were {top.iloc[0].feature}, {top.iloc[1].feature}, {top.iloc[2].feature}, and {top.iloc[3].feature}.** Their mean average-precision decreases were {top.iloc[0].mean_ap_decrease:.3f}, {top.iloc[1].mean_ap_decrease:.3f}, {top.iloc[2].mean_ap_decrease:.3f}, and {top.iloc[3].mean_ap_decrease:.3f}. These are model-and-sample contributions; correlated predictors can exchange or dilute importance. See [`validation_grouped_permutation_importance.csv`](tables/validation_grouped_permutation_importance.csv) and [`12_validation_grouped_permutation_importance.png`](figures/12_validation_grouped_permutation_importance.png).

   The [`business evidence dashboard`](figures/14_business_evidence_dashboard.png)
   combines the leading validation factors with visitor conversion rates.

4. **PageValues is a sensitivity result, not a recommendation.** Including PageValues produced test average precision **{pagevalues.average_precision:.3f}** and log loss **{pagevalues.log_loss:.3f}**, better than the primary model. Because PageValues is transaction-linked and its timing is not established, this may reflect information unavailable before purchase completion. It is not evidence for live targeting. See [`pagevalues_sensitivity_metrics.csv`](tables/pagevalues_sensitivity_metrics.csv).

## Business recommendations as testable hypotheses

| Observed evidence | Possible action | Proposed experiment | Success metric | Limitation |
|---|---|---|---|---|
{chr(10).join(recommendations)}

These are experiment ideas, not promised conversion gains. Randomized assignment and predeclared success criteria are needed.

## Limitations

- One historical dataset with summarized sessions, not event-level journeys or user histories.
- No prices, costs, profit, customer lifetime value, or ROI fields.
- Page counts, durations, ExitRates, and other final-session summaries may not be available when a live intervention must act.
- The target is imbalanced; accuracy alone is insufficient.
- Small segments, correlated predictors, and exploratory comparisons limit stability.
- The model was not externally validated and is not a live prediction system.
- Associations and permutation contributions do not establish causation or business impact.
"""
    REPORT.write_text(report)
    return report


if __name__ == "__main__":
    print(generate())
