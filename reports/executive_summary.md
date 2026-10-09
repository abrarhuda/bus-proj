# Executive summary

## Scope and question

This project analyzes **12,330 summarized online-shopping sessions** from the UCI Online Shoppers Purchasing Intention Dataset. There were **1,908 purchase sessions**, an observed conversion rate of **15.47%**. The question is which session characteristics are most strongly associated with purchase completion. `Revenue=True` means a purchase occurrence, not dollar revenue.

## Data quality and design

The audit found 0 missing values, 0 invalid boolean values, 0 invalid count/duration measurements, and 125 exact duplicate rows. Exact duplicates were retained because identical session summaries can belong to different users. The seed-42 split contains 7,397 training, 2,465 validation, and 2,468 test rows; validation and test outcomes were reserved for their documented roles.

See the [`data-quality audit`](figures/00_data_quality_audit.png) and
[`split-balance chart`](figures/00_split_balance.png) for visual checks of
the source quality and partition composition.

## Findings

1. **Visitor type shows a descriptive difference.** New visitors converted at 25.02% (251/1,003; 95% Wilson interval 22.44%-27.80%), versus 13.98% for returning visitors (886/6,338; interval 13.15%-14.85%). The bootstrap difference was **11.05 percentage points**, with a 95% interval of **8.34-13.70 points**. This is association, not a causal visitor-status effect. See [`visitor_conversion_training.csv`](tables/visitor_conversion_training.csv) and [`02_visitor_conversion_training.png`](figures/02_visitor_conversion_training.png).

2. **The primary model provides ranking signal beyond prevalence.** The PageValues-excluded model achieved test average precision **0.291** and log loss **0.393**, versus **0.155** and **0.432** for the training-prevalence baseline. At the validation-selected F1 threshold of **0.1717**, test F1 was **0.372**. This is retrospective performance, not deployment evidence. See [`model_metrics.csv`](tables/model_metrics.csv) and [`09_test_roc_curve.png`](figures/09_test_roc_curve.png).

   The [`test model comparison`](figures/13_test_model_comparison.png) shows
   the locked primary and baseline metrics side by side.

3. **The strongest validation permutation groups were Month, ExitRates, ProductRelated, and ProductRelated_Duration.** Their mean average-precision decreases were 0.035, 0.024, 0.017, and 0.016. These are model-and-sample contributions; correlated predictors can exchange or dilute importance. See [`validation_grouped_permutation_importance.csv`](tables/validation_grouped_permutation_importance.csv) and [`12_validation_grouped_permutation_importance.png`](figures/12_validation_grouped_permutation_importance.png).

   The [`business evidence dashboard`](figures/14_business_evidence_dashboard.png)
   combines the leading validation factors with visitor conversion rates.

4. **PageValues is a sensitivity result, not a recommendation.** Including PageValues produced test average precision **0.588** and log loss **0.329**, better than the primary model. Because PageValues is transaction-linked and its timing is not established, this may reflect information unavailable before purchase completion. It is not evidence for live targeting. See [`pagevalues_sensitivity_metrics.csv`](tables/pagevalues_sensitivity_metrics.csv).

## Business recommendations as testable hypotheses

| Observed evidence | Possible action | Proposed experiment | Success metric | Limitation |
|---|---|---|---|---|
| New visitors converted at 25.0% versus 14.0% for returning visitors. | Test onboarding or messaging tailored to new visitors with randomized assignment. | Conversion rate; bounce or abandonment guardrails. | Historical visitor type is observational and does not establish causation. |
| ProductRelated and ProductRelated_Duration ranked highly in validation permutation importance. | Test product-information layout, navigation, or discovery improvements. | Conversion rate plus usability guardrails. | Session summaries are retrospective; association is not causation. |
| Month and ExitRates ranked highly in this fitted model. | Test seasonal content or checkout-friction hypotheses by predeclared segment. | Conversion rate and segment guardrails. | Month is a time label and ExitRates is a session summary, not a causal intervention. |

These are experiment ideas, not promised conversion gains. Randomized assignment and predeclared success criteria are needed.

## Limitations

- One historical dataset with summarized sessions, not event-level journeys or user histories.
- No prices, costs, profit, customer lifetime value, or ROI fields.
- Page counts, durations, ExitRates, and other final-session summaries may not be available when a live intervention must act.
- The target is imbalanced; accuracy alone is insufficient.
- Small segments, correlated predictors, and exploratory comparisons limit stability.
- The model was not externally validated and is not a live prediction system.
- Associations and permutation contributions do not establish causation or business impact.
