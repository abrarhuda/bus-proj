# Understanding Online Purchase Conversion

## 1. The question

This project investigates a practical but deliberately limited question:

> **Which characteristics of an online browsing session are most strongly
> associated with whether that session ends in a purchase?**

The analysis is designed to distinguish three ideas that are often conflated:

1. **Association:** do purchase sessions look different from non-purchase
   sessions?
2. **Predictive contribution:** which available variables improve a model's
   ability to rank likely purchasers?
3. **Causation:** can the data justify saying that changing a variable would
   cause more purchases?

The first two questions can be studied here. The third cannot be established
from this retrospective, observational dataset. `Revenue=True` means that a
session ended in a purchase; it is not a dollar-revenue measure.

## 2. The data

The source is the **UCI Online Shoppers Purchasing Intention Dataset**:

- Sakar, C. & Kastro, Y. (2018), UCI Machine Learning Repository, dataset 468
- [Dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- [DOI](https://doi.org/10.24432/C5F88Q)
- Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Each row summarizes one browsing session. The raw data contains 12,330
sessions, 1,908 purchase sessions, and an observed purchase rate of 15.47%.
The low positive rate makes accuracy an inadequate primary measure, so the
analysis emphasizes average precision, ROC AUC, log loss, calibration, and
threshold-aware metrics.

Download the data with:

```bash
python scripts/download_data.py
```

The downloader records retrieval metadata and does not overwrite an existing
raw file. Manual-download instructions and provenance are in
[`data/README.md`](data/README.md).

## 3. How the analysis was performed

The notebooks are intentionally ordered as an analysis story. They can be
reproduced from a clean kernel with:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_notebooks.py
python -m unittest discover -s tests
```

All random operations use seed 42. The runner executes every numbered
notebook in order and stops on the first error.

### Step 1: audit and prepare the sessions

[`01_data_audit_and_cleaning.ipynb`](notebooks/01_data_audit_and_cleaning.ipynb)
checks schema, missing values, boolean encodings, numeric bounds, duplicates,
and month/category representations. Cleaning decisions are written to
[`cleaning_decisions.csv`](reports/tables/cleaning_decisions.csv), rather than
being implicit in notebook code.

The audit found no missing values, invalid boolean values, or invalid
measurement bounds. There are 125 exact duplicate rows; they are retained
because they may represent repeated sessions and removing them without
session identifiers would be an unsupported assumption. A synthetic `row_id`
is added only for traceability.

The cleaned data is split once, stratified by `Revenue`, and saved to
[`split_assignments.csv`](data/processed/split_assignments.csv):

- Train: 7,397 sessions
- Validation: 2,465 sessions
- Test: 2,468 sessions

The test set remains untouched until the final model evaluation.

### Step 2: describe conversion patterns with uncertainty

[`02_conversion_patterns_and_uncertainty.ipynb`](notebooks/02_conversion_patterns_and_uncertainty.ipynb)
uses training data for descriptive comparisons. It examines visitor type,
month, traffic source, weekend status, product engagement, and numeric
relationships with the outcome.

The notebook reports Wilson confidence intervals and a seeded 2,000-replicate
bootstrap comparison for new versus returning visitors. It also fits
training-derived engagement bins so that cut points are not learned from
validation or test observations. These results describe differences in this
sample; they are not evidence that a visitor segment causes conversion.

### Step 3: build and evaluate a transparent model

[`03_purchase_prediction_and_factor_ranking.ipynb`](notebooks/03_purchase_prediction_and_factor_ranking.ipynb)
implements a small NumPy logistic regression with train-fitted
preprocessing. Numeric variables are transformed and standardized from the
training data only; categorical vocabularies are also fixed from training.
Numerical gradient and loss checks are covered by
[`test_statistics_and_modeling.py`](tests/test_statistics_and_modeling.py).

The primary model excludes:

- `Revenue`, the target
- `row_id`, a synthetic identifier
- `PageValues`, because it is transaction-linked and its availability before
  purchase is uncertain

Hyperparameters and the F1 threshold are selected on validation data. The
locked model is then evaluated once on the test data. On that untouched test
split, the primary model achieved:

- ROC AUC: **0.722**
- Average precision: **0.291**
- Log loss: **0.393**
- F1 at the validation-selected threshold of 0.1717: **0.372**

For context, the training-prevalence baseline achieved average precision
0.155 and log loss 0.432. The primary model therefore ranks sessions better
than the baseline, but it is not presented as a production-ready targeting
system.

The model also runs grouped permutation importance on validation data. This
asks how much ranking performance changes when a feature group is disrupted;
it does not estimate a causal effect and it is specific to this fitted model
and sample.

### Step 4: synthesize the evidence

[`04_business_findings_and_recommendations.ipynb`](notebooks/04_business_findings_and_recommendations.ipynb)
generates the evidence-based executive report:
[`reports/executive_summary.md`](reports/executive_summary.md).

[`05_in_depth_results_analysis.ipynb`](notebooks/05_in_depth_results_analysis.ipynb)
then brings the descriptive comparisons, outcome medians, model performance,
permutation ranking, and limitations together in one direct answer to the
research question.

## 4. What the results say

The strongest validation grouped-permutation contributors in the
PageValues-excluded model were:

1. **Month**
2. **ExitRates**
3. **ProductRelated**
4. **ProductRelated_Duration**

The descriptive and model results point toward a broad interpretation:
conversion is associated with when a session occurs, how far a visitor
progresses through product content, and how the session's navigation
behavior unfolds. The ranking does not mean these variables independently
cause purchases. Correlated browsing measures can substitute for one another,
and permutation importance can change with the fitted model and sample.

`PageValues` was evaluated separately as a sensitivity analysis. Including it
raised test average precision to 0.588 and reduced log loss to 0.329, but
that result is explicitly **not** treated as deployable evidence. A variable
that is calculated from transaction-linked behavior or whose timing is
uncertain can create leakage or make a pre-purchase use case impossible.

## 5. Recommendations and next experiments

The findings support hypotheses for controlled follow-up work, not promises
of business impact:

- Test clearer product-information and comparison experiences for sessions
  showing deeper product engagement.
- Investigate whether exit-rate patterns identify navigation friction, then
  test targeted usability or checkout interventions.
- Account for seasonal or monthly context when designing experiments and
  interpreting conversion changes.
- Evaluate new-visitor onboarding separately from returning-visitor
  experiences because the observational conversion patterns differ.
- Before deployment, confirm feature availability at scoring time, monitor
  calibration and subgroup performance, and compare any intervention against
  a randomized control group.

The next business decision should therefore be an experiment-design decision,
not an automatic targeting rule based on the feature ranking.

## 6. Limitations

This is one historical, summarized-session dataset. It does not include
prices, costs, customer identifiers, user histories, randomized treatments,
or a confirmed live feature-availability contract. The data has class
imbalance, correlated predictors, potentially small segments, and retained
duplicate rows. Session duration and final-page measures may also be observed
only after substantial browsing has already occurred.

Accordingly, the results support descriptive and predictive statements about
this sample. They do not establish causality, return on investment, or
generalization to another site, period, population, or production system.

## 7. Repository and generated outputs

```text
data/       raw and processed data, split assignments, and provenance
notebooks/  ordered audit, descriptive, modeling, synthesis, and deep-analysis notebooks
reports/    generated tables, 22 Matplotlib figures, and executive summary
scripts/    dataset download and ordered notebook execution
src/        reusable data, statistics, modeling, and plotting helpers
tests/      numerical, preprocessing, metric, and split tests
```

The visual outputs are in [`reports/figures/`](reports/figures/), including
data-quality and split checks, conversion comparisons, engagement
distributions, correlation views, ROC/precision-recall/calibration plots,
coefficient and confusion-matrix views, grouped permutation importance,
model comparisons, and the in-depth evidence dashboard.
