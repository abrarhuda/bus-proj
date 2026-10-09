# Copilot implementation plan: Online Shopper Purchase Analysis

## Instructions to GitHub Copilot

Build a complete, reproducible analysis repository from this specification. The owner is a first-year UNC–Chapel Hill student studying business and statistics. Write readable Python and explanatory Jupyter notebooks the owner can understand, run, and discuss in an interview. Implement the repository, rather than returning only suggestions. Work in the phases below and verify each phase before proceeding. Do not fabricate results, charts, statistical significance, or completed-work claims. If the dataset cannot be downloaded, implement an actionable manual-download fallback and clearly mark the analysis as not yet executed.

**Research question:** What factors most strongly predict whether an online visitor makes a purchase?

**Project title:** Understanding Online Purchase Conversion: Statistical Analysis of Shopper Behavior.

**Business purpose:** Identify session characteristics associated with purchase completion and translate the evidence into marketing and website experiments. Demonstrate the data organization, analysis, reporting, and communication skills relevant to UNC Kenan-Flagler Undergraduate Business Program support work. This is a public e-commerce portfolio project; do not describe it as an analysis of UNC students or as university employment.

### Technology and scope

- Use pandas for ingestion, cleaning, group summaries, encoding, and CSV exports.
- Use NumPy for numerical calculations, random sampling, bootstrap intervals, logistic regression, and evaluation metrics.
- Use Matplotlib for every chart. Do not add seaborn or Plotly.
- Use Jupyter notebooks with Markdown explanations, working code, displayed tables, and actual outputs.
- Standard-library modules are permitted. Add `jupyter`, `nbconvert`, and `ipykernel` as notebook tooling, alongside pandas, NumPy, and Matplotlib. Use `nbformat` only if programmatically creating notebooks.
- Do not require scikit-learn, SciPy, statsmodels, TensorFlow, R, Tableau, or a web application. The goal is a statistical project using the listed Python skills, with one transparent predictive model.
- Avoid excessive abstraction. Put reusable functions in a small `src/` package; keep questions, interpretations, and analysis steps visible in notebooks.
- Use random seed 42. Record actual dependency versions after a successful installation/run; do not guess version pins.

## 1. Dataset and interpretation

Use the **UCI Online Shoppers Purchasing Intention Dataset**, dataset ID 468.

- Dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Dataset DOI: https://doi.org/10.24432/C5F88Q
- Download archive: https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip
- Expected CSV: `online_shoppers_intention.csv`.
- Attribution: Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. CC BY 4.0.

UCI describes 12,330 sessions, 17 predictors, and the binary target `Revenue`; 1,908 sessions ended in a purchase and 10,422 did not. Treat these as source expectations and compare them with the downloaded file. Each row represents a session, and UCI states the sessions belong to different users over a one-year period. There are no identifiers or event-level histories available for tracking repeat behavior.

`Revenue` means purchase occurrence, not the dollar value of revenue. Define conversion as `Revenue=True` and conversion rate as purchases / sessions. Do not calculate sales dollars, profit, customer lifetime value, or ROI from this dataset.

### Expected schema

| Columns | Treatment | Meaning |
| --- | --- | --- |
| Administrative, Informational, ProductRelated | Nonnegative counts | Number of pages of each type visited |
| Administrative_Duration, Informational_Duration, ProductRelated_Duration | Nonnegative numeric | Time spent in each page category; verify source units before labeling axes |
| BounceRates, ExitRates | Numeric rates | Aggregate analytics measures associated with visited pages |
| PageValues | Numeric; special handling | Page-value metric tied to transaction value |
| SpecialDay | Numeric proximity score | Closeness to a special shopping day, not an observed holiday indicator |
| Month | Categorical | Month labels; normalize documented spellings and order only observed months |
| OperatingSystems, Browser, Region, TrafficType | Categorical codes | Codes, not quantities or ordered measures |
| VisitorType | Categorical | Visitor classification, such as new or returning |
| Weekend | Boolean | Whether the visit occurred on a weekend |
| Revenue | Boolean target | Whether the session ended with a purchase |

Verify names, types, allowed values, and count expectations from the actual CSV. Do not assume all 12 months are represented. Do not invent names for traffic channels, browsers, regions, or operating systems from their codes.

### Prediction timing and leakage

The primary model is a **retrospective session-level classification analysis**, based on session summaries. It is not a validated live prediction system. Total durations, exit rates, and final page counts may not be available before a visitor finishes a session. Explain this timing limitation in the README and final report.

Exclude `PageValues` from the primary model because its transaction-linked definition creates a potential target-leakage concern. The provided documentation does not establish exactly when its values were computed, so do not claim leakage has been conclusively proven. Train a separate sensitivity model including it and label that model clearly. If performance improves, discuss the difference without treating it as evidence of a deployable model.

An optional arrival-information model can use only Month, Weekend, VisitorType, OperatingSystems, Browser, Region, TrafficType, and SpecialDay, subject to checking their availability. Keep this extension separate from the required scope.

## 2. Repository structure

```text
online-shopper-conversion-analysis/
  README.md
  requirements.txt
  .gitignore
  data/
    README.md
    raw/online_shoppers_intention.csv
    processed/clean_sessions.csv
    processed/split_assignments.csv
  notebooks/
    01_data_audit_and_cleaning.ipynb
    02_conversion_patterns_and_uncertainty.ipynb
    03_purchase_prediction_and_factor_ranking.ipynb
    04_business_findings_and_recommendations.ipynb
  src/
    __init__.py
    data.py
    statistics.py
    modeling.py
    plotting.py
  scripts/
    download_data.py
    run_notebooks.py
  reports/
    executive_summary.md
    figures/
    tables/
  tests/
    test_statistics_and_modeling.py
```

Commit the small public raw CSV with attribution if it can be obtained, so readers can reproduce the work offline. Preserve it unchanged. Ignore virtual environments, notebook checkpoints, Python caches, and temporary downloads. Keep useful executed notebook outputs and report figures. Do not create empty artifacts that imply the analysis has run.

`download_data.py` must use the standard library, a timeout, error handling, and extraction of only the expected CSV. Do not blindly extract ZIP paths. Avoid overwriting an existing raw CSV unless explicitly requested. Record retrieval date, source, SHA-256 checksum, and any deviations from UCI expectations in `data/README.md`. Give a manual download instruction if access fails.

Use pathlib paths resolved from the repository root. Notebooks must run from either the repository root or the notebooks directory without machine-specific paths.

## 3. Notebook 01 — Audit, cleaning, and split setup

Explain the unit of analysis and target, then:

1. Load the original CSV. Show shape, column types, sample rows, target counts, and descriptive statistics.
2. Build a data-quality table: missing values, unexpected categories, invalid booleans, negative counts/durations, nonfinite values, and exact duplicate rows. UCI reports no missing values, but verify this independently.
3. Normalize Boolean representations explicitly. Do not use `astype(bool)` on strings because the string `"False"` would become True. Validate target labels; quarantine rows with missing or invalid targets and report their count rather than imputing the outcome.
4. Convert categorical code columns to categorical/string representations. Preserve documented month labels with a clear mapping where needed. Report all alterations.
5. Do not automatically drop exact duplicates: different users can have identical summarized sessions. Count them, retain them by default, and document this choice. An optional sensitivity analysis may compare deduplicated results, but it must not silently replace the primary sample.
6. Retain plausible zeros and long sessions. Explore skew with `log1p` displays instead of removing outliers without evidence. A rate above 0.2 is a review flag against the observed dataset pattern, not automatically invalid: general valid rate bounds are 0 to 1. Treat impossible measurements explicitly and report the handling rule.
7. Assign a synthetic row ID for reproducibility, not as a predictive feature. Save cleaned data and an audit log table.
8. Create a reproducible stratified 60% training / 20% validation / 20% test split using NumPy, within each target class. Save row assignments and assert the partitions are disjoint, cover all eligible rows, and each contain both classes.

Audit the full dataset for structure and quality, but explore outcome relationships on the training set only. Reserve validation for tuning and test for final evaluation. Define these boundaries before outcome exploration.

## 4. Notebook 02 — Statistical relationships and uncertainty

Start with overall conversion rate and class balance. Describe full-sample prevalence as a descriptive fact; use training rows for the following factor analyses.

Answer these supporting questions:

- How do new and returning visitors differ in conversion rate?
- How do conversion rates vary across months, weekend status, and traffic-type codes?
- How do product-page counts and product-page duration differ between purchasing and non-purchasing sessions?
- How do bounce/exit measures relate to conversion?
- Are relationships consistent across segments, or do small samples make comparisons unreliable?

Required calculations:

- Group tables with sample size, purchase count, conversion rate, and 95% Wilson confidence intervals, implemented with NumPy. Use 1.96 as the conventional 95% normal approximation constant.
- New-minus-returning visitor conversion difference in percentage points, with a 2,000-replicate within-group bootstrap interval. Keep the RNG seeded. Explain that intervals express sampling uncertainty, not causal effects.
- Numeric comparisons by target using medians and interquartile ranges. Add point-biserial correlations where finite variance exists, with an explicit warning that linear correlation can miss nonlinear relationships.
- For counts/durations, build training-derived quantile bins with pandas, handle tied edges, and plot conversion rates with group counts. Use a separate zero bin where useful. Do not optimize bin boundaries against validation or test outcomes.
- Mark groups with fewer than 100 sessions as low-sample exploratory estimates. Do not discard them silently or treat the threshold as a formal significance test.

Required Matplotlib figures, each saved at 200+ DPI:

1. Purchase/non-purchase counts, with the conversion rate annotated.
2. Visitor-type conversion rates with confidence intervals and sample counts.
3. Conversion by observed month in calendar order, with counts and uncertainty.
4. Conversion by traffic-type code, with small groups visibly flagged.
5. Product count/duration distributions by purchase outcome, using log1p where helpful.
6. Conversion across product-engagement bins.
7. Correlation heatmap for meaningful numeric predictors; exclude categorical integer codes.

Use labeled axes, readable colors, concise titles, and honest scales. Each notebook section must include a short interpretation based on calculated output, not an assumed finding. Frame multiple segment comparisons as exploratory; do not generate a large collection of uncorrected significance tests.

## 5. Notebook 03 — Predictive model and strongest factors

### Preprocessing

Fit every learned transformation on training data only, then apply it unchanged to validation/test:

- Apply `log1p` to the six page count/duration predictors, preserving zeros. Do not log-transform the target or categorical codes.
- Standardize numeric features with training means and population standard deviations; handle constant columns safely.
- If any predictor values are missing after the audit, use training medians for numeric fields and an explicit missing category for categoricals. Show imputation counts; do not impute unnecessarily.
- One-hot encode categories with a fixed training vocabulary and one reference level per categorical feature. Explain reference categories; map unseen categories to the reference/all-zero representation and report their counts. Never create a different feature layout per split.
- Keep `PageValues`, row IDs, and `Revenue` out of the primary predictor matrix.

### NumPy logistic regression

Implement a small, documented binary logistic regression using batch gradient descent:

- Include an intercept, a numerically stable sigmoid, and binary cross-entropy computed with `np.logaddexp`.
- Minimize mean loss plus `(lambda / 2) * sum(beta_non_intercept**2)`; use the consistent gradient and do not penalize the intercept.
- Record training loss, iteration count, stopping reason, gradient norm, and convergence status. Raise a clear error or warning for nonfinite values or failure to converge; do not silently present unstable coefficients.
- Use a finite, predeclared grid of learning rates (0.01, 0.05, 0.1) and L2 strengths (0.001, 0.01, 0.1). Cap runs at 10,000 iterations with documented tolerance. Check convergence before comparing candidates. Plot the selected run's loss curve.
- Choose model settings using validation log loss only. For the final comparison, retain the selected model fitted on training data, so validation remains the reference for interpretation and tuning. Do not refit on test data.
- Explain probability, log odds, and regularization in accessible Markdown. Do not report classical p-values or coefficient confidence intervals from this penalized model.

### Baselines, metrics, and threshold

Compare against both an always-no-purchase classifier and a constant-probability predictor using training prevalence. Since purchases are the minority class, accuracy alone is insufficient.

Implement with NumPy: confusion matrix, precision, recall, F1, accuracy, log loss, Brier score, ROC AUC, and average precision. Handle tied scores correctly by grouping equal scores for ROC/precision-recall curves; define average precision as the recall-increment weighted precision sum. Explicitly handle division by zero and single-class input.

Report validation performance at threshold 0.5. Select an additional threshold that maximizes validation F1 over unique validation scores; break ties by choosing the higher threshold. Explain this is a technical tradeoff, not an economically optimal decision, because campaign costs and purchase values are unavailable.

Lock settings and thresholds, then evaluate once on the untouched test set. Save the metrics table, confusion matrix, ROC curve, precision-recall curve with prevalence baseline, and a calibration plot using probability bins with counts. Optional test-score bootstrap intervals may quantify uncertainty for the already locked model, without retuning it.

### Factor ranking

Use **grouped permutation importance** as the primary ranking:

1. On validation data, measure baseline average precision.
2. For each original predictor, permute its values across rows. For a one-hot categorical group, apply the same row permutation to every dummy column in that group to preserve valid encodings.
3. Recompute average precision; importance = baseline AP minus permuted AP.
4. Repeat 30 times per predictor with a fixed RNG. Export mean and standard deviation of AP decrease; label SD as permutation variability, not a confidence interval.
5. Rank original factors and plot the ranking. Retain negative estimates; do not truncate them to zero.

Use coefficients only as a complementary interpretation. Report standardized numeric coefficients and odds ratios per one standard deviation of the transformed feature; categorical odds ratios are relative to the stated reference category. Do not rank numeric and categorical dummy coefficients together as if they share the same scale.

Compare ranking with descriptive evidence. Discuss correlation between counts and durations, and between bounce/exit rates: interchangeable predictors can dilute or destabilize individual importance. Describe a factor as useful for this model and sample, not inherently the most important cause of purchasing.

### PageValues sensitivity check

Fit a separate model with `PageValues` included, using the same splits, training-only preprocessing, validation selection, and evaluation protocol. Freeze both model specifications before inspecting either test result; report their test comparison as one final evaluation. Document any performance increase and why the transaction-linked variable complicates prediction timing. Do not replace the main conclusions with this model or recommend targeting users based on transaction-derived information.

## 6. Notebook 04 — Business synthesis

Generate an executive summary from actual saved analysis outputs. Include:

- Scope, sample size, overall conversion rate, and the research question.
- Three to five supported findings, each with the calculated effect/importance, group sizes or metric context, uncertainty where applicable, and a linked chart/table.
- Whether the primary model improved over baselines, especially in AP and log loss. If it did not, say so; do not force a positive story.
- A clear answer to which factors ranked highest, distinguishing associations, predictive contribution, and causation.
- A recommendation table: observed evidence → possible action → proposed experiment → success metric → limitation. Recommendations must be hypotheses for testing, not promised conversion gains.
- Limitations: one historical dataset, summarized sessions, missing prices/costs/user histories, unknown live availability of features, class imbalance, small segments, correlated predictors, and exploratory comparisons.

Possible experiment ideas, only if the actual analysis supports them: test product-information presentation, onboarding for new visitors, or campaign messaging by visitor segment. Do not claim that increasing duration causes purchases or that a traffic code represents a known marketing channel.

Create `reports/executive_summary.md` with a concise business-facing narrative and figure links. Do not fabricate stakeholder interviews, university adoption, deployment, or impact. The owner can later use the project to demonstrate organized data management, clear reports, and evidence-based recommendations for program support roles.

## 7. README and exported deliverables

The README must include purpose, question, data citation/license, stack, repository map, setup, reproducibility, key results from the executed analysis, modeling assumptions, and limitations.

Provide these commands, implemented and verified:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_notebooks.py
python -m unittest discover -s tests
jupyter notebook
```

Include the Windows activation equivalent. Notebook execution must use nbconvert or equivalent Jupyter tooling, respect notebook order, stop on errors, and write executed outputs back to the notebooks. Notebook 04 should consume saved artifacts rather than manually copied results. Opening it directly without earlier outputs should show an actionable message.

Export: audit table, segment conversion tables, confidence intervals, split assignments, preprocessing metadata, model coefficients, validation factor ranking, validation/test model metrics, chosen thresholds, convergence logs, and all required figures. Tables should have descriptive names, units, and sample counts. Record split seed and actual versions in a run metadata file.

After execution only, add one truthful draft résumé bullet in the README using actual facts and metrics. Suggested structure: “Analyzed [verified number] e-commerce sessions with pandas and NumPy, built and evaluated a purchase-probability model, and visualized conversion patterns with Matplotlib.” Do not invent performance gains or describe association as business impact.

## 8. Verification and acceptance criteria

Use a small standard-library unittest suite for the nontrivial numerical functions, not tests that merely repeat the code. Verify:

- Wilson intervals contain known hand-calculated reference values; edge cases with zero/all purchases behave sensibly.
- Stratified splits have no overlap and full coverage.
- Train-fitted preprocessing keeps identical column order, handles unseen categories/constant columns, and does not learn from held-out rows.
- Logistic loss/gradient agree with a finite-difference check on a tiny synthetic example; loss decreases and probabilities are finite and bounded.
- Confusion metrics agree with a hand-calculated example; ranking metrics handle perfect, reversed, constant/tied scores, and single-class cases.
- Grouped permutation keeps categorical dummy blocks coherent and never includes the target.

Final acceptance:

1. Every notebook executes top-to-bottom in order from a clean kernel.
2. Primary predictors exclude `Revenue`, `PageValues`, and row IDs.
3. No transformation or model decision uses test outcomes before final evaluation.
4. Dataset discrepancies and cleaning choices are visible, including duplicates.
5. Model performance is compared against meaningful baselines, with convergence verified.
6. The final factor ranking is computed from the implemented model and honestly qualified.
7. Report findings match generated tables and figures; no placeholders remain after successful execution.
8. A reader can reproduce the project from the README without editing absolute paths.

## 9. Implementation order

1. Create repository scaffolding, requirements, data downloader, and citation.
2. Implement and run Notebook 01; verify audit and splits.
3. Implement and run Notebook 02; save statistical tables and figures.
4. Implement preprocessing, numerical model, metrics, and focused numerical checks.
5. Run validation selection, factor ranking, and the sensitivity model; lock choices before test evaluation.
6. Run final evaluation, build Notebook 04 and the executive summary, and populate README results.
7. Execute all notebooks again through the documented runner, run the numerical checks, and inspect exported charts for legibility.
8. Return the completed file tree, concise findings, verification outcomes, and any actual blockers. Do not describe unfinished work as complete.

**Scope boundary:** Build the analysis repository and its notebooks. Do not add deployment, live tracking, a dashboard, scraping, or unrelated models. Keep the first version focused enough for the owner to explain each calculation.
