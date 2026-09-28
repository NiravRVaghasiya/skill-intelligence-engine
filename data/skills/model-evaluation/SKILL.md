---
name: model-evaluation
display_name: Model Evaluation
description: >
  Use when the user wants to score, cross-validate, or calibrate a fitted model.
  Trigger phrases: "which metric should I use", "cross-validate this model",
  "is my model overfitting", "calibrate predicted probabilities", "build a
  confusion matrix", "compare two models rigorously". NOT for choosing which
  algorithm to fit (see supervised-learning) or searching hyperparameters (see
  hyperparameter-tuning).
type: workflow
domain: classical-ml
level: intermediate
lifecycle: stable
risk_level: medium
evidence_level: established-practice
last_verified: 2026-09-21
capabilities:
  - cross-validation-strategy-selection
  - classification-metric-selection
  - probability-calibration
  - imbalanced-classification-metrics
  - overfitting-diagnosis
  - confusion-matrix-analysis
requires:
conflicts:
related:
  - supervised-learning
  - hyperparameter-tuning
  - feature-engineering
  - data-preprocessing
inputs: A fitted model plus a held-out test set (or a train set and CV strategy) and a defined target/metric.
outputs: A metric report with fold-to-fold spread, confusion matrix, and calibration diagnostics establishing whether the model's reported performance and probabilities can be trusted.
---

## Overview
Covers how to *score* a model honestly: picking a metric that matches the business
problem, choosing a cross-validation strategy that matches the data's structure,
reading confusion-matrix-level detail beyond a single accuracy number, checking
whether predicted probabilities are trustworthy, and diagnosing over/underfitting
with learning curves. This skill assumes the model itself is already chosen (see
`supervised-learning`) — it's about producing a number (or curve) you can actually
trust and act on.

## Workflow
1. **Pick the metric before looking at any score.** Accuracy is misleading on
   imbalanced classes; pick based on what an error actually costs.
   ```python
   from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

   # Fraud detection: false negatives are expensive -> prioritize recall / PR-AUC
   # over plain accuracy on an imbalanced positive class.
   y_pred = model.predict(X_test)
   y_proba = model.predict_proba(X_test)[:, 1]
   print(classification_report(y_test, y_pred))
   print("ROC-AUC:", roc_auc_score(y_test, y_proba))
   print("PR-AUC:", average_precision_score(y_test, y_proba))
   ```
2. **Match the CV splitter to the data's structure.** Random k-fold is wrong for
   imbalanced classes (use stratification) and wrong for time-ordered data (use a
   forward-chaining split).
   ```python
   from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

   cv_classification = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   cv_timeseries = TimeSeriesSplit(n_splits=5)   # never shuffle time-ordered rows
   ```
3. **Use `cross_validate` for multiple metrics at once, and report spread.** A
   single mean hides how stable the model is fold-to-fold.
   ```python
   from sklearn.model_selection import cross_validate

   scoring = ["roc_auc", "average_precision", "f1"]
   cv_results = cross_validate(model, X_train, y_train, cv=cv_classification,
                                scoring=scoring, return_train_score=True)
   for metric in scoring:
       test_scores = cv_results[f"test_{metric}"]
       print(f"{metric}: {test_scores.mean():.3f} +/- {test_scores.std():.3f}")
   ```
4. **Diagnose over/underfitting with train vs. validation score.** A large gap
   between `train_<metric>` and `test_<metric>` from step 3 means overfitting;
   both being low means underfitting.
   ```python
   for metric in scoring:
       train_mean = cv_results[f"train_{metric}"].mean()
       test_mean = cv_results[f"test_{metric}"].mean()
       print(f"{metric}: train={train_mean:.3f} test={test_mean:.3f} gap={train_mean - test_mean:.3f}")
   ```
5. **Inspect the confusion matrix, not just one summary number.** Precision/recall
   trade-offs are invisible in a single F1 or accuracy figure.
   ```python
   from sklearn.metrics import ConfusionMatrixDisplay

   ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
   ```
6. **Check calibration before trusting predicted probabilities.** A model with
   great ROC-AUC can still have systematically wrong probability estimates —
   critical if downstream logic thresholds or sums probabilities.
   ```python
   from sklearn.calibration import calibration_curve, CalibratedClassifierCV

   prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
   # plot prob_pred vs prob_true; a diagonal line means well-calibrated

   # If miscalibrated, wrap the fitted estimator (fit calibration on held-out data,
   # not the same data the base model was trained on):
   calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
   calibrated.fit(X_train, y_train)
   ```
7. **Plot a learning curve for a clearer over/underfitting picture across
   training-set sizes.**
   ```python
   from sklearn.model_selection import learning_curve
   import numpy as np

   train_sizes, train_scores, val_scores = learning_curve(
       model, X_train, y_train, cv=cv_classification, scoring="roc_auc",
       train_sizes=np.linspace(0.1, 1.0, 5),
   )
   print("train:", train_scores.mean(axis=1))
   print("val:  ", val_scores.mean(axis=1))
   ```

## Gotchas
- **Accuracy on imbalanced classes is nearly meaningless.** A 95%-negative dataset
  gets 95% accuracy by predicting the majority class every time — use PR-AUC,
  recall/precision at a chosen threshold, or F1 instead.
- **Random k-fold on time-ordered data leaks the future into training.** Any row
  from "next month" ending up in a training fold whose validation fold is "last
  month" inflates the score unrealistically — use `TimeSeriesSplit`.
- **Tuning and final evaluation on the same fold(s).** If the same CV split was
  used to pick a model/threshold, the resulting score is optimistic — hold out a
  final untouched test set, or nest CV (see `hyperparameter-tuning`).
- **A high ROC-AUC does not imply well-calibrated probabilities.** ROC-AUC only
  cares about rank ordering; if the application needs actual probability values
  (e.g., expected-value calculations), check `calibration_curve` explicitly.
- **Calibrating on the same data the model was trained on overfits the
  calibration too.** `CalibratedClassifierCV` needs its own CV or held-out split —
  don't calibrate against training predictions.

## References
- [scikit-learn: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — full metric catalog and scoring-string reference.
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — splitter strategies including `TimeSeriesSplit` and `StratifiedKFold`.
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) — calibration curves and `CalibratedClassifierCV` usage.
