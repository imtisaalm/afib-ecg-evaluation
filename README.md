# AFib screening from RR intervals

A small biomedical engineering project using public ECG annotations, nine
interpretable RR features, logistic regression and a shallow random forest.

The clinical motivation is straightforward: AF can be intermittent and
asymptomatic, yet associated with stroke risk. A normal-looking ECG recorded
between episodes does not exclude AF. This project detects annotated AF
windows. It does not predict stroke risk or detect AF within genuinely
normal sinus rhythm.

Smartwatch screening makes the tradeoff between missed episodes and false
alarms especially relevant. This is a first exercise in learning that
tradeoff properly, with explicit labels and evaluation on unseen records.

## Run in about five minutes

Use Python 3.10 or newer. Save the files in this structure:

```text
afib-demo/
    src/
        afib_demo.py
    requirements.txt
    README.md
    NOTES.md
```

From the project directory on macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/afib_demo.py
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

The script streams headers and annotations through WFDB using
`pn_dir="afdb"`. It does not download the full ECG waveforms. Setup and
execution should usually take a few minutes on a laptop with a working
internet connection, but package installation and PhysioNet response times
can exceed five minutes.

A network or annotation error stops the run. Records are never silently
skipped. Rerun after resolving the error.

Outputs:

- `results/metrics.json`
- `results/confusion_matrix_logistic_regression.png`
- `results/confusion_matrix_random_forest.png`

Results are also printed. Re-running overwrites these outputs. Installed
package versions and per-record exclusion counts are saved in the JSON.

## Method in eight lines

1. Stream eight AFDB records using `.qrs` beat and `.atr` rhythm annotations.
2. Form disjoint blocks of 30 consecutive beats, giving 29 RR intervals each.
3. Track rhythm coverage over the entire block and drop unknown or mixed rhythms.
4. Label `(AFIB` as positive; stable `(N`, `(AFL` and `(J` as negative.
5. Calculate nine RR features using seconds and a fixed histogram definition.
6. Fit standardized logistic regression and a shallow random forest on four records.
7. Select each operating threshold on two separate validation records at specificity of at least 90%.
8. Evaluate once on two untouched test records and save metrics and confusion matrices.

## Record separation

| Partition | Record IDs | Purpose |
| --- | --- | --- |
| Training | 04015, 04043, 04048, 04126 | Fit models and logistic regression scaler |
| Validation | 04746, 04908 | Select operating thresholds |
| Test | 05121, 05261 | Final evaluation |

The split is BY RECORD. No windows from the same patient record appear in
both training and test, or in any two partitions. The split is fixed in
advance. The script checks that record IDs are disjoint and that each
partition contains both classes.

This avoids learning and testing on nearby windows from the same recording.
It does not establish generalization across hospitals, devices or populations.

## Feature definitions

All timing features use seconds. Each window contains 29 RR intervals and
28 successive RR differences.

| Feature | Definition |
| --- | --- |
| Mean RR | Arithmetic mean of RR intervals |
| SDNN | Sample standard deviation of RR intervals, `ddof=1` |
| RMSSD | Root mean square of successive RR differences |
| pNN50 | Fraction of successive differences strictly greater than 0.050 s in absolute value |
| RR coefficient of variation | Sample standard deviation divided by mean RR |
| Median absolute successive difference | Median of absolute successive RR differences |
| RR histogram Shannon entropy | `-sum(p * log2(p))`, excluding empty bins |
| Minimum RR | Smallest RR interval |
| Maximum RR | Largest RR interval |

Entropy uses fixed 100 ms bins between 0 and 2 seconds, plus an overflow
bin. It is measured in bits. This is histogram entropy, not sample entropy.

SDNN and pNN50 are conventional names here. These calculations use RR
intervals without verified normal-to-normal beat selection, so they should
not be interpreted as clinical NN-based heart-rate variability measurements.

No ectopic-beat correction or physiological RR clipping is performed.
Only nonpositive or nonfinite RR intervals are rejected. A single very
long or short interval can therefore influence the features.

## Results

No actual dataset run has been performed for this supplied project.
The table intentionally contains no estimated or invented results.

| Model | Test AUROC | Sensitivity at 0.5 | Specificity at 0.5 | Test ROC sensitivity at specificity >=90% |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.971 | 0.387 | 0.984 | 0.968 |
| Random forest | 0.955 | 0.060 | 0.995 | 0.949 |

The more operationally meaningful result uses a threshold chosen on
validation records and then held fixed:

| Model | Validation threshold | Test sensitivity at locked threshold | Test specificity at locked threshold |
| --- | --- | --- | --- |
| Logistic regression | 0.0330 | 0.918 | 0.945 |
| Random forest | 0.0498 | 0.943 | 0.907 |

### Reading the metrics

AFIB is the positive class.

- AUROC measures discrimination across thresholds.
- Sensitivity is `TP / (TP + FN)`.
- Specificity is `TN / (TN + FP)`.
- Sensitivity at 90% specificity is the largest empirical sensitivity with
  specificity at least 90%. Discrete scores may not permit exactly 90%.
- Confusion matrices use true classes as rows and predicted classes as
  columns, ordered `[non-AFIB, AFIB]`: `[[TN, FP], [FN, TP]]`.

The test ROC metric is descriptive. Its operating point uses test labels
and must not be presented as a threshold validated for deployment.

The separate validation-selected threshold is fixed before test scoring.
Its test specificity may fall below 90%. That failure would be a result
to report, not something to repair by adjusting the threshold on test data.
The saved PNGs show this locked-threshold evaluation. JSON also contains
confusion matrices at the default threshold of 0.5.

All metrics are pooled, window-level estimates. They are not patient-level
diagnostic accuracy, episode detection rates or false alarms per day.

## Interpreting the models

Logistic regression is the primary interpretable baseline. Its scaler is
fitted only on training data. Saved coefficients describe changes in
log-odds per training-set standard deviation of each feature, conditional
on the other features.

The forest is a limited nonlinear comparison: 200 trees, maximum depth 6,
and at least 20 training windows per leaf. Its impurity feature importances
are saved, but correlated predictors can make these rankings misleading.

Neither model establishes causation. Predicted probabilities are not
validated clinical risk estimates. Hyperparameters are fixed rather than
selected using test performance.

## Limitations

- RR-only features miss atrial activity, P waves and waveform morphology.
  Ectopy and other irregular rhythms can resemble AF.
- AFDB `.qrs` files contain automated, unaudited beat annotations. This
  project does not build or validate a beat detector.
- Eight records are a small cohort. Only two test records cannot establish
  stable generalization, regardless of the number of windows.
- Windows from the same record remain correlated. No confidence intervals
  are supplied; treating windows as independent would overstate precision.
- Dropping transition windows makes the task cleaner than real screening
  and leaves episode onset and offset performance untested.
- Non-AFIB does not mean normal. Atrial flutter and junctional rhythm are
  included as negative classes for this narrowly defined AF task.
- Thirty beats represent different durations at different heart rates.
  Window sensitivity does not measure time to detection.
- Hospital ambulatory ECG annotations are not smartwatch single-lead ECG
  or optical pulse measurements. Motion, contact quality, sampling,
  population and AF prevalence can all differ.
- This is not a medical device, diagnostic tool or treatment recommendation.
  A negative prediction cannot rule out AF.
- Results are not comparable to published deep-learning results without
  identical cohorts, labels, exclusions, splits and endpoints.

## Next steps

1. Expand to all suitable AFDB records and inspect exclusions by rhythm.
2. Use grouped cross-validation within development data, retaining an
   untouched external test cohort.
3. Add record-clustered uncertainty estimates once enough independent
   records are available.
4. Review false negatives and false positives against ECG waveforms.
5. Validate a beat detector, signal-quality rejection and ectopy handling.
6. Measure episode sensitivity, detection delay and false alerts per day
   under a prespecified alert aggregation rule.
7. Test a locked model and threshold on independently labelled wearable
   recordings before considering a prospective clinical study.

## Sources

- [PhysioNet MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/)
- [WFDB Python input/output documentation](https://wfdb.readthedocs.io/en/latest/io.html)
- [NHLBI: Atrial fibrillation](https://www.nhlbi.nih.gov/health/atrial-fibrillation)
