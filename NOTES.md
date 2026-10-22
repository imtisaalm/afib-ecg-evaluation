# What a clinician would need to trust this

The clinical question is whether an alert identifies someone who needs
confirmatory assessment, with an acceptable burden of missed episodes and
unnecessary alerts. A good classification score alone does not answer it.

## False negatives have a clinical cost

Missing AF can delay recognition and assessment of stroke risk. A negative
30-beat window cannot exclude intermittent AF elsewhere in the recording,
and a negative wearable screen must not be treated as reassurance that AF
is absent. The cost of a miss depends on episode duration, monitoring
coverage, patient risk and the action the result would trigger.

I would want the missed episodes reviewed on the ECG. Were they brief,
slow, relatively regular, noisy or concentrated in particular patients?
Window sensitivity needs to be accompanied by episode sensitivity and time
to detection. Detecting AF also does not itself establish an indication
for anticoagulation.

## False positives can undermine the whole service

A smartwatch may evaluate many windows in a day. Even modest false-positive
rates can create repeated alerts, anxiety, unnecessary consultations and
alarm fatigue. Users may eventually ignore a genuine alert.

Specificity of 90% is an educational operating target here, not a proposed
clinical standard. It permits false positives in up to 10% of negative
windows at that operating point. That does not translate directly into a
daily alert rate because windows are correlated and alerts may be grouped.
A credible wearable evaluation must report false alerts per person-day,
positive predictive value and the follow-up workload at realistic prevalence.

## Why sensitivity at fixed specificity is the headline metric

AUROC describes overall discrimination, including operating regions that
may be unusable in practice. Sensitivity at a fixed specificity asks a more
concrete question: how much AF can we detect while limiting false positives?

The specificity target must ultimately come from the intended use and
acceptable alert burden. In this demo, the test ROC summary describes
sensitivity where specificity is at least 90%. It is not proof that an
independently chosen threshold will achieve that target.

The stronger operational check selects the threshold on validation
records, locks it, and reports both sensitivity and achieved specificity
on unseen test records. If specificity falls, that is evidence of limited
generalization. It must not be hidden by retuning on the test set.

## Hospital ECG does not validate a smartwatch

AFDB contains ambulatory ECG recordings with existing beat and rhythm
annotations. A smartwatch single-lead ECG changes electrode geometry,
contact, recording duration and signal quality. Optical pulse sensing adds
a different measurement process and should not be treated as ECG.

Motion, missed beats, ectopy, medication, heart rate and population
differences can all alter RR features. An AF-enriched database also does
not represent the prevalence of AF in a general screening population.
Performance needs separate examination across devices, clinical groups
and signal-quality levels.

## What validation comes next

First, expand the independent patient sample and review annotation quality
and exclusions. Then validate the complete pipeline, including beat
detection and signal-quality handling, on an external wearable dataset
with independent expert rhythm adjudication.

Before prospective testing, lock the model, threshold, alert aggregation
rule and intended clinical action. Prespecify episode sensitivity, detection
delay, false alerts per person-day, positive predictive value, subgroup
performance and uncertainty estimated at the patient level.

A prospective study must then establish whether the system supports a
workable confirmatory assessment pathway. This eight-record demo is a
learning baseline. It cannot establish clinical safety, improved outcomes
or readiness for deployment.
