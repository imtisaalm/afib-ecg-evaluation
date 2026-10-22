#!/usr/bin/env python3
"""Educational RR-only AF screening with record-disjoint evaluation."""


import json
import platform
from importlib.metadata import version
from pathlib import Path


import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
BEATS_PER_WINDOW = 30
TARGET_SPECIFICITY = 0.90
RESULTS = Path(__file__).resolve().parents[1] / "results"


# Fixed before evaluation. Never move records after inspecting performance.
# All eight records have ECG headers, beat annotations and rhythm annotations.
SPLITS = {
    "train": ["04015", "04043", "04048", "04126"],
    "validation": ["04746", "04908"],
    "test": ["05121", "05261"],
}


FEATURES = [
    "mean_rr_s",
    "sdnn_rr_s",
    "rmssd_s",
    "pnn50",
    "rr_cv",
    "median_abs_successive_rr_diff_s",
    "rr_histogram_entropy_bits",
    "min_rr_s",
    "max_rr_s",
]
# Fixed edges, shared across every window and split.
# 100 ms bins from 0 to 2 seconds, plus an overflow bin.
ENTROPY_EDGES = np.r_[np.arange(0.0, 2.01, 0.1), np.inf]


# WFDB beat symbols. Non-beat annotations must not become RR intervals.
BEAT_SYMBOLS = set("NLRaVFJASEj/QfnrBe")
KNOWN_RHYTHMS = {"(AFIB", "(AFL", "(J", "(N"}


def rr_features(rr):
    """Nine features from 29 RR intervals, measured in seconds."""
    differences = np.diff(rr)
    absolute_differences = np.abs(differences)
    mean_rr = float(np.mean(rr))
    sdnn_rr = float(np.std(rr, ddof=1))

    counts, _ = np.histogram(rr, bins=ENTROPY_EDGES)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    return {
        "mean_rr_s": mean_rr,
        # Conventional feature name; these are RR, not cleaned NN, intervals.
        "sdnn_rr_s": sdnn_rr,
        "rmssd_s": float(np.sqrt(np.mean(differences**2))),
        "pnn50": float(np.mean(absolute_differences > 0.050)),
        "rr_cv": sdnn_rr / mean_rr,
        "median_abs_successive_rr_diff_s": float(
            np.median(absolute_differences)
        ),
        "rr_histogram_entropy_bits": entropy,
        "min_rr_s": float(np.min(rr)),
        "max_rr_s": float(np.max(rr)),
    }


def load_record(record):
    """Stream header, automated beat annotations and reference rhythms."""
    print(f"Streaming AFDB record{record}...", flush=True)
    header = wfdb.rdheader(record, pn_dir="afdb")
    beats_annotation = wfdb.rdann(record, "qrs", pn_dir="afdb")
    rhythm_annotation = wfdb.rdann(record, "atr", pn_dir="afdb")

    fs = float(header.fs)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"{record}: invalid sampling frequency")

    beat_mask = np.array(
        [symbol in BEAT_SYMBOLS for symbol in beats_annotation.symbol],
        dtype=bool,
    )
    beats = np.asarray(beats_annotation.sample[beat_mask], dtype=np.int64)
    if len(beats) < BEATS_PER_WINDOW or np.any(np.diff(beats) <= 0):
        raise ValueError(f"{record}: insufficient or unordered beat annotations")
    if beats[0] < 0 or beats[-1] >= header.sig_len:
        raise ValueError(f"{record}: beat annotations outside the record")

    # A rhythm remains active from its annotation until the next change.
    # Keep unknown rhythm strings so they invalidate coverage, not disappear.
    events = {}
    for sample, note in zip(
        rhythm_annotation.sample, rhythm_annotation.aux_note
    ):
        rhythm = note.replace("\x00", "").strip()
        if rhythm.startswith("("):
            events[int(sample)] = rhythm
    if not events:
        raise ValueError(f"{record}: no rhythm annotations")
    change_samples = np.array(sorted(events), dtype=np.int64)
    rhythms = np.array([events[int(sample)] for sample in change_samples])

    audit = {
        "record": record,
        "sampling_frequency_hz": fs,
        "beat_annotation_extension": "qrs",
        "beat_count": int(len(beats)),
        "candidate_windows": 0,
        "dropped_unknown_rhythm": 0,
        "dropped_mixed_rhythm": 0,
        "dropped_invalid_rr": 0,
        "unused_trailing_beats": int(len(beats) % BEATS_PER_WINDOW),
    }
    rows = []

    # Disjoint blocks: exactly 30 beat timestamps give 29 RR intervals.
    # The interval connecting consecutive blocks is deliberately unused.
    for start in range(0, len(beats) - BEATS_PER_WINDOW + 1, BEATS_PER_WINDOW):
        block = beats[start : start + BEATS_PER_WINDOW]
        first_sample, last_sample = int(block[0]), int(block[-1])
        audit["candidate_windows"] += 1

        active_index = int(
            np.searchsorted(change_samples, first_sample, side="right") - 1
        )
        if active_index < 0:
            audit["dropped_unknown_rhythm"] += 1
            continue

        # Include a change exactly at the last beat conservatively.
        stop_index = int(
            np.searchsorted(change_samples, last_sample, side="right")
        )
        covering_rhythms = set(rhythms[active_index:stop_index])

        if not covering_rhythms.issubset(KNOWN_RHYTHMS):
            audit["dropped_unknown_rhythm"] += 1
            continue
        if len(covering_rhythms) != 1:
            # Also drops N-to-AFL and other non-AF rhythm transitions.
            audit["dropped_mixed_rhythm"] += 1
            continue

        rr = np.diff(block) / fs
        if not np.all(np.isfinite(rr)) or np.any(rr <= 0):
            audit["dropped_invalid_rr"] += 1
            continue

        rhythm = next(iter(covering_rhythms))
        rows.append(
            {
                "record": record,
                "start_sample": first_sample,
                "end_sample": last_sample,
                "rhythm": rhythm,
                "label": int(rhythm == "(AFIB"),
                **rr_features(rr),
            }
        )

    if not rows:
        raise ValueError(f"{record}: no usable windows")

    frame = pd.DataFrame(rows)
    audit["retained_windows"] = int(len(frame))
    audit["afib_windows"] = int(frame["label"].sum())
    audit["non_afib_windows"] = int((frame["label"] == 0).sum())
    return frame, audit


def operating_point(y_true, probabilities, specificity=TARGET_SPECIFICITY):
    """Best empirical sensitivity with specificity at least the target.


    Uses actual ROC steps, without interpolation or randomization.
    This is descriptive when called on test labels.
    """
    fpr, tpr, thresholds = roc_curve(
        y_true, probabilities, drop_intermediate=False
    )
    eligible = np.flatnonzero(fpr <= 1.0 - specificity + 1e-12)
    best_sensitivity = np.max(tpr[eligible])
    tied = eligible[np.isclose(tpr[eligible], best_sensitivity)]
    # Among equal sensitivities, prefer the lower false-positive rate.
    index = int(tied[np.argmin(fpr[tied])])
    threshold = float(thresholds[index])

    # The first ROC point means "predict no positives". Represent it using
    # a finite value above every valid predict_proba output for valid JSON.
    if not np.isfinite(threshold) or threshold > 1.0:
        threshold = float(np.nextafter(1.0, 2.0))

    return {
        "threshold": threshold,
        "sensitivity": float(tpr[index]),
        "specificity": float(1.0 - fpr[index]),
    }


def threshold_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_plot(y_true, probabilities, threshold, name):
    predictions = (probabilities >= threshold).astype(int)
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=["Non-AFIB", "AFIB"],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{name}\nValidation-selected threshold")
    figure.tight_layout()
    figure.savefig(RESULTS / f"confusion_matrix_{name}.png", dpi=160)
    plt.close(figure)


def main():
    all_records = [record for records in SPLITS.values() for record in records]
    if len(all_records) < 6 or len(set(all_records)) != len(all_records):
        raise ValueError("Need at least six distinct, split-disjoint records")

    RESULTS.mkdir(parents=True, exist_ok=True)
    frames, audits = [], []
    for split, records in SPLITS.items():
        for record in records:
            # Fail visibly on download errors. Never silently change the cohort.
            frame, audit = load_record(record)
            frame["split"] = split
            frames.append(frame)
            audits.append({**audit, "split": split})
    data = pd.concat(frames, ignore_index=True)

    partitions = {
        split: data.loc[data["split"] == split].copy() for split in SPLITS
    }
    for split, frame in partitions.items():
        if set(frame["label"].unique()) != {0, 1}:
            raise ValueError(f"{split}: both classes are required")
        if not np.isfinite(frame[FEATURES].to_numpy()).all():
            raise ValueError(f"{split}: nonfinite features")

    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    report = {
        "seed": SEED,
        "database": "PhysioNet AFDB",
        "split_records": SPLITS,
        "beats_per_window": BEATS_PER_WINDOW,
        "rr_intervals_per_window": BEATS_PER_WINDOW - 1,
        "features": FEATURES,
        "target_specificity": TARGET_SPECIFICITY,
        "confusion_matrix_order": {
            "rows": ["true_non_afib", "true_afib"],
            "columns": ["predicted_non_afib", "predicted_afib"],
        },
        "record_audit": audits,
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in [
                    "wfdb", "scikit-learn", "numpy", "pandas", "matplotlib"
                ]
            },
        },
        "models": {},
    }
    summary = []

    for name, model in models.items():
        model.fit(train[FEATURES], train["label"])
        validation_scores = model.predict_proba(validation[FEATURES])[:, 1]

        # Select the deployable threshold using validation records only.
        selected = operating_point(validation["label"], validation_scores)
        test_scores = model.predict_proba(test[FEATURES])[:, 1]

        fixed = threshold_metrics(test["label"], test_scores, 0.5)
        locked = threshold_metrics(
            test["label"], test_scores, selected["threshold"]
        )
        # Test-ROC summary only. Do not deploy this test-selected threshold.
        descriptive = operating_point(test["label"], test_scores)
        auc = float(roc_auc_score(test["label"], test_scores))
        if name == "logistic_regression":
            coefficients = model.named_steps["logisticregression"].coef_[0]
            interpretation = {
                "standardized_coefficients": dict(
                    zip(FEATURES, coefficients.tolist())
                ),
                "intercept": float(
                    model.named_steps["logisticregression"].intercept_[0]
                ),
            }
        else:
            interpretation = {
                "impurity_feature_importances": dict(
                    zip(FEATURES, model.feature_importances_.tolist())
                ),
            }
        report["models"][name] = {
            "test_auroc": auc,
            "test_at_threshold_0_5": fixed,
            "validation_operating_point": selected,
            "test_at_validation_selected_threshold": locked,
            "test_sensitivity_at_90_specificity_descriptive": {
                "sensitivity": descriptive["sensitivity"],
                "achieved_specificity": descriptive["specificity"],
                "note": (
                    "Empirical test ROC summary at specificity >= 0.90; "
                    "not an independently validated operating threshold."
                ),
            },
            "interpretation": interpretation,
        }
        save_confusion_plot(
            test["label"], test_scores, selected["threshold"], name
        )
        summary.append(
            {
                "model": name,
                "AUROC": auc,
                "sensitivity@0.5": fixed["sensitivity"],
                "specificity@0.5": fixed["specificity"],
                "test_ROC_sens@spec>=0.90": descriptive["sensitivity"],
                "locked_sensitivity": locked["sensitivity"],
                "locked_specificity": locked["specificity"],
            }
        )

    output = RESULTS / "metrics.json"
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\nRecord audit:")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\nHeld-out test results:")
    print(pd.DataFrame(summary).to_string(index=False, float_format="%.4f"))
    print("\nFull metrics, including confusion matrices:")
    print(json.dumps(report["models"], indent=2, allow_nan=False))
    print(f"\nSaved metrics and confusion matrices to{RESULTS}")


if __name__ == "__main__":
    main()
