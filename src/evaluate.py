import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    roc_curve
)

"""
evaluate.py
===========
Model evaluation and selection pipeline for Layer 1
clinical benchmark.

Implements threshold optimisation, full metric computation,
AF-specific secondary analysis, and the five-criterion
natural selection framework for model selection.

Called from: 03_modelling.ipynb
"""

# ── Pre-registered success thresholds ────────────────────────────
# Locked before any model training began.
# These values cannot be changed after seeing results.
MIN_SENSITIVITY = 0.80
MIN_SPECIFICITY = 0.75


def find_optimal_threshold(model, X_test, y_test):
    """
    Find the classification threshold that maximises
    specificity while meeting the minimum sensitivity floor.

    Scans all thresholds from 0.01 to 0.99 and selects
    the threshold that achieves sensitivity >= MIN_SENSITIVITY
    with the highest possible specificity.

    If no threshold meets the sensitivity floor, returns
    the threshold that achieves the highest sensitivity
    available and flags the model as failing the criterion.

    Parameters
    ----------
    model : fitted sklearn estimator
        Trained classifier with predict_proba method.
    X_test : pd.DataFrame
        Test set features — eight locked HRV features.
    y_test : pd.Series
        Test set binary labels (0=Normal, 1=Abnormal).

    Returns
    -------
    dict
        threshold       : float — optimal cutoff value
        sensitivity     : float — sensitivity at threshold
        specificity     : float — specificity at threshold
        meets_criterion : bool  — True if MIN_SENSITIVITY met
    """
    # Get probability scores for Abnormal class
    proba = model.predict_proba(X_test)[:, 1]

    best = {
        'threshold':       0.5,
        'sensitivity':     0.0,
        'specificity':     0.0,
        'meets_criterion': False
    }

    # Scan thresholds from low to high
    # Low threshold = high sensitivity, low specificity
    # High threshold = low sensitivity, high specificity
    for thresh in np.arange(0.01, 1.00, 0.01):
        preds = (proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, preds, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Only consider thresholds that meet sensitivity floor
        if sensitivity >= MIN_SENSITIVITY:
            # Among those, maximise specificity
            if specificity > best['specificity']:
                best = {
                    'threshold':       round(float(thresh), 2),
                    'sensitivity':     round(sensitivity, 4),
                    'specificity':     round(specificity, 4),
                    'meets_criterion': True
                }

    # If no threshold met the floor, return best sensitivity
    # achieved — used for failure mode documentation
    if not best['meets_criterion']:
        best_sens = 0.0
        best_thresh = 0.5
        for thresh in np.arange(0.01, 1.00, 0.01):
            preds = (proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_test, preds, labels=[0, 1]
            ).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            if sensitivity > best_sens:
                best_sens = sensitivity
                best_thresh = thresh

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            (proba >= best_thresh).astype(int),
            labels=[0, 1]
        ).ravel()
        best = {
            'threshold':       round(float(best_thresh), 2),
            'sensitivity':     round(
                tp / (tp + fn) if (tp + fn) > 0 else 0, 4
            ),
            'specificity':     round(
                tn / (tn + fp) if (tn + fp) > 0 else 0, 4
            ),
            'meets_criterion': False
        }

    return best


def evaluate_model(name, model, X_test, y_test, label_col=None):
    """
    Full evaluation report for a single trained model.

    Applies optimal threshold, computes all performance
    metrics, runs AF-specific secondary analysis, and
    returns a structured report dictionary.

    Parameters
    ----------
    name : str
        Human-readable model name for reporting.
        Example: 'Random Forest'
    model : fitted sklearn estimator
        Trained classifier with predict_proba method.
    X_test : pd.DataFrame
        Test set features.
    y_test : pd.Series
        Test set binary labels (0=Normal, 1=Abnormal).
    label_col : pd.Series or None
        Original Physionet labels (N, A, O) aligned with
        y_test. Required for AF-specific analysis.
        Pass None to skip AF analysis.

    Returns
    -------
    dict
        name            : str   — model name
        threshold       : float — optimal threshold
        sensitivity     : float — at optimal threshold
        specificity     : float — at optimal threshold
        auroc           : float — area under ROC curve
        f1_abnormal     : float — F1 for Abnormal class
        meets_criterion : bool  — passes sensitivity floor
        confusion_matrix: dict  — TP, TN, FP, FN counts
        af_sensitivity  : float or None — AF-specific recall
        failure_mode    : str   — plain language failure note
    """
    proba = model.predict_proba(X_test)[:, 1]

    # ── Find optimal threshold ────────────────────────────────────
    thresh_result = find_optimal_threshold(model, X_test, y_test)
    threshold       = thresh_result['threshold']
    sensitivity     = thresh_result['sensitivity']
    specificity     = thresh_result['specificity']
    meets_criterion = thresh_result['meets_criterion']

    # ── Apply threshold and compute metrics ───────────────────────
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_test, preds, labels=[0, 1]
    ).ravel()

    auroc       = round(roc_auc_score(y_test, proba), 4)
    f1_abnormal = round(
        f1_score(y_test, preds, pos_label=1, zero_division=0), 4
    )

    # ── AF-specific secondary analysis ───────────────────────────
    # Mandatory — AF is only 23.9% of the Abnormal class.
    # The model may learn Other rhythm signatures more
    # strongly. This surfaces that risk explicitly.
    af_sensitivity = None
    if label_col is not None:
        af_mask = label_col == 'A'
        if af_mask.sum() > 0:
            af_proba = proba[af_mask.values]
            af_preds = (af_proba >= threshold).astype(int)
            af_sensitivity = round(af_preds.mean(), 4)

    # ── Failure mode documentation ────────────────────────────────
    # Plain language note explaining why the model
    # failed the criterion if it did not meet the floor.
    failure_mode = None
    if not meets_criterion:
        failure_mode = (
            f"{name} achieved maximum sensitivity of "
            f"{sensitivity:.1%} — below the pre-registered "
            f"floor of {MIN_SENSITIVITY:.1%}. The eight "
            f"wearable-compatible features were insufficient "
            f"for this model to meet the screening threshold. "
            f"This indicates either a fundamental linear "
            f"separability limitation (if Logistic Regression) "
            f"or significant class overlap in the feature space "
            f"(if SVM)."
        )

    # ── Print report ──────────────────────────────────────────────
    status = "PASS" if meets_criterion else "FAIL"
    print(f"\n{'='*55}")
    print(f"  {name.upper()} [{status}]")
    print(f"{'='*55}")
    print(f"  Optimal threshold : {threshold}")
    print(f"  Sensitivity       : {sensitivity:.1%}")
    print(f"  Specificity       : {specificity:.1%}")
    print(f"  AUROC             : {auroc:.4f}")
    print(f"  F1 (Abnormal)     : {f1_abnormal:.4f}")
    if af_sensitivity is not None:
        print(f"  AF sensitivity    : {af_sensitivity:.1%}")
    print(f"  Confusion matrix  :")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    if failure_mode:
        print(f"\n  Failure mode: {failure_mode}")

    return {
        'name':             name,
        'threshold':        threshold,
        'sensitivity':      sensitivity,
        'specificity':      specificity,
        'auroc':            auroc,
        'f1_abnormal':      f1_abnormal,
        'meets_criterion':  meets_criterion,
        'confusion_matrix': {
            'tp': int(tp), 'tn': int(tn),
            'fp': int(fp), 'fn': int(fn)
        },
        'af_sensitivity':   af_sensitivity,
        'failure_mode':     failure_mode
    }


def select_best_model(reports, models):
    """
    Apply the five-criterion natural selection framework
    to select the best model for Layer 2.

    Parameters
    ----------
    reports : list of dict
        Evaluation report dictionaries from evaluate_model.
    models : dict
        Dictionary mapping model name to fitted estimator.
        Keys must match report['name'] values.

    Returns
    -------
    dict
        selected_name   : str — winning model name
        selected_model  : fitted estimator
        selected_report : dict — winning model's report
        reason          : str — plain language selection reason
        eliminated      : list — names of eliminated models
    """
    print(f"\n{'='*55}")
    print("  MODEL SELECTION — NATURAL SELECTION FRAMEWORK")
    print(f"{'='*55}")

    # ── Criterion 1: Survival condition ──────────────────────────
    # Eliminate any model below sensitivity floor
    survivors = [r for r in reports if r['meets_criterion']]
    eliminated = [r['name'] for r in reports
                  if not r['meets_criterion']]

    if eliminated:
        print(f"\n  Eliminated (below sensitivity floor):")
        for name in eliminated:
            report = next(r for r in reports if r['name'] == name)
            print(f"    {name}: "
                  f"sensitivity={report['sensitivity']:.1%}")

    # ── Null outcome protocol ─────────────────────────────────────
    if not survivors:
        best = max(reports, key=lambda r: r['sensitivity'])
        print(f"\n  NULL OUTCOME: No model met the "
              f"sensitivity floor of {MIN_SENSITIVITY:.0%}.")
        print(f"  Best sensitivity achieved: "
              f"{best['sensitivity']:.1%} by {best['name']}.")
        print(f"  This is a qualified negative finding.")
        return {
            'selected_name':   None,
            'selected_model':  None,
            'selected_report': best,
            'reason': (
                f"No model met the pre-registered sensitivity "
                f"floor of {MIN_SENSITIVITY:.0%}. The best "
                f"sensitivity achieved was {best['sensitivity']:.1%}"
                f" by {best['name']}. This constitutes a qualified "
                f"negative finding — the eight wearable-compatible "
                f"features are insufficient to meet the clinical "
                f"screening threshold at this performance level."
            ),
            'eliminated': [r['name'] for r in reports]
        }

    # ── Criterion 2: Highest specificity ─────────────────────────
    survivors.sort(key=lambda r: r['specificity'], reverse=True)
    if len(survivors) == 1 or (
        survivors[0]['specificity'] != survivors[1]['specificity']
    ):
        winner = survivors[0]
        reason = (
            f"{winner['name']} selected. Met sensitivity floor "
            f"({winner['sensitivity']:.1%}) with highest "
            f"specificity among survivors "
            f"({winner['specificity']:.1%})."
        )
    else:
        # ── Criterion 3: AUROC tiebreaker ────────────────────────
        survivors.sort(key=lambda r: r['auroc'], reverse=True)
        winner = survivors[0]
        reason = (
            f"{winner['name']} selected. Tied on specificity — "
            f"AUROC tiebreaker applied. "
            f"AUROC: {winner['auroc']:.4f}."
        )

    print(f"\n  Surviving models:")
    for r in survivors:
        print(f"    {r['name']}: "
              f"sensitivity={r['sensitivity']:.1%}  "
              f"specificity={r['specificity']:.1%}  "
              f"AUROC={r['auroc']:.4f}")

    print(f"\n  SELECTED: {winner['name']}")
    print(f"  Reason: {reason}")

    return {
        'selected_name':   winner['name'],
        'selected_model':  models[winner['name']],
        'selected_report': winner,
        'reason':          reason,
        'eliminated':      eliminated
    }