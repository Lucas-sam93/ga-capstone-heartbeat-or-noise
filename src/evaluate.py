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

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

from .constants import SEPARATOR

# ── Pre-registered success thresholds ────────────────────────────
# Locked before any model training began.
# These values cannot be changed after seeing results.
MIN_SENSITIVITY = 0.80
MIN_SPECIFICITY = 0.75


def _compute_sens_spec(y_true: np.ndarray,
                       y_pred: np.ndarray) -> tuple[float, float]:
    """Compute sensitivity and specificity from true/predicted labels."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def find_optimal_threshold(model, X_test: pd.DataFrame,
                           y_test: pd.Series) -> dict:
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
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)

    # Single scan — collect metrics at every threshold
    best_meeting_floor = None   # Best specificity among those meeting sensitivity floor
    best_overall_sens = None    # Highest sensitivity if no threshold meets floor

    for thresh in thresholds:
        preds = (proba >= thresh).astype(int)
        sens, spec = _compute_sens_spec(y_test, preds)

        # Track best threshold meeting sensitivity floor
        if sens >= MIN_SENSITIVITY:
            if (best_meeting_floor is None
                    or spec > best_meeting_floor['specificity']):
                best_meeting_floor = {
                    'threshold':       round(float(thresh), 2),
                    'sensitivity':     round(sens, 4),
                    'specificity':     round(spec, 4),
                    'meets_criterion': True
                }

        # Track best sensitivity overall (fallback)
        if (best_overall_sens is None
                or sens > best_overall_sens['sensitivity']):
            best_overall_sens = {
                'threshold':       round(float(thresh), 2),
                'sensitivity':     round(sens, 4),
                'specificity':     round(spec, 4),
                'meets_criterion': False
            }

    return best_meeting_floor if best_meeting_floor else best_overall_sens


def cross_validate_models(models: dict, X_train: pd.DataFrame,
                          y_train: pd.Series, n_folds: int = 5,
                          random_state: int = 42) -> dict:
    """
    Five-fold stratified cross-validation with threshold
    optimisation applied independently within each fold.

    For each model and each fold:
    1. Split training data into fold-train and fold-val
    2. Fit a fresh StandardScaler on fold-train only
    3. Train a cloned model on fold-train
    4. Call find_optimal_threshold on fold-val to get
       threshold-optimised sensitivity and specificity
    5. Compute AUROC on fold-val probabilities

    This ensures the cross-validation estimate reflects
    the actual evaluation pipeline — not a fixed-threshold
    approximation.

    Parameters
    ----------
    models : dict
        Dictionary mapping model name to fitted estimator.
        Each model is cloned before fitting per fold.
    X_train : pd.DataFrame
        Training set features (pre-split, unscaled).
    y_train : pd.Series
        Training set binary labels.
    n_folds : int
        Number of stratified folds (default 5).
    random_state : int
        Random seed for fold generation.

    Returns
    -------
    dict
        Keys are model names. Values are dicts containing:
            sensitivity_mean : float
            sensitivity_std  : float
            specificity_mean : float
            specificity_std  : float
            auroc_mean       : float
            auroc_std        : float
            fold_details     : list of per-fold result dicts
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state
    )

    results = {}

    for name, model in models.items():
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), 1
        ):
            # ── Split into fold-train and fold-val ──────────────
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            y_fold_val   = y_train.iloc[val_idx]

            # ── Fresh scaler per fold — no leakage ──────────────
            fold_scaler = StandardScaler()
            X_fold_train_scaled = pd.DataFrame(
                fold_scaler.fit_transform(X_fold_train),
                columns=X_train.columns,
                index=X_fold_train.index
            )
            X_fold_val_scaled = pd.DataFrame(
                fold_scaler.transform(X_fold_val),
                columns=X_train.columns,
                index=X_fold_val.index
            )

            # ── Clone and train on fold-train ───────────────────
            fold_model = clone(model)
            fold_model.fit(X_fold_train_scaled, y_fold_train)

            # ── Threshold-optimised evaluation on fold-val ──────
            thresh_result = find_optimal_threshold(
                fold_model, X_fold_val_scaled, y_fold_val
            )

            # ── AUROC on fold-val ───────────────────────────────
            fold_proba = fold_model.predict_proba(
                X_fold_val_scaled
            )[:, 1]
            fold_auroc = roc_auc_score(y_fold_val, fold_proba)

            fold_metrics.append({
                'fold':        fold_idx,
                'sensitivity': thresh_result['sensitivity'],
                'specificity': thresh_result['specificity'],
                'auroc':       round(fold_auroc, 4),
                'threshold':   thresh_result['threshold'],
                'meets_floor': thresh_result['meets_criterion']
            })

        # ── Aggregate across folds ──────────────────────────────
        sens_vals = [f['sensitivity'] for f in fold_metrics]
        spec_vals = [f['specificity'] for f in fold_metrics]
        auroc_vals = [f['auroc'] for f in fold_metrics]

        results[name] = {
            'sensitivity_mean': round(np.mean(sens_vals), 4),
            'sensitivity_std':  round(np.std(sens_vals), 4),
            'specificity_mean': round(np.mean(spec_vals), 4),
            'specificity_std':  round(np.std(spec_vals), 4),
            'auroc_mean':       round(np.mean(auroc_vals), 4),
            'auroc_std':        round(np.std(auroc_vals), 4),
            'fold_details':     fold_metrics
        }

    return results


def evaluate_model(name: str, model, X_test: pd.DataFrame,
                   y_test: pd.Series,
                   label_col: pd.Series | None = None) -> dict:
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
    print(f"\n{SEPARATOR}")
    print(f"  {name.upper()} [{status}]")
    print(f"{SEPARATOR}")
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


def select_best_model(reports: list[dict], models: dict) -> dict:
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
    print(f"\n{SEPARATOR}")
    print("  MODEL SELECTION — NATURAL SELECTION FRAMEWORK")
    print(f"{SEPARATOR}")

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