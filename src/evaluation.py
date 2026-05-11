from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score


def classify(score: float, high_thresh: float = 0.5, low_thresh: float = 0.3) -> str:
    """transforme un score continu en classe (Good Fit / Potential Fit / No Fit)"""
    if score >= high_thresh:
        return "Good Fit"
    if score >= low_thresh:
        return "Potential Fit"
    return "No Fit"


def find_best_thresholds(
    scores: Sequence[float],
    labels: Sequence[str],
) -> tuple[float, float, float]:
    """grid search des meilleurs seuils (high, low) maximisant le F1 weighted"""
    best_f1 = 0.0
    best_high = 0.5
    best_low = 0.3

    for high in np.arange(0.1, 0.95, 0.05):
        for low in np.arange(0.05, high, 0.05):
            preds = [classify(s, high, low) for s in scores]
            f1 = f1_score(labels, preds, average='weighted', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_high = float(high)
                best_low = float(low)

    return best_high, best_low, best_f1


def compute_f1(
    scores: Sequence[float],
    labels: Sequence[str],
    high_thresh: float,
    low_thresh: float,
) -> float:
    """calcule le F1 weighted pour des seuils donnés"""
    preds = [classify(s, high_thresh, low_thresh) for s in scores]
    return f1_score(labels, preds, average='weighted', zero_division=0)
