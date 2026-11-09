"""Shared evaluation metrics for the importance and mistranscription detectors."""

import numpy as np


def top_k_precision(labels, scores, meetings, ks=(10, 20, 30, 40, 50)) -> dict:
    """
    Precision of the k highest-scoring lines in each meeting, matching how lines are picked for clarification.

    For each k, returns:
        mean      average precision across meetings
        std       standard deviation of precision across meetings
        baseline  average share of positive lines in a meeting, i.e. the expected precision of random picks

    Tied scores go to the line that comes first, so the result is the same on every run.
    """
    labels = np.asarray(labels).ravel().astype(bool)
    scores = np.asarray(scores).ravel().astype(float)
    meetings = np.asarray(meetings).ravel()
    if not len(labels) == len(scores) == len(meetings):
        raise ValueError(f"Length mismatch: labels {len(labels)}, scores {len(scores)}, meetings {len(meetings)}")

    unique_meetings = np.unique(meetings)
    baseline = float(np.mean([labels[meetings == m].mean() for m in unique_meetings]))

    results = {}
    for k in ks:
        precisions = []
        for meeting in unique_meetings:
            idxs = np.flatnonzero(meetings == meeting)
            top = idxs[np.argsort(-scores[idxs], kind="stable")[:k]]
            precisions.append(labels[top].mean())
        results[k] = {"mean": float(np.mean(precisions)), "std": float(np.std(precisions)), "baseline": baseline}
    return results
