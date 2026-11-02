"""Classifiers, CV protocols and metrics (no hyper-parameter tuning: fixed, documented settings)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 0
N_JOBS = 4          # keep the laptop responsive


def classifiers():
    return {
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced",
                                                       random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                               class_weight="balanced_subsample",
                                               n_jobs=N_JOBS, random_state=SEED),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    }


def feature_cols(df: pd.DataFrame, exclude_prefixes=("meta_", "label_"), extra_exclude=()):
    skip = {"experiment_id", "block", "t_start_s", *extra_exclude}
    return [c for c in df.columns if c not in skip and not c.startswith(exclude_prefixes)]


def cv_predict(model, X, y, folds, train_stride=1):
    """folds: list of (train_idx, test_idx). Returns out-of-fold predictions (-1 = never tested).
    train_stride=2 keeps every 2nd TRAINING window (50 %-overlap windows are redundant); test
    windows are never subsampled."""
    pred = np.full(len(y), -1, dtype=int)
    for tr, te in folds:
        tr = tr[::train_stride]
        if len(np.unique(y[tr])) < 2:
            # degenerate fold (single class in training) -> predict that class
            pred[te] = y[tr][0]
            continue
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def summarize(y, pred, labels=None):
    ok = pred >= 0
    y, pred = y[ok], pred[ok]
    labels = labels if labels is not None else np.unique(np.r_[y, pred])
    cm = confusion_matrix(y, pred, labels=labels)
    per_class = {int(l): float(cm[i, i] / cm[i].sum()) if cm[i].sum() else np.nan
                 for i, l in enumerate(labels)}
    return {"accuracy": accuracy_score(y, pred),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "macro_F1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
            "per_class_recall": per_class, "cm": cm, "labels": list(labels), "n": int(len(y))}


def majority_vote(groups, pred):
    s = pd.Series(pred, index=groups)
    return s.groupby(level=0).agg(lambda v: np.bincount(v[v >= 0]).argmax() if (v >= 0).any() else -1)
