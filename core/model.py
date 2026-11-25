"""Model training, evaluation, tuning and explanation helpers (all cached)."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from core.data import get_split, pretty

# sklearn defaults
BASE_SK = dict(criterion="gini", max_depth=None, min_samples_split=2,
               min_samples_leaf=1, class_weight=None, ccp_alpha=0.0)

# App-level settings (model params + threshold + split)
DEFAULT_CFG = dict(BASE_SK, max_depth=6, min_samples_split=50, min_samples_leaf=20,
                   class_weight="balanced", threshold=0.5, test_size=0.2, seed=42)

# Same configurations as the notebook
PRESETS = {
    "Basic (default)": dict(),
    "max_depth=5": dict(max_depth=5),
    "entropy, depth=5": dict(criterion="entropy", max_depth=5),
    "depth=5, leaf=20": dict(max_depth=5, min_samples_leaf=20),
    "depth=5, balanced": dict(max_depth=5, class_weight="balanced"),
    "depth=6, leaf=20, split=50, balanced": dict(max_depth=6, min_samples_leaf=20,
                                                 min_samples_split=50,
                                                 class_weight="balanced"),
}

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def sk_params(cfg: dict) -> dict:
    return {k: cfg[k] for k in BASE_SK}


def preset_to_cfg(params: dict, keep: dict | None = None) -> dict:
    cfg = dict(DEFAULT_CFG)
    if keep:
        for k in ("threshold", "test_size", "seed"):
            cfg[k] = keep[k]
    cfg.update(BASE_SK)
    cfg.update(params)
    return cfg


def metrics_at(y_true, prob, thr: float = 0.5) -> dict:
    pred = (np.asarray(prob) > thr).astype(int)
    auc = roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": auc,
    }


@st.cache_resource(show_spinner=False)
def fit_model(params: dict, test_size: float, seed: int) -> dict:
    X_train, X_test, y_train, y_test = get_split(test_size, seed)
    model = DecisionTreeClassifier(random_state=seed, **params).fit(X_train, y_train)
    return {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "prob_train": model.predict_proba(X_train)[:, 1],
        "prob_test": model.predict_proba(X_test)[:, 1],
        "features": list(X_train.columns),
    }


def fit_from_cfg(cfg: dict) -> dict:
    return fit_model(sk_params(cfg), cfg["test_size"], cfg["seed"])


@st.cache_data(show_spinner=False)
def preset_table(test_size: float, seed: int) -> pd.DataFrame:
    rows = []
    for name, p in PRESETS.items():
        full = {**BASE_SK, **p}
        r = fit_model(full, test_size, seed)
        m = metrics_at(r["y_test"], r["prob_test"], 0.5)
        rows.append({
            "Model": name,
            "Train Acc": accuracy_score(r["y_train"], (r["prob_train"] > 0.5).astype(int)),
            "Test Acc": m["Accuracy"], "Precision": m["Precision"], "Recall": m["Recall"],
            "F1": m["F1"], "ROC-AUC": m["ROC-AUC"],
            "Depth": r["model"].get_depth(), "Leaves": r["model"].get_n_leaves(),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def depth_sweep(params: dict, test_size: float, seed: int, max_depth: int = 20) -> pd.DataFrame:
    rows = []
    for d in range(1, max_depth + 1):
        r = fit_model({**params, "max_depth": d}, test_size, seed)
        mt = metrics_at(r["y_test"], r["prob_test"])
        rows.append({
            "max_depth": d,
            "Train Acc": accuracy_score(r["y_train"], (r["prob_train"] > 0.5).astype(int)),
            "Test Acc": mt["Accuracy"], "Test F1": mt["F1"], "Test Recall": mt["Recall"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cv_scores(params: dict, test_size: float, seed: int, folds: int = 5) -> pd.DataFrame:
    X_train, _, y_train, _ = get_split(test_size, seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(
        DecisionTreeClassifier(random_state=seed, **params), X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    out = pd.DataFrame({
        "Accuracy": res["test_accuracy"], "Precision": res["test_precision"],
        "Recall": res["test_recall"], "F1": res["test_f1"], "ROC-AUC": res["test_roc_auc"],
    })
    out.index = [f"Fold {i+1}" for i in range(folds)]
    return out


GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_samples_split": [2, 20, 50],
    "min_samples_leaf": [1, 10, 20, 40],
    "class_weight": [None, "balanced"],
}


@st.cache_resource(show_spinner=False)
def run_grid_search(test_size: float, seed: int, scoring: str) -> dict:
    X_train, X_test, y_train, y_test = get_split(test_size, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(DecisionTreeClassifier(random_state=seed), GRID,
                      scoring=scoring, cv=cv, n_jobs=1)
    gs.fit(X_train, y_train)
    res = pd.DataFrame(gs.cv_results_)
    top = (res.sort_values("rank_test_score").head(10)
           [["rank_test_score", "mean_test_score", "std_test_score", "param_criterion",
             "param_max_depth", "param_min_samples_split", "param_min_samples_leaf",
             "param_class_weight"]]
           .rename(columns=lambda c: c.replace("param_", "")).reset_index(drop=True))
    top["class_weight"] = top["class_weight"].astype(str)
    best = gs.best_estimator_
    prob = best.predict_proba(X_test)[:, 1]
    return {
        "best_params": gs.best_params_, "best_score": gs.best_score_, "top": top,
        "test_metrics": metrics_at(y_test, prob), "n_fits": len(res) * 5,
    }


def explain_path(model: DecisionTreeClassifier, x_row: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rules the tree applied to one customer, plus info about the leaf reached."""
    tree = model.tree_
    names = list(x_row.columns)
    node_ids = model.decision_path(x_row).indices
    rows = []
    for nid in node_ids:
        if tree.children_left[nid] == tree.children_right[nid]:  # leaf
            continue
        f = tree.feature[nid]
        thr = tree.threshold[nid]
        val = float(x_row.iloc[0, f])
        left = val <= thr
        rows.append({
            "Feature": pretty(names[f]), "Customer value": round(val, 2),
            "Rule": f"{'≤' if left else '>'} {thr:.2f}",
            "Went": "left" if left else "right",
        })
    leaf = int(node_ids[-1])
    info = {"leaf_samples": int(tree.n_node_samples[leaf]), "depth": len(rows)}
    return pd.DataFrame(rows), info


def predict_proba_records(model, records: pd.DataFrame, features: list) -> np.ndarray:
    return model.predict_proba(records[features])[:, 1]
