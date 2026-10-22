"""Experimento offline: GridSearchCV de SVM lineal y RBF + evaluación en test.

Se corre UNA vez antes de la demo (tarda varios minutos) y deja en `artifacts/`:
  - results.json   métricas, tablas de la búsqueda, tiempos, curvas ROC, etc.
  - linear.joblib  mejor SVM lineal (pipeline StandardScaler → SVC)
  - rbf.joblib     mejor SVM RBF

Uso:  python train.py            (muestra de 20 000 tiros)
      $env:MP3_SAMPLE=3000; python train.py   (prueba rápida)
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import ml

# Grillas logarítmicas (órdenes de magnitud), como en las diapositivas del curso.
GRIDS = {
    "linear": {"svm__C": [0.01, 0.1, 1, 10]},
    "rbf": {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [0.001, 0.01, 0.1, 1, "scale"]},
}
# Criterio de selección: ROC-AUC. Las clases están casi balanceadas (45/55) y el AUC
# mide qué tan bien f(x) ORDENA encestes sobre fallos sin depender del umbral 0.
# Accuracy y F1 se registran también para poder discutirlos.
SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1"}
REFIT = "roc_auc"
CV_FOLDS = 5
SCALING_SIZES = [1_000, 2_000, 4_000, 8_000, 15_000]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def evaluate(model, s: ml.Split) -> dict:
    """Métricas en test + tiempos de predicción + tamaño del modelo."""
    t = time.perf_counter()
    pred = model.predict(s.X_test)
    predict_s = time.perf_counter() - t
    if hasattr(model, "decision_function"):
        score = model.decision_function(s.X_test)
    else:  # Dummy: no tiene f(x); usamos la probabilidad (constante → AUC 0.5)
        score = model.predict_proba(s.X_test)[:, 1]
    fpr, tpr, _ = roc_curve(s.y_test, score)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))  # curva liviana para la app
    out = {
        "accuracy": accuracy_score(s.y_test, pred),
        "precision": precision_score(s.y_test, pred, zero_division=0),
        "recall": recall_score(s.y_test, pred),
        "f1": f1_score(s.y_test, pred),
        "roc_auc": roc_auc_score(s.y_test, score),
        "train_accuracy": accuracy_score(s.y_train, model.predict(s.X_train)),
        "confusion": confusion_matrix(s.y_test, pred, labels=[0, 1]).tolist(),
        "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "predict_seconds": predict_s,
    }
    if hasattr(model, "named_steps"):
        svm = model.named_steps["svm"]
        out["n_support"] = int(svm.n_support_.sum())
        out["support_fraction"] = float(svm.n_support_.sum() / len(s.X_train))
    return out


def search(kernel: str, s: ml.Split) -> tuple[GridSearchCV, list[dict]]:
    log(f"GridSearch {kernel}: {GRIDS[kernel]}")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=ml.SEED)
    gs = GridSearchCV(ml.make_pipeline(kernel), GRIDS[kernel], scoring=SCORING, refit=REFIT,
                      cv=cv, n_jobs=-1, return_train_score=True, verbose=1)
    t = time.perf_counter()
    gs.fit(s.X_train, s.y_train)
    log(f"  listo en {time.perf_counter() - t:.1f} s · mejor {gs.best_params_} "
        f"AUC-CV={gs.best_score_:.4f}")
    r = gs.cv_results_
    table = []
    for i, params in enumerate(r["params"]):
        table.append({
            "C": params["svm__C"],
            "gamma": str(params.get("svm__gamma", "—")),
            "cv_roc_auc": r["mean_test_roc_auc"][i], "cv_roc_auc_std": r["std_test_roc_auc"][i],
            "cv_accuracy": r["mean_test_accuracy"][i], "cv_f1": r["mean_test_f1"][i],
            "train_roc_auc": r["mean_train_roc_auc"][i],
            "mean_fit_seconds": r["mean_fit_time"][i],
            "rank": int(r["rank_test_roc_auc"][i]),
        })
    return gs, table


def timed_fit(model, X, y) -> float:
    t = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - t


def main() -> None:
    n = int(os.environ.get("MP3_SAMPLE", ml.SAMPLE_SIZE))
    log("Cargando y limpiando datos…")
    clean = ml.load_clean()
    s = ml.make_split(clean.data, n=n)
    log(f"  {len(clean.data)} filas limpias · muestra {n} · train {len(s.X_train)} · "
        f"test {len(s.X_test)} · d = {len(ml.FEATURES)}")

    results: dict = {
        "dataset": {
            "funnel": clean.funnel, "n_clean": len(clean.data), "sample": n,
            "n_train": len(s.X_train), "n_test": len(s.X_test),
            "features": ml.FEATURES, "dimensions": len(ml.FEATURES),
            "positive_rate_clean": float(clean.data[ml.TARGET].mean()),
            "positive_rate_train": float(s.y_train.mean()),
        },
        "config": {"grids": {k: {p: [str(v) for v in vals] for p, vals in g.items()}
                             for k, g in GRIDS.items()},
                   "cv_folds": CV_FOLDS, "refit": REFIT, "seed": ml.SEED,
                   "gamma_scale_value": 1 / len(ml.FEATURES)},  # tras estandarizar Var(X)=1
        "models": {}, "search": {},
    }

    # Baseline: predecir siempre la clase mayoritaria (fallo).
    dummy = DummyClassifier(strategy="most_frequent")
    fit_s = timed_fit(dummy, s.X_train, s.y_train)
    results["models"]["dummy"] = {"params": {"strategy": "most_frequent"},
                                  "fit_seconds": fit_s, **evaluate(dummy, s)}

    best = {}
    for kernel in ("linear", "rbf"):
        gs, table = search(kernel, s)
        results["search"][kernel] = {"table": table, "best_params": gs.best_params_,
                                     "best_cv_roc_auc": gs.best_score_,
                                     "total_seconds": float(np.sum(gs.cv_results_["mean_fit_time"])
                                                            * CV_FOLDS)}
        # Re-entrenamos el mejor modelo aparte para medir su tiempo de ajuste limpio (1 núcleo).
        model = ml.make_pipeline(kernel)
        model.set_params(**gs.best_params_)
        fit_s = timed_fit(model, s.X_train, s.y_train)
        log(f"  refit {kernel}: {fit_s:.2f} s")
        results["models"][kernel] = {"params": {k.replace("svm__", ""): v
                                                for k, v in gs.best_params_.items()},
                                     "fit_seconds": fit_s, **evaluate(model, s)}
        joblib.dump(model, ml.ARTIFACTS / f"{kernel}.joblib")
        best[kernel] = gs.best_params_

    # Costo computacional: tiempo de entrenamiento vs tamaño de la muestra.
    log("Curva de tiempo vs n…")
    scaling = {"n": [], "linear": [], "rbf": []}
    for size in [m for m in SCALING_SIZES if m <= len(s.X_train)]:
        Xs, ys = s.X_train.iloc[:size], s.y_train.iloc[:size]
        scaling["n"].append(size)
        for kernel in ("linear", "rbf"):
            model = ml.make_pipeline(kernel)
            model.set_params(**best[kernel])
            scaling[kernel].append(timed_fit(model, Xs, ys))
        log(f"  n={size}: lineal {scaling['linear'][-1]:.2f} s · rbf {scaling['rbf'][-1]:.2f} s")
    results["scaling"] = scaling

    with open(ml.ARTIFACTS / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    for k in ("dummy", "linear", "rbf"):
        m = results["models"][k]
        log(f"{k:7s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f} "
            f"fit={m['fit_seconds']:.2f}s")
    log("Artefactos guardados en " + str(ml.ARTIFACTS))


if __name__ == "__main__":
    ml.ARTIFACTS.mkdir(exist_ok=True)
    main()
