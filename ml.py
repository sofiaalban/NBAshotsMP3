"""Capa de datos y modelos del prototipo (sin Streamlit).

Problema: clasificación binaria de enceste (y = +1) vs fallo (y = -1) en los
tiros de la NBA 2014-15 (dataset "NBA shot logs", Kaggle), comparando una SVM
lineal contra una SVM con kernel RBF.

`train.py` usa estas funciones para el experimento offline (GridSearch) y
`app.py` las usa para la interfaz y el laboratorio 2D en vivo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------------------
# Constantes del experimento
# --------------------------------------------------------------------------
FEATURES = ["SHOT_DIST", "CLOSE_DEF_DIST", "SHOT_CLOCK", "DRIBBLES", "TOUCH_TIME", "PERIOD"]
FEATURES_2D = ["SHOT_DIST", "CLOSE_DEF_DIST"]
TARGET = "FGM"  # 1 = encestado, 0 = fallado (la SVM lo trata como +1 / -1)

# Columnas descartadas a propósito: o son identificadores, o son información
# que solo se conoce DESPUÉS del tiro (fuga de información / data leakage).
LEAKAGE = {
    "SHOT_RESULT": "Es la etiqueta misma en texto",
    "PTS": "Puntos del tiro: 0 si falló → revela la etiqueta",
    "W": "Resultado final del partido (posterior al tiro)",
    "FINAL_MARGIN": "Diferencia final del partido (posterior al tiro)",
}
IDENTIFIERS = ["GAME_ID", "MATCHUP", "player_name", "player_id",
               "CLOSEST_DEFENDER", "CLOSEST_DEFENDER_PLAYER_ID"]

LABELS = {
    "SHOT_DIST": "Distancia al aro (ft)",
    "CLOSE_DEF_DIST": "Distancia del defensor más cercano (ft)",
    "SHOT_CLOCK": "Reloj de posesión (s)",
    "DRIBBLES": "Botes antes del tiro",
    "TOUCH_TIME": "Tiempo con el balón (s)",
    "PERIOD": "Periodo (5+ = prórroga)",
}
# Rango de los controles del simulador (percentil ~0.5 a ~99.5 de los datos limpios).
RANGES = {
    "SHOT_DIST": (0.0, 30.0, 0.5),
    "CLOSE_DEF_DIST": (0.0, 15.0, 0.1),
    "SHOT_CLOCK": (0.0, 24.0, 0.5),
    "DRIBBLES": (0, 20, 1),
    "TOUCH_TIME": (0.0, 20.0, 0.1),
    "PERIOD": (1, 5, 1),
}

SEED = 42
SAMPLE_SIZE = 20_000   # SVC entrena en ~O(n²); 120k filas es inviable en tiempo razonable
TEST_SIZE = 0.25
SAMPLE_2D = 2_000      # laboratorio en vivo: debe reentrenar en < 1 s

_HERE = Path(__file__).resolve().parent
# Repositorio: data/shot_logs.csv · carpeta del curso: ../db/shot_logs.csv · o SHOTS_CSV.
_CANDIDATES = [_HERE / "data" / "shot_logs.csv", _HERE.parent / "db" / "shot_logs.csv"]
DATA_PATH = Path(os.environ.get("SHOTS_CSV")
                 or next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0]))
ARTIFACTS = _HERE / "artifacts"


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@dataclass
class Clean:
    data: pd.DataFrame                  # filas limpias, todas las columnas
    funnel: list[tuple[str, int]]       # (paso, filas que quedan)


def load_clean(path: Path = DATA_PATH) -> Clean:
    """Carga el CSV y aplica la limpieza, registrando cuántas filas deja cada paso."""
    df = pd.read_csv(path)
    funnel = [("Datos crudos", len(df))]

    # Reloj de posesión vacío: ocurre cuando quedan < 24 s en el periodo y el reloj se apaga.
    df = df.dropna(subset=["SHOT_CLOCK"])
    funnel.append(("Sin SHOT_CLOCK nulo", len(df)))

    # Tiempo de posesión negativo: error de medición del tracking.
    df = df[df["TOUCH_TIME"] >= 0]
    funnel.append(("Sin TOUCH_TIME negativo", len(df)))

    # Distancia incoherente con el tipo de tiro (triple a < 22 ft o doble a > 23.75 ft):
    # SHOT_DIST mal medida, y es nuestra variable más importante.
    bad = ((df["PTS_TYPE"] == 3) & (df["SHOT_DIST"] < 22)) | \
          ((df["PTS_TYPE"] == 2) & (df["SHOT_DIST"] > 23.75))
    df = df[~bad]
    funnel.append(("Distancia coherente con PTS_TYPE", len(df)))

    return Clean(data=df.reset_index(drop=True), funnel=funnel)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def make_split(clean: pd.DataFrame, features: list[str] = FEATURES,
               n: int = SAMPLE_SIZE, seed: int = SEED) -> Split:
    """Muestra estratificada de n filas y split train/test estratificado (75/25)."""
    sample, _ = train_test_split(clean, train_size=n, stratify=clean[TARGET], random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        sample[features], sample[TARGET], test_size=TEST_SIZE,
        stratify=sample[TARGET], random_state=seed)
    return Split(X_tr, X_te, y_tr, y_te)


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def make_pipeline(kernel: str, C: float = 1.0, gamma: float | str = "scale") -> Pipeline:
    """StandardScaler → SVC. El escalado va DENTRO del pipeline para que en la
    validación cruzada se ajuste solo con el fold de entrenamiento (sin leakage)."""
    svc = SVC(kernel=kernel, C=C, gamma=gamma, cache_size=1000)
    return Pipeline([("scaler", StandardScaler()), ("svm", svc)])


def fg_heatmap(clean: pd.DataFrame) -> pd.DataFrame:
    """% de acierto por (distancia al aro × distancia del defensor) y conteos."""
    d = clean.assign(
        dist=pd.cut(clean["SHOT_DIST"], [0, 3, 6, 10, 16, 22, 26, 50], include_lowest=True),
        defd=pd.cut(clean["CLOSE_DEF_DIST"], [0, 2, 4, 6, 10, 60], include_lowest=True),
    )
    return d.groupby(["dist", "defd"], observed=True)[TARGET].agg(["mean", "size"]).reset_index()


@dataclass
class Lab2D:
    model: Pipeline
    X: np.ndarray            # puntos de entrenamiento (unidades originales)
    y: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    f: np.ndarray            # f(x) evaluada en la rejilla
    support: np.ndarray      # índices de los vectores de soporte
    train_acc: float
    test_acc: float


def lab_2d(clean: pd.DataFrame, kernel: str, C: float, gamma: float,
           n: int = SAMPLE_2D, seed: int = SEED, resolution: int = 120) -> Lab2D:
    """SVM sobre (SHOT_DIST, CLOSE_DEF_DIST) para visualizar la frontera en vivo."""
    # Recortamos defensores > 15 ft (outliers) solo para que el plano sea legible.
    view = clean[clean["CLOSE_DEF_DIST"] <= 15]
    s = make_split(view, FEATURES_2D, n=int(n / (1 - TEST_SIZE)) + 1, seed=seed)
    model = make_pipeline(kernel, C=C, gamma=gamma).fit(s.X_train, s.y_train)

    gx = np.linspace(0, 30, resolution)
    gy = np.linspace(0, 15, resolution)
    xx, yy = np.meshgrid(gx, gy)
    grid = pd.DataFrame({FEATURES_2D[0]: xx.ravel(), FEATURES_2D[1]: yy.ravel()})
    f = model.decision_function(grid).reshape(xx.shape)

    return Lab2D(
        model=model, X=s.X_train.to_numpy(), y=s.y_train.to_numpy(),
        grid_x=gx, grid_y=gy, f=f, support=model[-1].support_,
        train_acc=model.score(s.X_train, s.y_train), test_acc=model.score(s.X_test, s.y_test),
    )
