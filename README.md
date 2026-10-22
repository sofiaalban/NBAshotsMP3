# ¿Entra o no entra? · SVM lineal vs kernel RBF sobre tiros de la NBA

Mini-proyecto 3 · *Métodos y modelos matemáticos para IA* · Pontificia Universidad Javeriana Cali.

Clasificación binaria de **enceste (y = +1) vs fallo (y = −1)** con Máquinas de Soporte Vectorial,
comparando una frontera **lineal** contra una **no lineal (kernel RBF)**, con búsqueda de hiperparámetros
por **GridSearchCV** y una interfaz interactiva en Streamlit + Plotly.

## Presentación

Diapositivas de la sustentación: [`presentacion/MP3_SVM_NBA.pptx`](presentacion/MP3_SVM_NBA.pptx)
(incluye notas del orador).

## Datos

- Fuente: *NBA shot logs* (temporada 2014-15, datos de tracking SportVU), publicado en Kaggle
  (`dansbecker/nba-shot-logs`). Incluido en el repositorio como `data/shot_logs.csv`
  (128 069 tiros × 21 columnas). Otra ubicación: variable de entorno `SHOTS_CSV`.
- Limpieza (`ml.load_clean`): se eliminan filas con `SHOT_CLOCK` nulo, `TOUCH_TIME < 0` y tiros con
  distancia incoherente con su tipo (triple a < 22 ft o doble a > 23.75 ft) → 120 921 tiros.
- Variables (d = 6): `SHOT_DIST`, `CLOSE_DEF_DIST`, `SHOT_CLOCK`, `DRIBBLES`, `TOUCH_TIME`, `PERIOD`.
  Se excluyen por **fuga de información** `SHOT_RESULT`, `PTS`, `W`, `FINAL_MARGIN`, e identificadores.
- Muestra estratificada de 20 000 tiros (semilla 42), split 75/25 estratificado. Estandarización z-score
  dentro de un `Pipeline` (sin leakage en la validación cruzada).

## Ejecutar (PowerShell)

```powershell
git clone https://github.com/sofiaalban/NBAshotsMP3.git
cd NBAshotsMP3
python -m venv $HOME\.venvs\mp3-svm
& $HOME\.venvs\mp3-svm\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py          # experimento offline (~10–20 min); deja los resultados en artifacts/
streamlit run app.py     # interfaz en http://localhost:8501
```

`artifacts/` ya viene con los resultados del experimento, así que `train.py` solo hace falta para
reproducirlos (misma semilla ⇒ mismas métricas). Para una prueba rápida: `$env:MP3_SAMPLE=3000; python train.py`.

**Desde el celular** (misma red Wi-Fi): `streamlit run app.py --server.address 0.0.0.0` y abrir
`http://<IP-del-PC>:8501` (la IP sale con `ipconfig`).

## Experimento (`train.py`)

| | Grilla | Validación | Criterio |
|---|---|---|---|
| SVM lineal | C ∈ {0.01, 0.1, 1, 10} | 5-fold estratificado | ROC-AUC (también accuracy y F1) |
| SVM RBF | C ∈ {0.1, 1, 10, 100} × γ ∈ {0.001, 0.01, 0.1, 1, scale} | 5-fold estratificado | ROC-AUC |
| Baseline | clase mayoritaria (siempre «fallo») | — | — |

Se reportan accuracy, precisión, recall, F1, ROC-AUC, matriz de confusión, accuracy de train,
tiempos de entrenamiento y predicción, número de vectores de soporte y una curva de tiempo vs n.

## Interfaz (`app.py`)

1. **Datos** — embudo de limpieza, variables, heatmap de % de acierto (distancia × defensor: paradoja de Simpson).
2. **Fundamento** — formulación (margen, margen blando, dual, kernel RBF) y conexión teoría ↔ aplicación.
3. **Lineal vs RBF** — métricas en test, matrices de confusión, curvas ROC, tiempo vs n.
4. **GridSearch** — heatmaps de AUC-CV, sobreajuste y tiempo por combinación de C y γ.
5. **Laboratorio 2D** — SVM reentrenada en vivo sobre 2 variables, con frontera, margen y vectores de soporte.
6. **Simulador de tiro** — predicción de ambos modelos para un tiro descrito por el usuario y casos de prueba.

## Archivos

| Archivo | Contenido |
|---|---|
| `ml.py` | Carga, limpieza, muestreo, pipelines y laboratorio 2D (sin Streamlit) |
| `train.py` | GridSearch, evaluación y guardado de artefactos |
| `app.py` | Interfaz (Streamlit + Plotly) |
| `artifacts/` | `results.json`, `linear.joblib`, `rbf.joblib` |
| `data/shot_logs.csv` | Dataset original (sin modificar) |
| `presentacion/MP3_SVM_NBA.pptx` | Diapositivas de la sustentación |

Probado con Python 3.13, scikit-learn 1.9, Streamlit 1.64, Plotly 7.1, pandas 3.0.
