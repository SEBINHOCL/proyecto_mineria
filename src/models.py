#Modelos de aprendizaje no supervisado.

# Expone funciones de alto nivel para entrenar, evaluar y guardar
# los modelos de clustering usados en el dashboard.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score,
    log_loss, roc_auc_score, silhouette_score,
)

from .config import MODELS_DIR


# ========================================================================
# Helpers de persistencia
# ========================================================================

def save_model(model: Any, name: str) -> Path:
    # Guarda un modelo entrenado como archivo .pkl en data/models/.
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> Any:
    # Carga un modelo desde data/models/<name>.pkl.
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo '{name}' no encontrado en {MODELS_DIR}. "
            "Ejecuta primero el script de entrenamiento."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ========================================================================
# Seleccion de k
# ========================================================================

def find_optimal_k(
    X_scaled: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict[str, list]:

    # Calcula inertia y silhouette para un rango de valores de k

    results: dict[str, list] = {"k": [], "inertia": [], "silhouette": []}

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)
        results["silhouette"].append(silhouette_score(X_scaled, labels))

    return results


# ========================================================================
# Clustering de partidos
# ========================================================================

def train_match_clustering(
    X_scaled: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> KMeans:
    # Entrena K-Means sobre la matriz de features de partidos
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    return km


def get_match_cluster_profiles(
    matches: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    # Retorna el perfil promedio de cada cluster de partidos.

    profile = matches.groupby("cluster")[features].mean().round(2)
    profile["n_partidos"] = matches["cluster"].value_counts().sort_index()
    return profile


# ========================================================================
# Clustering de equipos
# ========================================================================

def train_team_clustering(
    X_scaled: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> KMeans:
    # Entrena K-Means sobre la matriz de features de equipos.
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    return km


# ========================================================================
# PCA para visualizacion
# ========================================================================

def reduce_to_2d(
    X_scaled: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, PCA]:
    # Reduce la matriz de features a 2 componentes principales.

    pca    = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df_2d  = pd.DataFrame(coords, columns=["PC1", "PC2"], index=X_scaled.index)
    return df_2d, pca


# ========================================================================
# Agglomerative Clustering
# ========================================================================

def train_agglomerative(
    X_scaled: pd.DataFrame,
    k: int,
    linkage: str = "ward",
) -> AgglomerativeClustering:
    # Entrena Agglomerative Clustering con enlace Ward.
    ag = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    ag.fit(X_scaled)
    return ag


# ========================================================================
# Modelos supervisados
# ========================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 0.1,
) -> LogisticRegression:
    # Regresión Logistica con regularizacion L2. C bajo = más regularizacion
    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    # Random Forest con restricciones para evitar overfitting en datasets pequeños
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def poisson_baseline_proba(
    matches_train: pd.DataFrame,
    matches_test: pd.DataFrame,
) -> np.ndarray:

    from scipy.stats import poisson

    team_attack: dict[str, float] = {}
    global_avg = (matches_train["home_goals"].mean() + matches_train["away_goals"].mean()) / 2

    for team in pd.unique(pd.concat([matches_train["home_team"], matches_train["away_team"]])):
        home = matches_train[matches_train["home_team"] == team]
        away = matches_train[matches_train["away_team"] == team]
        total = len(home) + len(away)
        if total == 0:
            continue
        gf = home["home_goals"].sum() + away["away_goals"].sum()
        team_attack[team] = gf / total

    probas = []
    for _, row in matches_test.iterrows():
        lam_h = team_attack.get(row["home_team"], global_avg)
        lam_a = team_attack.get(row["away_team"], global_avg)
        lam   = lam_h + lam_a
        p_over = 1.0 - sum(poisson.pmf(k, lam) for k in range(3))
        probas.append(p_over)

    return np.array(probas)


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str = "Model",
) -> dict:
    # Evalua un clasificador sklearn y retorna métricas completas.
    # Incluye accuracy, F1, ROC-AUC, Brier Score y log-loss.

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "brier": round(brier_score_loss(y_test, y_proba), 4),
        "log_loss": round(log_loss(y_test, y_proba), 4),
    }


def evaluate_proba_array(
    y_proba: np.ndarray,
    y_test: pd.Series,
    name: str = "Baseline",
) -> dict:
    # Evalua un array de probabilidades (ej: baseline Poisson)
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "brier": round(brier_score_loss(y_test, y_proba), 4),
        "log_loss": round(log_loss(y_test, y_proba), 4),
    }


def get_pca_loadings(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    # Retorna la contribucion de cada feature original a PC1 y PC2.
    # Util para interpretar que significa cada eje en el grafico.

    return pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=["PC1", "PC2"],
    ).round(3)
