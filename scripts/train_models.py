# Fase 3 - Entrenamiento de todos los modelos ML.

# No supervisado : K-Means y Agglomerative Clustering
# Supervisado : Regresion Logistica, Random Forest y Baseline Poisson
# Target : Over/Under 2.5 goles

# Usar: python scripts/train_models.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.db import get_matches, get_goals
from src.features import (
    build_match_features, scale_match_features, MATCH_FEATURES,
    build_team_features, scale_team_features, TEAM_FEATURES,
    build_temporal_features,
    build_supervised_features, scale_supervised_features, SUPERVISED_FEATURES, SUPERVISED_TARGET,
)
from src.models import (
    find_optimal_k,
    train_match_clustering, get_match_cluster_profiles,
    train_team_clustering,
    train_agglomerative,
    reduce_to_2d, get_pca_loadings,
    train_logistic_regression,
    train_random_forest,
    poisson_baseline_proba,
    evaluate_classifier,
    evaluate_proba_array,
    save_model,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

SEP = "=" * 50


# ========================================================================
# 1- Cargar datos
# ========================================================================
print(f"{SEP}\nCargando datos\n{SEP}")
matches_raw = get_matches()
goals_raw   = get_goals()
print(f"  Partidos totales : {len(matches_raw)}")
print(f"  Goles en DB      : {len(goals_raw)}")

matches = matches_raw.dropna(subset=["home_goals", "away_goals"]).copy()
print(f"  Partidos con datos completos: {len(matches)}\n")


# ========================================================================
# 2- No supervisado | Clustering de partidos
# ========================================================================
print(f"{SEP}\nNo supervisado - Clustering de partidos\n{SEP}")

matches_feat       = build_match_features(matches)
X_match, scaler_m  = scale_match_features(matches_feat)

k_results = find_optimal_k(X_match, k_range=range(2, 9))
best_k_match = k_results["k"][k_results["silhouette"].index(max(k_results["silhouette"]))]

print("  k  | Silhouette | Davies-Bouldin")
print("  ---|------------|---------------")
for i, k in enumerate(k_results["k"]):
    from sklearn.cluster import KMeans
    km_tmp    = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_match)
    db_score  = davies_bouldin_score(X_match, km_tmp.labels_)
    sil       = k_results["silhouette"][i]
    marker    = " ← óptimo" if k == best_k_match else ""
    print(f"  {k}  | {sil:.4f}     | {db_score:.4f}{marker}")

# K-Means
km_match = train_match_clustering(X_match, k=best_k_match)
matches_feat["cluster"] = km_match.labels_
print(f"\n  K-Means k={best_k_match} entrenado.")

# Agglomerative
ag_match = train_agglomerative(X_match, k=best_k_match)
matches_feat["cluster_ag"] = ag_match.labels_

# Comparar consistencia entre métodos
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(km_match.labels_, ag_match.labels_)
print(f"  Agglomerative k={best_k_match} entrenado.")
print(f"  Consistencia entre métodos (ARI): {ari:.4f}  (1.0 = idénticos)")

# PCA para visualizacion
df_2d_match, pca_match = reduce_to_2d(X_match)

# Perfiles K-Means
profiles = get_match_cluster_profiles(matches_feat, MATCH_FEATURES)
print("\n  Perfiles por cluster (K-Means):")
print(profiles.to_string())

# Guardar
save_model(km_match,      "km_match")
save_model(scaler_m,      "scaler_match")
save_model(pca_match,     "pca_match")
save_model(k_results,     "k_results_match")
save_model(ag_match,      "ag_match")
save_model(matches_feat,  "matches_features")
print("\n  Modelos de partidos guardados.")


# ========================================================================
# 3- No supervisado | Clustering de equipos
# ========================================================================
print(f"\n{SEP}\nNo supervisado - Clustering de equipos\n{SEP}")

teams_feat         = build_team_features(matches)
X_team, scaler_t   = scale_team_features(teams_feat)

k_results_team = find_optimal_k(X_team, k_range=range(2, 7))
best_k_team    = k_results_team["k"][k_results_team["silhouette"].index(max(k_results_team["silhouette"]))]

# K-Means
km_team = train_team_clustering(X_team, k=best_k_team)
teams_feat["cluster"] = km_team.labels_

# Agglomerative
ag_team = train_agglomerative(X_team, k=best_k_team)
teams_feat["cluster_ag"] = ag_team.labels_

ari_team = adjusted_rand_score(km_team.labels_, ag_team.labels_)
print(f"  K-Means y Agglomerative k={best_k_team} entrenados.")
print(f"  Consistencia (ARI): {ari_team:.4f}")

df_2d_team, pca_team = reduce_to_2d(X_team)

print("\n  Equipos por cluster (K-Means):")
for c in sorted(teams_feat["cluster"].unique()):
    equipos = teams_feat[teams_feat["cluster"] == c].index.tolist()
    print(f"  Cluster {c}: {", ".join(sorted(equipos))}")

save_model(km_team,        "km_team")
save_model(scaler_t,       "scaler_team")
save_model(pca_team,       "pca_team")
save_model(k_results_team, "k_results_team")
save_model(ag_team,        "ag_team")
save_model(teams_feat,     "teams_features")
print("  Modelos de equipos guardados.")


# ========================================================================
# 4- No supervisado | Análisis temporal
# ========================================================================
print(f"\n{SEP}\nNo supervisado - Análisis temporal\n{SEP}")

goals_2023    = goals_raw[goals_raw["season"] == 2023].copy()
temporal_feat = build_temporal_features(goals_2023)
temporal_cols = temporal_feat.columns.tolist()

scaler_temporal = StandardScaler()
X_temp = scaler_temporal.fit_transform(temporal_feat.values)
X_temp_df = pd.DataFrame(X_temp, columns=temporal_cols, index=temporal_feat.index)

k_results_temp = find_optimal_k(X_temp_df, k_range=range(2, 7))
best_k_temp    = k_results_temp["k"][k_results_temp["silhouette"].index(max(k_results_temp["silhouette"]))]

km_temporal = train_team_clustering(X_temp_df, k=best_k_temp)
temporal_feat["cluster"] = km_temporal.labels_
print(f"  K-Means temporal k={best_k_temp}.")

save_model(km_temporal,     "km_temporal")
save_model(scaler_temporal, "scaler_temporal")
save_model(temporal_feat,   "temporal_features")
print("  Modelos temporales guardados.")


# ========================================================================
# 5- Supervisado | Preparar features
# ========================================================================
print(f"\n{SEP}\nSupervisado - Preparando features\n{SEP}")

print("  Calculando rolling features (puede tardar ~30 seg)...")
sup_df = build_supervised_features(matches, window=5)

print(f"  Partidos con features completas: {len(sup_df)}")
print(f"  Distribución target Over 2.5:")
vc = sup_df[SUPERVISED_TARGET].value_counts(normalize=True)
print(f"    Over  (1): {vc.get(1, 0):.1%}")
print(f"    Under (0): {vc.get(0, 0):.1%}")

# Split cronologico: 2023 = train, 2024 = test
train_df = sup_df[sup_df["season"] == 2023].copy()
test_df  = sup_df[sup_df["season"] == 2024].copy()

print(f"\n  Train (2023): {len(train_df)} partidos")
print(f"  Test  (2024): {len(test_df)}  partidos")

# Escalar con scaler ajustado solo en train
scaler_sup = StandardScaler()
X_train = scaler_sup.fit_transform(train_df[SUPERVISED_FEATURES])
X_test  = scaler_sup.transform(test_df[SUPERVISED_FEATURES])

X_train_df = pd.DataFrame(X_train, columns=SUPERVISED_FEATURES)
X_test_df  = pd.DataFrame(X_test,  columns=SUPERVISED_FEATURES)

y_train = train_df[SUPERVISED_TARGET].reset_index(drop=True)
y_test  = test_df[SUPERVISED_TARGET].reset_index(drop=True)


# ========================================================================
# 6- Supervisado | Entrenar y evaluar modelos
# ========================================================================
print(f"\n{SEP}\nSupervisado - Entrenando modelos\n{SEP}")

# Regresion Logistica
lr = train_logistic_regression(X_train_df, y_train)
metrics_lr = evaluate_classifier(lr, X_test_df, y_test, "Logistic Regression")

# Random Forest
rf = train_random_forest(X_train_df, y_train)
metrics_rf = evaluate_classifier(rf, X_test_df, y_test, "Random Forest")

# Baseline Poisson
poisson_probas = poisson_baseline_proba(
    train_df.rename(columns={"home_scored_5": "home_goals", "away_scored_5": "away_goals"}),
    test_df,
)
# Poisson baseline usa las columnas originales de matches para calcular tasas
poisson_probas_real = poisson_baseline_proba(
    matches[matches["season"] == 2023],
    matches[matches["season"] == 2024],
)

test_fixture_ids = test_df["fixture_id"].values
matches_2024 = matches[matches["season"] == 2024].copy()
matches_2024_feat = matches_2024[matches_2024["fixture_id"].isin(test_fixture_ids)].copy()

# Re-ordenar igual que test_df
matches_2024_feat = (
    test_df[["fixture_id"]]
    .merge(matches_2024_feat, on="fixture_id", how="left")
)
poisson_probas_aligned = poisson_baseline_proba(
    matches[matches["season"] == 2023],
    matches_2024_feat,
)
metrics_poisson = evaluate_proba_array(poisson_probas_aligned, y_test, "Poisson Baseline")

# Mostrar comparacion
print(f"\n  {"Modelo":<22} {"Accuracy":>8} {"F1":>6} {"ROC-AUC":>8} {"Brier":>6} {"LogLoss":>8}")
print(f"  {"-"*22} {"-"*8} {"-"*6} {"-"*8} {"-"*6} {"-"*8}")
for m in [metrics_poisson, metrics_lr, metrics_rf]:
    print(
        f"  {m["model"]:<22} {m["accuracy"]:>8.4f} {m["f1"]:>6.4f} "
        f"{m["roc_auc"]:>8.4f} {m["brier"]:>6.4f} {m["log_loss"]:>8.4f}"
    )

# Importancia de features (Random Forest)
feat_imp = pd.Series(rf.feature_importances_, index=SUPERVISED_FEATURES).sort_values(ascending=False)
print("\n  Importancia de features (Random Forest):")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<22} {bar} {imp:.4f}")

# Coeficientes (Regresion Logistica)
coef = pd.Series(lr.coef_[0], index=SUPERVISED_FEATURES).sort_values(key=abs, ascending=False)
print("\n  Coeficientes (Regresión Logística):")
for feat, c in coef.items():
    print(f"  {feat:<22}  {c:+.4f}")


# ========================================================================
# 7- Guardar todo
# ========================================================================
all_metrics = {
    "logistic_regression": metrics_lr,
    "random_forest":       metrics_rf,
    "poisson_baseline":    metrics_poisson,
}

save_model(lr,             "lr_over25")
save_model(rf,             "rf_over25")
save_model(scaler_sup,     "scaler_supervised")
save_model(sup_df,         "supervised_features")
save_model(all_metrics,    "supervised_metrics")

print(f"\n{SEP}\nEntrenamiento completado\n{SEP}")
print("  Archivos en data/models/:")
from src.config import MODELS_DIR
for f in sorted(MODELS_DIR.glob("*.pkl")):
    print(f"  {f.name:<35} {f.stat().st_size/1024:5.1f} KB")
