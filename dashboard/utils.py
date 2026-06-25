from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Permite importar src/ desde cualquier pag
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import get_matches, get_goals, get_top_scorers, get_summary
from src.features import (
    build_match_features, scale_match_features, MATCH_FEATURES,
    build_team_features, TEAM_FEATURES,
    build_temporal_features,
)
from src.models import load_model, reduce_to_2d, get_pca_loadings

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


@st.cache_data
def load_all_data() -> dict:
    # Carga y cachea todos los datos necesarios para el dashboard
    matches = get_matches()
    goals = get_goals()
    scorers = get_top_scorers(limit=30)
    summary = get_summary()

    matches_feat = build_match_features(matches)
    X_match, _ = scale_match_features(matches_feat)

    teams_feat = build_team_features(matches)

    goals_2023 = goals[goals["season"] == 2023].copy()
    temporal_feat = build_temporal_features(goals_2023)

    return {
        "matches": matches,
        "matches_feat": matches_feat,
        "X_match": X_match,
        "goals": goals,
        "goals_2023": goals_2023,
        "scorers": scorers,
        "summary": summary,
        "teams_feat": teams_feat,
        "temporal_feat": temporal_feat,
    }


@st.cache_resource
def load_all_models() -> dict:
    # Carga y cachea todos los modelos entrenados.
    return {
        "km_match": load_model("km_match"),
        "ag_match": load_model("ag_match"),
        "scaler_match": load_model("scaler_match"),
        "pca_match": load_model("pca_match"),
        "k_results_match": load_model("k_results_match"),
        "km_team": load_model("km_team"),
        "ag_team": load_model("ag_team"),
        "pca_team": load_model("pca_team"),
        "k_results_team": load_model("k_results_team"),
        "km_temporal": load_model("km_temporal"),
        "teams_features": load_model("teams_features"),
        "matches_features": load_model("matches_features"),
        "temporal_features": load_model("temporal_features"),
        "lr_over25": load_model("lr_over25"),
        "rf_over25": load_model("rf_over25"),
        "scaler_supervised": load_model("scaler_supervised"),
        "supervised_features": load_model("supervised_features"),
        "supervised_metrics": load_model("supervised_metrics"),
    }
