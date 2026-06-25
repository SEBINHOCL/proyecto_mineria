# Feature engineering para los modelos de ML.

# Transforma los DataFrames crudos de la DB en matrices de features
# listas para ser consumidas por los modelos de clustering y clasificación.

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ========================================================================
# Features de partidos
# ========================================================================

MATCH_FEATURES = [
    "home_goals",
    "away_goals",
    "total_goals",
    "goal_diff",
    "halftime_total",
    "second_half_goals",
    "halftime_diff",
]


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    # Construye las variables de clustering a nivel partido.
    df = matches.copy()
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["halftime_total"] = df["halftime_home"] + df["halftime_away"]
    df["second_half_goals"] = df["total_goals"] - df["halftime_total"]
    df["halftime_diff"] = df["halftime_home"] - df["halftime_away"]
    return df


def scale_match_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[MATCH_FEATURES].values)
    scaled_df = pd.DataFrame(X_scaled, columns=MATCH_FEATURES, index=df.index)
    return scaled_df, scaler


# ========================================================================
# Features de equipos
# ========================================================================

TEAM_FEATURES = [
    "goles_favor_por_partido",
    "goles_contra_por_partido",
    "diferencia_goles",
    "pct_victorias",
    "pct_empates",
    "pct_victorias_local",
    "pct_victorias_visitante",
]


def build_team_features(matches: pd.DataFrame) -> pd.DataFrame:
    # Construye las variables de clustering a nivel equipo.
    # Recibe el DataFrame de get_matches() y devuelve un DataFrame
    # con una fila por equipo y sus métricas agregadas.

    records = []

    for team in pd.unique(
        pd.concat([matches["home_team"], matches["away_team"]])
    ):
        home = matches[matches["home_team"] == team]
        away = matches[matches["away_team"] == team]
        total = len(home) + len(away)

        if total == 0:
            continue

        gf = home["home_goals"].sum() + away["away_goals"].sum()
        gc = home["away_goals"].sum() + away["home_goals"].sum()

        wins_h = (home["home_goals"] > home["away_goals"]).sum()
        wins_a = (away["away_goals"] > away["home_goals"]).sum()
        draws = ((home["home_goals"] == home["away_goals"]).sum() +
                   (away["away_goals"] == away["home_goals"]).sum())

        records.append({
            "team" : team,
            "partidos" : total,
            "goles_favor_por_partido" : gf / total,
            "goles_contra_por_partido" : gc / total,
            "diferencia_goles" : (gf - gc) / total,
            "pct_victorias" : (wins_h + wins_a) / total,
            "pct_empates" : draws / total,
            "pct_victorias_local" : wins_h / len(home) if len(home) > 0 else 0,
            "pct_victorias_visitante" : wins_a / len(away) if len(away) > 0 else 0,
        })

    return pd.DataFrame(records).set_index("team")


def scale_team_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    # Estandariza los features de equipo.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[TEAM_FEATURES].values)
    scaled_df = pd.DataFrame(X_scaled, columns=TEAM_FEATURES, index=df.index)
    return scaled_df, scaler


# ========================================================================
# Features temporales (goles por minuto)
# ========================================================================

MINUTE_BINS   = [0, 15, 30, 45, 60, 75, 90, 120]
MINUTE_LABELS = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "90+"]


# ========================================================================
# Features para clasificación supervisada (Over/Under 2.5 goles)
# ========================================================================

SUPERVISED_FEATURES = [
    "home_scored_5",
    "home_conceded_5",
    "home_form_5",
    "away_scored_5",
    "away_conceded_5",
    "away_form_5",
    "combined_attack_5",
]
SUPERVISED_TARGET = "over_2_5"


def _team_rolling_stats(
    team: str, date: pd.Timestamp, matches: pd.DataFrame, window: int
) -> dict | None:
    # Retorna estadisticas rolling del equipo en sus ultimos "window" partidos antes de "date".
    past = matches[
        (matches["date"] < date) &
        ((matches["home_team"] == team) | (matches["away_team"] == team))
    ].sort_values("date").tail(window)

    if len(past) < window:
        return None

    scored, conceded, points = [], [], []
    for _, m in past.iterrows():
        if m["home_team"] == team:
            g_for, g_ag = m["home_goals"], m["away_goals"]
        else:
            g_for, g_ag = m["away_goals"], m["home_goals"]

        scored.append(g_for)
        conceded.append(g_ag)
        if g_for > g_ag:    points.append(3)
        elif g_for == g_ag: points.append(1)
        else:               points.append(0)

    return {
        "scored":   float(np.mean(scored)),
        "conceded": float(np.mean(conceded)),
        "form":     float(np.mean(points)),
    }


def build_supervised_features(
    matches: pd.DataFrame, window: int = 5
) -> pd.DataFrame:
    # Construye la matriz de features para predicción supervisada.

    matches = matches.sort_values("date").reset_index(drop=True)
    records = []

    for _, row in matches.iterrows():
        home_s = _team_rolling_stats(row["home_team"], row["date"], matches, window)
        away_s = _team_rolling_stats(row["away_team"], row["date"], matches, window)

        if home_s is None or away_s is None:
            continue

        records.append({
            "fixture_id":        row["fixture_id"],
            "date":              row["date"],
            "season":            row["season"],
            "home_team":         row["home_team"],
            "away_team":         row["away_team"],
            "home_scored_5":     home_s["scored"],
            "home_conceded_5":   home_s["conceded"],
            "home_form_5":       home_s["form"],
            "away_scored_5":     away_s["scored"],
            "away_conceded_5":   away_s["conceded"],
            "away_form_5":       away_s["form"],
            "combined_attack_5": home_s["scored"] + away_s["scored"],
            SUPERVISED_TARGET:   int((row["home_goals"] + row["away_goals"]) > 2.5),
        })

    return pd.DataFrame(records)


def scale_supervised_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    # Estandariza los features supervisados.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[SUPERVISED_FEATURES].values)
    return pd.DataFrame(X_scaled, columns=SUPERVISED_FEATURES, index=df.index), scaler


def build_temporal_features(goals: pd.DataFrame) -> pd.DataFrame:
    # Construye la distribucion de goles por bloque de 15 minutos.

    df = goals[goals["goal_type"] != "OwnGoal"].copy()
    df["bloque"] = pd.cut(
        df["minute"], bins=MINUTE_BINS, labels=MINUTE_LABELS, right=True
    )

    pivot = (
        df.groupby(["team", "bloque"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=MINUTE_LABELS, fill_value=0)
    )

    # Normalizar por total de goles del equipo para comparar sin sesgo de volumen
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    pivot_norm.columns = [f"min_{c}" for c in pivot_norm.columns]

    return pivot_norm
