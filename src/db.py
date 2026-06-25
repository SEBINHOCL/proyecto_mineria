from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator

import pandas as pd

from .config import DB_PATH


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager que abre y cierra la conexión automáticamente."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ========================================================================
# Matches
# ========================================================================

def get_matches(season: int | None = None) -> pd.DataFrame:
    # Retorna todos los partidos finalizados con nombres de equipos.
    # Si se pasa season, filtra por temporada.

    where = "WHERE m.status = 'FT'"
    params: list = []
    if season is not None:
        where += " AND m.season = ?"
        params.append(season)

    query = f"""
        SELECT
            m.fixture_id,
            m.season,
            m.round,
            m.date,
            ht.name  AS home_team,
            at.name  AS away_team,
            m.home_goals,
            m.away_goals,
            m.halftime_home,
            m.halftime_away
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.team_id
        JOIN teams at ON m.away_team_id = at.team_id
        {where}
        ORDER BY m.date
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params, parse_dates=["date"])


def get_teams() -> pd.DataFrame:
    """Retorna todos los equipos con sus estadísticas agregadas."""
    query = """
        SELECT
            t.team_id,
            t.name,
            COUNT(DISTINCT CASE WHEN m.home_team_id = t.team_id OR m.away_team_id = t.team_id
                                THEN m.fixture_id END)                         AS partidos,
            SUM(CASE WHEN m.home_team_id = t.team_id THEN m.home_goals
                     WHEN m.away_team_id = t.team_id THEN m.away_goals END)    AS goles_a_favor,
            SUM(CASE WHEN m.home_team_id = t.team_id THEN m.away_goals
                     WHEN m.away_team_id = t.team_id THEN m.home_goals END)    AS goles_en_contra
        FROM teams t
        JOIN matches m ON (m.home_team_id = t.team_id OR m.away_team_id = t.team_id)
        WHERE m.status = 'FT'
        GROUP BY t.team_id, t.name
        ORDER BY goles_a_favor DESC
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


# ========================================================================
# Goals
# ========================================================================

def get_goals(season: int | None = None) -> pd.DataFrame:
    # Retorna todos los eventos de gol con nombres de equipo y jugador.
    # Si se pasa season, filtra por temporada.

    where = ""
    params: list = []
    if season is not None:
        where = "WHERE m.season = ?"
        params.append(season)

    query = f"""
        SELECT
            g.id,
            g.fixture_id,
            m.season,
            m.date,
            g.minute,
            g.extra_minute,
            g.goal_type,
            t.name   AS team,
            p.name   AS player,
            ap.name  AS assist
        FROM goals g
        JOIN matches  m  ON g.fixture_id        = m.fixture_id
        JOIN teams    t  ON g.team_id            = t.team_id
        LEFT JOIN players p  ON g.player_id     = p.player_id
        LEFT JOIN players ap ON g.assist_player_id = ap.player_id
        {where}
        ORDER BY m.date, g.minute
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params, parse_dates=["date"])


def get_top_scorers(season: int | None = None, limit: int = 20) -> pd.DataFrame:
    # Retorna el ranking de goleadores (excluye autogoles).

    where = "WHERE g.goal_type != 'OwnGoal'"
    params: list = []
    if season is not None:
        where += " AND m.season = ?"
        params.append(season)

    query = f"""
        SELECT
            p.name      AS player,
            t.name      AS team,
            COUNT(*)    AS goles
        FROM goals g
        JOIN matches  m ON g.fixture_id  = m.fixture_id
        JOIN teams    t ON g.team_id     = t.team_id
        JOIN players  p ON g.player_id   = p.player_id
        {where}
        GROUP BY p.player_id, p.name, t.name
        ORDER BY goles DESC
        LIMIT ?
    """
    params.append(limit)
    with get_connection() as conn:
        return pd.read_sql(query, conn, params=params)


# ========================================================================
# Resumen general
# ========================================================================

def get_summary() -> dict:
    # Retorna metricas generales de la base de datos.
    with get_connection() as conn:
        cur = conn.cursor()
        summary = {}
        for table in ("leagues", "teams", "players", "matches", "goals"):
            summary[table] = cur.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

        row = cur.execute("""
            SELECT
                SUM(home_goals + away_goals)                        AS total_goles,
                AVG(home_goals + away_goals)                        AS promedio_goles,
                COUNT(CASE WHEN home_goals > away_goals THEN 1 END) AS victorias_local,
                COUNT(CASE WHEN home_goals = away_goals THEN 1 END) AS empates,
                COUNT(CASE WHEN home_goals < away_goals THEN 1 END) AS victorias_visitante
            FROM matches
            WHERE status = 'FT'
        """).fetchone()

        summary["total_goles"]          = row[0]
        summary["promedio_goles"]       = round(row[1], 2) if row[1] else 0
        summary["victorias_local"]      = row[2]
        summary["empates"]              = row[3]
        summary["victorias_visitante"]  = row[4]

    return summary
