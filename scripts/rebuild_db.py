import json, sqlite3
from pathlib import Path

ROOT_DIR   = Path(".")
RAW_DIR    = ROOT_DIR / "data" / "raw"
DB_PATH    = ROOT_DIR / "data" / "football.db"
SEASONS    = [2023, 2024]
DETAIL_MAP = {"Normal Goal": "Normal", "Penalty": "Penalty", "Own Goal": "OwnGoal"}

DDL = """
CREATE TABLE IF NOT EXISTS leagues (league_id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS teams (team_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS players (player_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS matches (
    fixture_id INTEGER PRIMARY KEY, league_id INTEGER NOT NULL, season INTEGER NOT NULL,
    round TEXT NOT NULL, date TEXT NOT NULL, home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL, home_goals INTEGER, away_goals INTEGER,
    halftime_home INTEGER, halftime_away INTEGER, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id INTEGER NOT NULL,
    minute INTEGER, extra_minute INTEGER, team_id INTEGER NOT NULL,
    player_id INTEGER, assist_player_id INTEGER, goal_type TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS idx_goals_fixture  ON goals(fixture_id);
CREATE INDEX IF NOT EXISTS idx_goals_player   ON goals(player_id);
CREATE INDEX IF NOT EXISTS idx_goals_team     ON goals(team_id);
"""

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")
conn.executescript(DDL)
cur = conn.cursor()

for season in SEASONS:
    path = RAW_DIR / f"fixtures_pl_{season}.json"
    if not path.exists(): continue
    with open(path) as f:
        data = json.load(f)
    for m in data["response"]:
        lg, fx, sc = m["league"], m["fixture"], m["score"]
        cur.execute("INSERT OR IGNORE INTO leagues VALUES (?,?,?)", (lg["id"], lg["name"], lg["country"]))
        for side in ("home", "away"):
            t = m["teams"][side]
            cur.execute("INSERT OR IGNORE INTO teams VALUES (?,?)", (t["id"], t["name"]))
        cur.execute("""INSERT OR IGNORE INTO matches
            (fixture_id,league_id,season,round,date,home_team_id,away_team_id,
             home_goals,away_goals,halftime_home,halftime_away,status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fx["id"],lg["id"],lg["season"],lg["round"],fx["date"],
             m["teams"]["home"]["id"],m["teams"]["away"]["id"],
             m["goals"]["home"],m["goals"]["away"],
             sc["halftime"]["home"],sc["halftime"]["away"],fx["status"]["short"]))
conn.commit()

for season in SEASONS:
    path = RAW_DIR / f"events_pl_{season}.json"
    if not path.exists(): continue
    with open(path) as f:
        events = json.load(f)
    for fid_str, evs in events.items():
        fid = int(fid_str)
        cur.execute("DELETE FROM goals WHERE fixture_id = ?", (fid,))
        for ev in evs:
            player = ev.get("player") or {}
            assist = ev.get("assist") or {}
            pid, aid = player.get("id"), assist.get("id")
            if pid: cur.execute("INSERT OR IGNORE INTO players VALUES (?,?)", (pid, player["name"]))
            if aid: cur.execute("INSERT OR IGNORE INTO players VALUES (?,?)", (aid, assist["name"]))
            cur.execute("""INSERT INTO goals
                (fixture_id,minute,extra_minute,team_id,player_id,assist_player_id,goal_type)
                VALUES (?,?,?,?,?,?,?)""",
                (fid,ev["time"]["elapsed"],ev["time"]["extra"],ev["team"]["id"],
                 pid,aid,DETAIL_MAP.get(ev.get("detail",""),"Normal")))
conn.commit()

print(f"DB actualizada: {DB_PATH}\n")
for table in ("leagues","teams","players","matches","goals"):
    n = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<10} {n:>6} filas")
conn.close()
