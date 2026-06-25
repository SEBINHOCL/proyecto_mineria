# Descarga eventos de goles desde API-Football de forma incremental.
# Retoma automáticamente desde donde quedó en sesiones anteriores.
# Usar: python scripts/download_events.py

import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DIR, SEASONS
from src.api_client import ApiFootballClient, ApiFootballError

SECONDS_BETWEEN = 7.0
SAVE_EVERY      = 10

client = ApiFootballClient(min_seconds_between_requests=SECONDS_BETWEEN)

for season in SEASONS:
    fixtures_path = RAW_DIR / f"fixtures_pl_{season}.json"
    events_path   = RAW_DIR / f"events_pl_{season}.json"

    if not fixtures_path.exists():
        print(f"Temporada {season}: sin fixtures. Ejecuta la extracción de fixtures primero.")
        continue

    with open(fixtures_path) as f:
        all_fixtures = json.load(f)["response"]

    events = {}
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)

    pending = [
        m["fixture"]["id"]
        for m in all_fixtures
        if str(m["fixture"]["id"]) not in events
    ]

    print(f"\nTemporada {season}: {len(all_fixtures)} partidos | "
          f"{len(events)} descargados | {len(pending)} pendientes")

    if not pending:
        print(f"  Temporada {season} ya completa.")
        continue

    count = 0
    for fid in pending:
        try:
            resp = client.get("fixtures/events", {"fixture": fid, "type": "Goal"})
        except ApiFootballError as e:
            print(f"\n  Límite alcanzado en fixture {fid}: {e}")
            with open(events_path, "w") as f:
                json.dump(events, f)
            print(f"  Guardado: {len(events)} partidos totales en {events_path.name}")
            sys.exit(0)

        events[str(fid)] = resp.get("response", [])
        count += 1

        if count % SAVE_EVERY == 0:
            with open(events_path, "w") as f:
                json.dump(events, f)
            print(f"  [{count}/{len(pending)}] guardado ({len(events)} totales)")

    with open(events_path, "w") as f:
        json.dump(events, f)
    print(f"  Temporada {season} completa. ({len(events)} partidos)")

print("\nExtracción finalizada.")
