# Configuración central del proyecto.

# Carga variables de entorno desde .env y expone constantes
# usadas por todos los módulos: paths, IDs de liga, temporadas,
# URL base de la API.

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "football.db"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Variables de entorno
load_dotenv(ROOT_DIR / ".env")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "").strip()

# Constantes de la API
API_BASE_URL = "https://v3.football.api-sports.io"

# Alcance del proyecto
PREMIER_LEAGUE_ID = 39
SEASONS = [2023, 2024]
