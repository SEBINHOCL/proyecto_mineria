# Sistema de análisis y estimación probabilística de goles en fútbol

**Curso:** TEL354 - Minería de Datos · UTFSM  
**Equipo:** Vicente Beiza · Lucas Andrade · Sebastián Vicuña  
**Liga:** Premier League · Temporadas 2023/24 y 2024/25

---

## ¿Qué hace este proyecto?

Extrae datos históricos de partidos de la Premier League desde la API de API-Football, construye variables analíticas y aplica técnicas de minería de datos para:

1. **Caracterizar patrones de juego** mediante clustering no supervisado (K-Means + Agglomerative Clustering)
2. **Estimar probabilidades** de que un partido termine con más de 2.5 goles (Over/Under) mediante clasificación supervisada
3. **Presentar los resultados** en un dashboard web interactivo construido con Streamlit

---

## Estructura del proyecto

```
mineria_futbol/
├── src/                        # Lógica central del proyecto
│   ├── config.py               # Rutas, constantes y carga del .env
│   ├── api_client.py           # Cliente HTTP para API-Football
│   ├── db.py                   # Acceso a la base de datos SQLite
│   ├── features.py             # Feature engineering (clustering y supervisado)
│   └── models.py               # Entrenamiento, evaluación y persistencia de modelos
│
├── dashboard/                  # Aplicación web
│   ├── app.py                  # Punto de entrada Streamlit
│   └── utils.py                # Carga de datos y modelos cacheados
│
├── scripts/                    # Utilidades de mantenimiento
│   ├── download_events.py      # Descarga eventos de gol desde la API
│   ├── rebuild_db.py           # Reconstruye la DB desde los JSON crudos
│   └── train_models.py         # Entrena y guarda todos los modelos ML
│
├── data/
│   ├── football.db             # Base de datos SQLite (generada)
│   ├── raw/                    # JSON crudos descargados desde la API
│   └── models/                 # Modelos entrenados (.pkl)
│
├── .env                        # API key (no incluida en el repositorio)
├── .env.example                # Plantilla del archivo .env
└── requirements.txt            # Dependencias Python
```

---

## Requisitos

- Python 3.10 o superior
- La API key de api-football.com es **opcional**: los datos ya están incluidos en el repositorio (`data/raw/` y `data/football.db`), por lo que se puede ejecutar el dashboard directamente sin cuenta. Solo es necesaria si se quiere re-descargar los datos desde la API.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/SEBINHOCL/proyecto_mineria.git
cd proyecto_mineria

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Configurar la API key — solo si se quiere re-descargar datos
cp .env.example .env
# Editar .env y agregar: API_FOOTBALL_KEY=tu_clave_aqui
```

---

## Reproducir el proyecto paso a paso

### Paso 1 - Descargar los datos (opcional)

Los datos ya están incluidos en el repositorio: `data/raw/` contiene los JSON crudos y `data/football.db` la base de datos lista. **Se puede saltar al Paso 4 directamente.**

Si se quieren regenerar desde cero usando la API:

```bash
# Descarga eventos de gol (respeta el límite de 100 req/día del plan gratuito)
# Ejecutar una vez por día hasta completar las temporadas
python scripts/download_events.py
```

El script es **incremental**: retoma automáticamente donde se quedó si se interrumpe.

### Paso 2 - Construir la base de datos

```bash
python scripts/rebuild_db.py
```

Transforma los JSON crudos en una base de datos SQLite con 5 tablas normalizadas: `leagues`, `teams`, `players`, `matches`, `goals`.

### Paso 3 - Entrenar los modelos

```bash
python scripts/train_models.py
```

Entrena y guarda en `data/models/` todos los modelos:
- No supervisado: K-Means y Agglomerative Clustering para partidos, equipos y análisis temporal
- Supervisado: Regresión Logística, Random Forest y Baseline Poisson para predicción Over/Under 2.5

### Paso 4 - Ejecutar el dashboard

```bash
streamlit run dashboard/app.py
```

Abre el navegador en `http://localhost:8501`.

---

## Dataset

| Fuente | API-Football v3 (api-sports.io) |
|--------|--------------------------------|
| Liga | Premier League (ID 39) |
| Temporadas | 2023/24 y 2024/25 |
| Partidos | 760 (380 por temporada) |
| Goles registrados | 2.375 eventos |
| Jugadores | 512 |
| Endpoints usados | `/fixtures`, `/fixtures/events` |

Los archivos JSON crudos están en `data/raw/` y la base de datos SQLite en `data/football.db`.

---

## Resultados principales

### No supervisado

| Modelo | Aplicado a | k óptimo | Silhouette | Resultado destacado |
|--------|-----------|----------|------------|---------------------|
| K-Means | Partidos | 3 | 0.311 | Baja intensidad · Dominio local · Dominio visitante |
| K-Means | Equipos | 2 | 0.488 | Equipos establecidos vs recién ascendidos |
| Agglomerative | Equipos | 2 | - | ARI=1.0: coincidencia perfecta con K-Means |

### Supervisado - Predicción Over/Under 2.5 goles

| Modelo | Accuracy (test) | ROC-AUC |
|--------|----------------|---------|
| Baseline Poisson | 57.6% | 0.535 |
| Regresión Logística | 57.1% | 0.482 |
| Random Forest | 57.1% | 0.531 |

**Validación:** entrenamiento en 2023/24, evaluación en 2024/25 (split cronológico).

El rendimiento similar al baseline refleja una limitación conocida: con solo historial de goles como features, la señal predictiva es débil. Features adicionales como xG o estadísticas de jugadores mejorarían la discriminación.

---

## Variables construidas

**Clustering de partidos** (`src/features.py`):
- `total_goals`, `goal_diff`, `halftime_total`, `second_half_goals`, `halftime_diff`

**Clustering de equipos:**
- Goles a favor/contra por partido, % victorias local/visitante, % empates

**Supervisado - rolling últimos 5 partidos por equipo:**
- `home_scored_5`, `home_conceded_5`, `home_form_5`
- `away_scored_5`, `away_conceded_5`, `away_form_5`
- `combined_attack_5`

---

## Dependencias principales

```
requests, python-dotenv          # Extracción de datos
pandas, numpy                    # Manipulación de datos
scikit-learn                     # Modelos ML
scipy                            # Baseline Poisson
streamlit                        # Dashboard web
plotly                           # Visualizaciones interactivas
```

Lista completa en `requirements.txt`.
