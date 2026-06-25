# Dashboard - Premier League: analisis de goles
# Ejecutar con: streamlit run dashboard/app.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from dashboard.utils import load_all_data, load_all_models, PALETTE

st.set_page_config(
    page_title="Premier League - Análisis de Goles",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚽ Premier League")
st.sidebar.markdown("**Análisis de Goles 2023/24 - 2024/25**")
st.sidebar.markdown("TEL354 · Minería de Datos · UTFSM")
st.sidebar.divider()

PAGES = [
    "Introducción",
    "Análisis Exploratorio",
    "Clustering: Partidos",
    "Clustering: Equipos",
    "Análisis Temporal",
    "Predicción: Over 2.5",
    "Goleadores",
    "Conclusiones",
]

page = st.sidebar.radio("Sección", PAGES, label_visibility="collapsed")

data   = load_all_data()
models = load_all_models()


# ===========================================================================
# INTRODUCCION
# ===========================================================================
if page == "Introducción":
    st.title("Sistema de análisis y estimación probabilística de goles")
    st.markdown("### Premier League · Temporadas 2023/24 y 2024/25")
    st.divider()

    st.markdown("""
    ## Problema
    ¿Es posible identificar patrones estadísticos en los partidos de fútbol
    y estimar la probabilidad de que un partido tenga más o menos de 2.5 goles,
    usando solo datos históricos disponibles antes del partido?

    ## Objetivo
    Diseñar e implementar un sistema de minería de datos que, a partir de datos
    históricos de la Premier League obtenidos desde API-Football:

    - **Caracterizar** patrones de juego mediante aprendizaje no supervisado
    - **Estimar** probabilidades de eventos mediante aprendizaje supervisado
    - **Comunicar** los hallazgos a través de este dashboard interactivo
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)
    s = data["summary"]
    col1.metric("Partidos analizados", f"{s["matches"]}")
    col2.metric("Goles registrados",   f"{s["total_goles"]}")
    col3.metric("Promedio goles/partido", f"{s["promedio_goles"]}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Jugadores en DB", f"{s["players"]:,}")
    col5.metric("% Victoria local",     f"{round(s["victorias_local"]/s["matches"]*100, 1)}%")
    col6.metric("% Victoria visitante", f"{round(s["victorias_visitante"]/s["matches"]*100, 1)}%")

    st.divider()

    col_i, col_d = st.columns(2)
    with col_i:
        st.markdown("## Metodología")
        st.markdown("""
        1. **Recolección**: API-Football v3, endpoints `/fixtures` y `/fixtures/events`.
        2. **Almacenamiento**: Base de datos SQLite con 5 tablas normalizadas.
        3. **EDA**: Análisis exploratorio de distribuciones y patrones.
        4. **Feature engineering**: Variables rolling de últimos 5 partidos por equipo.
        5. **No supervisado**: K-Means + Agglomerative Clustering + PCA.
        6. **Supervisado**: Regresión Logística + Random Forest + Baseline Poisson.
        7. **Evaluación**: Silhouette, Davies-Bouldin, ROC-AUC, Brier Score.
        """)

    with col_d:
        st.markdown("## Fuente de datos")
        st.markdown("""
        **API-Football (api-sports.io)**
        - Plan gratuito: 100 requests/día
        - Liga: Premier League (ID 39)
        - Temporadas: 2023/24 y 2024/25
        - Datos: resultados, eventos de gol por minuto, tipo (normal/penal/autogol)

        **Limitación conocida:** el plan gratuito no incluye estadísticas
        avanzadas por partido (xG, tiros, posesión), lo que acota el poder
        predictivo de los modelos supervisados.
        """)


# ===========================================================================
# ANALISIS EXPLORATORIO
# ===========================================================================
elif page == "Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos")
    st.divider()

    matches = data["matches_feat"]
    goals   = data["goals"]

    tab1, tab2, tab3 = st.tabs(["Distribuciones", "Relaciones", "Por temporada"])

    with tab1:
        st.subheader("Distribución de goles por partido")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                matches, x="total_goals", nbins=12,
                color_discrete_sequence=[PALETTE[0]],
                labels={"total_goals": "Goles totales", "count": "Partidos"},
                title="Total de goles por partido",
            )
            fig.add_vline(x=2.5, line_dash="dash", line_color="red",
                          annotation_text="Línea 2.5")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            pct_over = (matches["total_goals"] > 2.5).mean()
            st.caption(f"**{round(pct_over * 100, 1)}%** de los partidos terminaron con más de 2.5 goles.")

        with col2:
            fig2 = px.box(
                matches.melt(value_vars=["home_goals", "away_goals"],
                             var_name="condición", value_name="goles"),
                x="condición", y="goles",
                color="condición",
                color_discrete_map={"home_goals": PALETTE[0], "away_goals": PALETTE[1]},
                labels={"condición": "", "goles": "Goles"},
                title="Boxplot: goles local vs visitante",
            )
            fig2.update_xaxes(ticktext=["Local", "Visitante"], tickvals=["home_goals", "away_goals"])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Media local: **{matches["home_goals"].mean():.2f}** · "
                f"Media visitante: **{matches["away_goals"].mean():.2f}**"
            )

        st.subheader("Distribución de features de clustering")
        feat_cols = ["total_goals", "goal_diff", "halftime_total", "second_half_goals"]
        feat_labels = ["Total goles", "Diferencia goles", "Goles 1er tiempo", "Goles 2do tiempo"]
        fig3 = px.box(
            matches.melt(value_vars=feat_cols, var_name="feature", value_name="valor"),
            x="feature", y="valor",
            color="feature",
            color_discrete_sequence=PALETTE,
            labels={"feature": "Variable", "valor": "Valor"},
            title="Distribución de variables principales (boxplot)",
        )
        fig3.update_xaxes(ticktext=feat_labels, tickvals=feat_cols)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Los goles en el segundo tiempo superan consistentemente al primer tiempo. "
            "La diferencia de goles tiene mediana cercana a 0, confirmando paridad general."
        )

    with tab2:
        st.subheader("Relación entre goles local y visitante")
        fig4 = px.scatter(
            matches, x="home_goals", y="away_goals",
            opacity=0.4,
            color_discrete_sequence=[PALETTE[0]],
            labels={"home_goals": "Goles local", "away_goals": "Goles visitante"},
            title="Dispersión: goles local vs visitante",
            trendline="ols",
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(
            "Correlación débil entre goles de ambos equipos - los partidos de alto marcador "
            "no necesariamente implican que ambos equipos anotan mucho."
        )

        st.subheader("Goles promedio por jornada")
        goals_per_round = (
            matches.groupby("round")["total_goals"]
            .mean()
            .reset_index()
        )
        goals_per_round["jornada"] = goals_per_round["round"].str.extract(r"(\d+)").astype(float)
        goals_per_round = goals_per_round.dropna().sort_values("jornada")
        fig5 = px.line(
            goals_per_round, x="jornada", y="total_goals",
            labels={"jornada": "Jornada", "total_goals": "Goles promedio"},
            color_discrete_sequence=[PALETTE[0]],
            title="Promedio de goles por jornada",
        )
        fig5.add_hline(y=data["summary"]["promedio_goles"], line_dash="dash",
                       line_color="gray",
                       annotation_text=f"Media global {data["summary"]["promedio_goles"]:.2f}")
        st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.subheader("Comparación entre temporadas")
        res = (
            data["matches"]
            .assign(resultado=lambda d: d.apply(
                lambda r: "Local" if r.home_goals > r.away_goals
                else ("Visitante" if r.home_goals < r.away_goals else "Empate"), axis=1))
            .groupby(["season", "resultado"]).size().reset_index(name="partidos")
        )
        fig6 = px.bar(
            res, x="season", y="partidos", color="resultado", barmode="group",
            color_discrete_map={"Local": PALETTE[0], "Visitante": PALETTE[1], "Empate": PALETTE[2]},
            labels={"season": "Temporada", "partidos": "Partidos"},
            title="Resultados por temporada",
        )
        st.plotly_chart(fig6, use_container_width=True)

        col1, col2 = st.columns(2)
        for i, season in enumerate([2023, 2024]):
            s_data = data["matches"][data["matches"]["season"] == season]
            avg = (s_data["home_goals"] + s_data["away_goals"]).mean()
            pct_over_s = ((s_data["home_goals"] + s_data["away_goals"]) > 2.5).mean()
            [col1, col2][i].metric(
                f"Temporada {season}/{str(season+1)[2:]}",
                f"{avg:.2f} goles/partido",
                f"{round(pct_over_s * 100, 1)}% Over 2.5",
            )


# ===========================================================================
# CLUSTERING: PARTIDOS
# ===========================================================================
elif page == "Clustering: Partidos":
    st.title("Clustering de Partidos")
    st.markdown(""" 
    Agrupar los 760 partidos según sus patrones de goles,
    sin usar etiquetas predefinidas. El modelo descubre por sí solo si existen
    tipos distintos de partido en la Premier League.

    **Algoritmos usados:** K-Means y Agglomerative Clustering.
    """)
    st.divider()

    km = models["km_match"]
    ag = models["ag_match"]
    pca = models["pca_match"]
    k_res = models["k_results_match"]
    matches = models["matches_features"].copy()
    X = data["X_match"]

    matches["cluster"] = km.labels_
    cluster_names = {0: "Baja intensidad", 1: "Dominio local", 2: "Dominio visitante"}
    matches["Tipo"] = matches["cluster"].map(cluster_names)

    tab1, tab2, tab3, tab4 = st.tabs(["Selección de k", "Mapa PCA", "Perfiles", "Comparación de métodos"])

    with tab1:
        st.markdown("### ¿Cuántos grupos existen en los datos?")
        st.markdown(
            "El **método del codo** busca el punto donde agregar más clusters deja de "
            "reducir significativamente la inercia. El **Silhouette Score** mide qué tan "
            "bien separados están los clusters (más alto = mejor)."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig_elbow = px.line(x=k_res["k"], y=k_res["inertia"], markers=True,
                                labels={"x": "k (clusters)", "y": "Inercia"},
                                title="Método del codo",
                                color_discrete_sequence=[PALETTE[0]])
            st.plotly_chart(fig_elbow, use_container_width=True)
        with c2:
            best_k = k_res["k"][k_res["silhouette"].index(max(k_res["silhouette"]))]
            fig_sil = px.line(x=k_res["k"], y=k_res["silhouette"], markers=True,
                              labels={"x": "k (clusters)", "y": "Silhouette"},
                              title="Silhouette Score",
                              color_discrete_sequence=[PALETTE[1]])
            fig_sil.add_vline(x=best_k, line_dash="dash", line_color="green",
                              annotation_text=f"k={best_k} óptimo")
            st.plotly_chart(fig_sil, use_container_width=True)

        st.success(f"**k={best_k} seleccionado** - mayor Silhouette Score ({max(k_res["silhouette"]):.4f}). "
                   "Silhouette Score positivo indica que los puntos están más cerca de su propio cluster "
                   "que del vecino.")

    with tab2:
        st.markdown("### Visualización 2D con PCA")
        st.markdown(
            "Se reducen las 7 features a 2 componentes principales para poder visualizar "
            "los clusters. Cada punto es un partido."
        )
        coords = pca.transform(X.values)
        df_2d  = pd.DataFrame(coords, columns=["PC1", "PC2"], index=X.index)
        df_2d["Tipo"]    = matches["Tipo"].values
        df_2d["tooltip"] = (matches["home_team"].values + " vs " + matches["away_team"].values +
                            " (" + matches["home_goals"].astype(str).values + "-" +
                            matches["away_goals"].astype(str).values + ")")
        var = pca.explained_variance_ratio_
        fig_pca = px.scatter(
            df_2d, x="PC1", y="PC2", color="Tipo",
            hover_data={"tooltip": True, "PC1": False, "PC2": False, "Tipo": False},
            color_discrete_sequence=PALETTE[:3],
            labels={"PC1": f"PC1 ({var[0]:.1%} var.)", "PC2": f"PC2 ({var[1]:.1%} var.)"},
            title=f"Clusters de partidos - PCA 2D (varianza explicada: {sum(var):.1%})",
            opacity=0.7,
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        from src.models import get_pca_loadings
        loadings = get_pca_loadings(pca, list(X.columns))
        st.markdown("**Contribución de cada variable a los componentes:**")
        st.dataframe(loadings.style.background_gradient(cmap="RdBu", axis=None), use_container_width=True)

    with tab3:
        st.markdown("### Perfil promedio por cluster")
        from src.features import MATCH_FEATURES
        profile = matches.groupby("Tipo")[MATCH_FEATURES].mean().round(2)
        profile["n_partidos"] = matches["Tipo"].value_counts()
        st.dataframe(profile, use_container_width=True)

        st.markdown("### Interpretación de los clusters")
        col1, col2, col3 = st.columns(3)
        col1.info("**Baja intensidad** (404 partidos · 53%)\n\n"
                  "~1.9 goles totales. Partidos cerrados y equilibrados. "
                  "Ningún equipo domina claramente.")
        col2.success("**Dominio local** (199 partidos · 26%)\n\n"
                     "~4.5 goles. El equipo local gana de forma contundente "
                     "(3.4 vs 1.1 en promedio).")
        col3.warning("**Dominio visitante** (157 partidos · 21%)\n\n"
                     "~4.5 goles. El equipo visitante gana claramente "
                     "(1.3 vs 3.2 en promedio).")

        st.markdown(
            "**Hallazgo clave:** más de la mitad de los partidos de Premier League "
            "son de baja intensidad. Los partidos de alto marcador tienden a ser "
            "dominantes - raramente ambos equipos anotan muchos goles."
        )

    with tab4:
        st.markdown("### K-Means vs Agglomerative Clustering")
        st.markdown(
            "Para validar que los clusters son reales y no un artefacto del algoritmo, "
            "se comparan los resultados de K-Means con Agglomerative Clustering usando "
            "el **Adjusted Rand Index (ARI)**: 1.0 = coincidencia perfecta, 0.0 = aleatoria."
        )
        from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
        ari = adjusted_rand_score(km.labels_, ag.labels_)
        sil_km = silhouette_score(X, km.labels_)
        sil_ag = silhouette_score(X, ag.labels_)
        db_km  = davies_bouldin_score(X, km.labels_)
        db_ag  = davies_bouldin_score(X, ag.labels_)

        comp = pd.DataFrame({
            "Silhouette ↑": [sil_km, sil_ag],
            "Davies-Bouldin ↓": [db_km, db_ag],
        }, index=["K-Means", "Agglomerative"])
        st.dataframe(comp.round(4), use_container_width=True)
        if ari >= 0.7:
            st.success(f"**ARI = {ari:.4f}** - alta concordancia entre métodos. Los clusters son robustos.")
        elif ari >= 0.4:
            st.warning(f"**ARI = {ari:.4f}** - concordancia moderada. Los clusters de partidos se solapan naturalmente.")
        else:
            st.error(f"**ARI = {ari:.4f}** - baja concordancia entre métodos.")

        st.caption(
            "Una concordancia moderada en partidos es esperable: los partidos de fútbol "
            "no forman grupos perfectamente separados - existe un espectro continuo entre "
            "baja y alta intensidad."
        )


# ===========================================================================
# CLUSTERING: EQUIPOS
# ===========================================================================
elif page == "Clustering: Equipos":
    st.title("Clustering de Equipos")
    st.markdown("""
    Identificar perfiles de equipo basados en su rendimiento
    a lo largo de las dos temporadas: goles anotados/recibidos,
    porcentaje de victorias y patrón local/visitante.

    **Valor para la toma de decisiones:** permite segmentar equipos en grupos
    estadísticamente coherentes, útil para comparar estilos de juego y anticipar
    el comportamiento en futuros enfrentamientos.
    """)
    st.divider()

    km         = models["km_team"]
    ag         = models["ag_team"]
    pca        = models["pca_team"]
    teams_feat = models["teams_features"].copy()
    k_res      = models["k_results_team"]

    from src.features import TEAM_FEATURES, scale_team_features
    X_team, _ = scale_team_features(teams_feat)
    teams_feat["cluster"] = km.labels_
    teams_feat["cluster_ag"] = ag.labels_
    cluster_labels = {0: "Equipos establecidos", 1: "Recién ascendidos / relegados"}
    teams_feat["Perfil"] = teams_feat["cluster"].map(cluster_labels)

    tab1, tab2, tab3 = st.tabs(["Mapa PCA", "Perfiles", "Validación"])

    with tab1:
        coords = pca.transform(X_team.values)
        df_2d  = pd.DataFrame(coords, columns=["PC1", "PC2"], index=X_team.index)
        df_2d["Perfil"] = teams_feat["Perfil"].values
        df_2d["equipo"] = df_2d.index
        var = pca.explained_variance_ratio_

        fig_pca = px.scatter(
            df_2d, x="PC1", y="PC2", color="Perfil", text="equipo",
            color_discrete_sequence=[PALETTE[0], PALETTE[1]],
            labels={"PC1": f"PC1 ({var[0]:.1%} var.)", "PC2": f"PC2 ({var[1]:.1%} var.)"},
            title=f"Perfiles de equipos - PCA 2D (varianza explicada: {sum(var):.1%})",
        )
        fig_pca.update_traces(textposition="top center", marker_size=10)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.success(
            f"**{sum(var):.1%} de la varianza explicada en 2 componentes** - "
            "la separación entre grupos es visualmente clara y estadísticamente sólida."
        )

    with tab2:
        profile = teams_feat.groupby("Perfil")[TEAM_FEATURES].mean().round(3)
        profile.columns = ["GF/partido", "GC/partido", "Dif. goles",
                           "% Victorias", "% Empates", "% Vic. local", "% Vic. visitante"]
        st.dataframe(profile.T, use_container_width=True)

        col1, col2 = st.columns(2)
        for perfil, grp in teams_feat.groupby("Perfil"):
            equipos = sorted(grp.index.tolist())
            if "establecidos" in perfil:
                col1.success(f"**{perfil}** ({len(equipos)} equipos)\n\n" + ", ".join(equipos))
            else:
                col2.warning(f"**{perfil}** ({len(equipos)} equipos)\n\n" + ", ".join(equipos))

        st.markdown("""
        **Interpretación:** la separación refleja la diferencia estructural entre equipos
        históricos de la Premier League y los que accedieron vía ascenso. Los equipos
        del Cluster 1 (Burnley, Luton, Sheffield Utd, Ipswich, Leicester, Southampton)
        corresponden a los descendidos o recién promovidos en ambas temporadas,
        confirmando que el modelo captura un patrón real del torneo.
        """)

    with tab3:
        from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
        ari_team = adjusted_rand_score(km.labels_, ag.labels_)
        sil = silhouette_score(X_team, km.labels_)
        db  = davies_bouldin_score(X_team, km.labels_)

        m1, m2, m3 = st.columns(3)
        m1.metric("ARI (K-Means vs Agglomerative)", f"{ari_team:.4f}")
        m2.metric("Silhouette Score", f"{sil:.4f}")
        m3.metric("Davies-Bouldin Index", f"{db:.4f}")

        st.success(
            f"**ARI = {ari_team:.1f}**, ambos algoritmos producen exactamente los mismos clusters. "
            "Esto es evidencia fuerte de que los 2 grupos existen en los datos y no son "
            "un artefacto del algoritmo elegido."
        )


# ===========================================================================
# ANALISIS TEMPORAL
# ===========================================================================
elif page == "Análisis Temporal":
    st.title("Análisis Temporal de Goles")
    st.markdown("""
    Entender en qué momentos del partido se marcan más goles
    y si los equipos tienen patrones temporales distintos.

    **Valor analítico:** un equipo que marca principalmente en los últimos 15 minutos
    tiene un perfil ofensivo diferente a uno que domina en el primer tiempo.
    Basado en temporada 2023/24.
    """)
    st.divider()

    goals_2023 = data["goals_2023"]
    temporal   = models["temporal_features"].copy()
    km_t       = models["km_temporal"]
    temporal["cluster"] = km_t.labels_

    tab1, tab2, tab3 = st.tabs(["Distribución global", "Por equipo", "Patrones por cluster"])

    with tab1:
        from src.features import MINUTE_LABELS
        goal_blocks = goals_2023[goals_2023["goal_type"] != "OwnGoal"].copy()
        goal_blocks["bloque"] = pd.cut(
            goal_blocks["minute"], bins=[0,15,30,45,60,75,90,120],
            labels=MINUTE_LABELS, right=True,
        )
        dist = goal_blocks["bloque"].value_counts().reindex(MINUTE_LABELS).reset_index()
        dist.columns = ["bloque", "goles"]
        dist["pct"] = dist["goles"] / dist["goles"].sum()

        fig = px.bar(dist, x="bloque", y="goles",
                     color_discrete_sequence=[PALETTE[0]],
                     labels={"bloque": "Bloque de 15 min", "goles": "Goles"},
                     title="Goles por bloque de 15 minutos - Premier League 2023/24",
                     text=dist["pct"].apply(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Hallazgo:** los últimos 15 minutos (76-90 min) concentran el mayor número de goles. "
            "El patrón refleja la tendencia de los equipos a presionar más al final del partido "
            "cuando buscan remontar o ampliar la ventaja. Los primeros 15 minutos son los más seguros "
            "defensivamente."
        )

    with tab2:
        minuto_cols = [c for c in temporal.columns if c.startswith("min_")]
        heatmap_data = temporal[minuto_cols].copy()
        heatmap_data.columns = [c.replace("min_", "") for c in heatmap_data.columns]
        fig_heat = px.imshow(
            heatmap_data, aspect="auto",
            color_continuous_scale="Blues",
            labels={"x": "Bloque", "y": "Equipo", "color": "Prop. goles"},
            title="Proporción de goles por bloque y equipo",
        )
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Colores más intensos indican mayor proporción de goles del equipo en ese bloque.")

    with tab3:
        st.markdown("### 4 patrones temporales distintos identificados por K-Means")
        cluster_means = temporal.groupby("cluster")[minuto_cols].mean()
        cluster_means.columns = [c.replace("min_", "") for c in cluster_means.columns]
        cluster_means.index = [f"Patrón {i}" for i in cluster_means.index]

        fig_c = px.bar(
            cluster_means.T.reset_index().melt(id_vars="index"),
            x="index", y="value", color="variable", barmode="group",
            labels={"index": "Bloque", "value": "Proporción media", "variable": "Patrón"},
            color_discrete_sequence=PALETTE,
            title="Perfil temporal promedio por cluster",
        )
        st.plotly_chart(fig_c, use_container_width=True)

        st.markdown("**Equipos por patrón:**")
        cols = st.columns(2)
        for i, c in enumerate(sorted(temporal["cluster"].unique())):
            equipos = sorted(temporal[temporal["cluster"] == c].index.tolist())
            cols[i % 2].markdown(f"**Patrón {c}:** {", ".join(equipos)}")


# ===========================================================================
# PREDICCION: OVER 2.5
# ===========================================================================
elif page == "Predicción: Over 2.5":
    st.title("Predicción Supervisada - Over/Under 2.5 Goles")
    st.markdown("""
    **Objetivo:** predecir si un partido terminará con más de 2.5 goles totales,
    usando el historial reciente de ambos equipos como información previa al partido.

    **Metodología:** entrenamiento en temporada 2023/24, evaluación en 2024/25
    (split cronológico - el modelo nunca ve datos del futuro durante el entrenamiento).
    """)
    st.divider()

    metrics   = models["supervised_metrics"]
    sup_feat  = models["supervised_features"].copy()
    lr        = models["lr_over25"]
    rf        = models["rf_over25"]
    scaler_s  = models["scaler_supervised"]

    from src.features import SUPERVISED_FEATURES, SUPERVISED_TARGET

    test_df  = sup_feat[sup_feat["season"] == 2024].copy()
    train_df = sup_feat[sup_feat["season"] == 2023].copy()
    X_test_s = pd.DataFrame(scaler_s.transform(test_df[SUPERVISED_FEATURES]), columns=SUPERVISED_FEATURES)
    y_test_s = test_df[SUPERVISED_TARGET].values
    majority_acc = float((test_df[SUPERVISED_TARGET] == test_df[SUPERVISED_TARGET].mode()[0]).mean())

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Comparación de modelos", "Simulador", "Curvas ROC", "Features", "Calibración"
    ])

    with tab1:
        st.subheader("Métricas en conjunto de test (2024/25)")

        baseline_row = {
            "model": "Siempre predecir Over",
            "accuracy": round(majority_acc, 4),
            "f1": round(2*majority_acc/(1+majority_acc), 4),
            "roc_auc": 0.5, "brier": round(majority_acc*(1-majority_acc), 4), "log_loss": 0.693
        }
        all_rows = [baseline_row] + list(metrics.values())
        metrics_df = pd.DataFrame(all_rows).set_index("model")
        st.dataframe(
            metrics_df.style
            .highlight_max(axis=0, color="#c8e6c9")
            .highlight_min(subset=["brier", "log_loss"], axis=0, color="#c8e6c9")
            .format("{:.4f}"),
            use_container_width=True,
        )

        st.warning(
            "**Resultado honesto:** los modelos ML igualan al baseline más simple "
            "(siempre predecir Over). El ROC-AUC de ~0.53 indica señal mínima pero real. "
            "Esto es esperado con solo historial de goles - se requieren features adicionales "
            "(xG, tiros al arco) para mejorar la discriminación."
        )

        st.subheader("Matriz de confusión - Random Forest")
        from sklearn.metrics import confusion_matrix
        y_pred_rf = rf.predict(X_test_s)
        cm = confusion_matrix(y_test_s, y_pred_rf)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Pred. Under", "Pred. Over"],
            y=["Real Under", "Real Over"],
            color_continuous_scale="Blues",
            title="Matriz de confusión - Random Forest (test 2024/25)",
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"Verdaderos positivos (Over correctos): **{tp}** · "
            f"Verdaderos negativos (Under correctos): **{tn}** · "
            f"Falsos positivos: **{fp}** · Falsos negativos: **{fn}**"
        )

    with tab2:
        st.subheader("Simulador de predicción")
        st.markdown(
            "Selecciona dos equipos y el modelo calcula la probabilidad de Over 2.5 "
            "basándose en su forma real de los últimos 5 partidos."
        )
        from src.features import _team_rolling_stats
        all_matches = data["matches"].sort_values("date")
        equipos = sorted(pd.unique(pd.concat([all_matches["home_team"], all_matches["away_team"]])))
        last_date = all_matches["date"].max()

        c1, c2 = st.columns(2)
        home_pick = c1.selectbox("Equipo local", equipos,
                                 index=equipos.index("Arsenal") if "Arsenal" in equipos else 0)
        away_pick = c2.selectbox("Equipo visitante",
                                 [e for e in equipos if e != home_pick], index=0)

        if st.button("Predecir", type="primary"):
            home_s = _team_rolling_stats(home_pick, last_date + pd.Timedelta(days=1), all_matches, 5)
            away_s = _team_rolling_stats(away_pick, last_date + pd.Timedelta(days=1), all_matches, 5)
            if home_s is None or away_s is None:
                st.error("No hay suficiente historial para uno de los equipos.")
            else:
                feat_row = pd.DataFrame([{
                    "home_scored_5": home_s["scored"], "home_conceded_5": home_s["conceded"],
                    "home_form_5": home_s["form"], "away_scored_5": away_s["scored"],
                    "away_conceded_5": away_s["conceded"], "away_form_5": away_s["form"],
                    "combined_attack_5": home_s["scored"] + away_s["scored"],
                }])[SUPERVISED_FEATURES]
                X_pred = pd.DataFrame(scaler_s.transform(feat_row), columns=SUPERVISED_FEATURES)
                prob_lr  = lr.predict_proba(X_pred)[0, 1]
                prob_rf  = rf.predict_proba(X_pred)[0, 1]
                prob_avg = (prob_lr + prob_rf) / 2

                st.divider()
                m1, m2, m3 = st.columns(3)
                m1.metric("Regresión Logística", f"{prob_lr:.1%}")
                m2.metric("Random Forest",        f"{prob_rf:.1%}")
                m3.metric("Promedio modelos",     f"{prob_avg:.1%}")

                veredicto = "Over 2.5" if prob_avg >= 0.5 else "Under 2.5"
                nivel = "Alta" if abs(prob_avg - 0.5) > 0.15 else "Moderada"
                color = "🟢" if prob_avg >= 0.5 else "🔴"
                st.markdown(f"### {color} Predicción: **{veredicto}** - Confianza {nivel} ({prob_avg:.1%})")

                st.divider()
                fc1, fc2 = st.columns(2)
                fc1.markdown(f"**{home_pick}** - últimos 5 partidos")
                fc1.write(f"Goles anotados/partido: {home_s["scored"]:.2f}")
                fc1.write(f"Goles recibidos/partido: {home_s["conceded"]:.2f}")
                fc1.write(f"Puntos/partido: {home_s["form"]:.2f}")
                fc2.markdown(f"**{away_pick}** - últimos 5 partidos")
                fc2.write(f"Goles anotados/partido: {away_s["scored"]:.2f}")
                fc2.write(f"Goles recibidos/partido: {away_s["conceded"]:.2f}")
                fc2.write(f"Puntos/partido: {away_s["form"]:.2f}")

                st.caption("⚠️ ROC-AUC ~0.53 - predicción orientativa con señal débil.")

    with tab3:
        from sklearn.metrics import roc_curve
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        for model_obj, name, color in [(lr, "Logistic Regression", PALETTE[0]),
                                        (rf, "Random Forest", PALETTE[1])]:
            y_proba = model_obj.predict_proba(X_test_s)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_s, y_proba)
            auc = metrics[name.lower().replace(" ", "_")]["roc_auc"]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                          name=f"{name} (AUC={auc:.3f})",
                                          line=dict(color=color, width=2)))
        fig_roc.update_layout(
            title="Curvas ROC - Test 2024/25",
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("Un AUC cercano a 0.5 indica discriminación cercana al azar. "
                   "El modelo aprende señal mínima pero real con estas features.")

    with tab4:
        st.subheader("Importancia de features - Random Forest")
        feat_imp = pd.Series(rf.feature_importances_, index=SUPERVISED_FEATURES).sort_values()
        fig_imp = px.bar(feat_imp, x=feat_imp.values, y=feat_imp.index, orientation="h",
                         color_discrete_sequence=[PALETTE[0]],
                         labels={"x": "Importancia", "y": "Feature"},
                         title="Importancia de variables")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Coeficientes - Regresión Logística")
        coef = pd.Series(lr.coef_[0], index=SUPERVISED_FEATURES).sort_values()
        fig_coef = px.bar(coef, x=coef.values, y=coef.index, orientation="h",
                          color=coef.values > 0,
                          color_discrete_map={True: PALETTE[0], False: PALETTE[1]},
                          labels={"x": "Coeficiente", "y": "Feature", "color": "Dirección"},
                          title="Coeficientes de la Regresión Logística")
        st.plotly_chart(fig_coef, use_container_width=True)
        st.info("`combined_attack_5` es la variable más relevante: la capacidad ofensiva "
                "combinada de ambos equipos en sus últimos 5 partidos es el mejor predictor disponible.")

    with tab5:
        from sklearn.calibration import calibration_curve as sk_cal
        st.subheader("Curva de calibración")
        st.markdown("Una curva bien calibrada sigue la diagonal: si el modelo dice 60%, "
                    "acierta ~60% de las veces.")
        fig_cal = go.Figure()
        fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        for model_obj, name, color in [(lr, "Logistic Regression", PALETTE[0]),
                                        (rf, "Random Forest", PALETTE[1])]:
            y_proba = model_obj.predict_proba(X_test_s)[:, 1]
            frac, mean_p = sk_cal(y_test_s, y_proba, n_bins=8)
            fig_cal.add_trace(go.Scatter(x=mean_p, y=frac, mode="lines+markers",
                                          name=name, line=dict(color=color, width=2)))
        fig_cal.update_layout(xaxis_title="Probabilidad predicha",
                               yaxis_title="Fracción positivos reales",
                               xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_cal, use_container_width=True)


# ===========================================================================
# GOLEADORES
# ===========================================================================
elif page == "Goleadores":
    st.title("Ranking de Goleadores")
    st.markdown("Top anotadores de la Premier League por temporada, basado en los "
                "eventos de gol registrados (excluye autogoles).")

    season_filter = st.selectbox("Temporada", ["Todas", "2023", "2024"])

    from src.db import get_top_scorers
    season_arg = None if season_filter == "Todas" else int(season_filter)
    scorers = get_top_scorers(season=season_arg, limit=30)

    top10 = scorers.head(10)
    fig = px.bar(top10, x="goles", y="player", orientation="h", color="team",
                 labels={"goles": "Goles", "player": "Jugador", "team": "Equipo"},
                 title=f"Top 10 goleadores - {season_filter}")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.dataframe(scorers.reset_index(drop=True),
                 column_config={"player": "Jugador", "team": "Equipo",
                                "goles": st.column_config.NumberColumn("Goles", format="%d")},
                 use_container_width=True, hide_index=True)


# ===========================================================================
# CONCLUSIONES
# ===========================================================================
elif page == "Conclusiones":
    st.title("Conclusiones y Hallazgos")
    st.divider()

    st.markdown("## Hallazgos principales")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### No supervisado")
        st.success("""
        **Partidos - 3 clusters:**
        - 53% son partidos de baja intensidad (~1.9 goles)
        - 26% con dominio local claro (~4.5 goles)
        - 21% con dominio visitante (~4.5 goles)

        Los partidos de alto marcador son dominantes - raramente
        ambos equipos anotan mucho en el mismo partido.
        """)
        st.success("""
        **Equipos - 2 clusters (ARI=1.0):**
        - Separación clara entre equipos históricos de la PL
          y recién ascendidos/relegados
        - Ambos algoritmos (K-Means y Agglomerative) coinciden
          perfectamente, confirmando que los grupos son reales
        """)
        st.success("""
        **Temporal - 4 patrones:**
        - Los goles se concentran en el tramo 76-90 min
        - Los equipos tienen perfiles temporales distintos
          y agrupables en 4 patrones estadísticos
        """)

    with col2:
        st.markdown("### Supervisado")
        st.warning("""
        **Predicción Over/Under 2.5:**
        - Accuracy ~57% en test (igual al baseline)
        - ROC-AUC ~0.53 - señal débil pero real
        - El historial de goles solo no es suficiente
          para predecir con poder discriminatorio

        **Limitación principal:** sin features de xG,
        tiros al arco o estadísticas de jugadores, la
        señal predictiva disponible es inherentemente débil.
        """)

    st.divider()
    st.markdown("## Limitaciones del proyecto")
    st.markdown("""
    | Limitación | Impacto | Solución futura |
    |-----------|---------|-----------------|
    | Plan gratuito API: 100 req/día | Extracción lenta (días) | Plan de pago por un mes |
    | Sin xG ni estadísticas por partido | Modelos supervisados débiles | Endpoints `/statistics` |
    | Sin datos de plantilla/lesiones | No se modela estado del equipo | Endpoints `/injuries`, `/lineups` |
    """)

    st.divider()
    st.markdown("## Trabajo futuro")
    st.markdown("""
    - **Features avanzadas:** xG, tiros al arco, posesión.
    - **Estado de plantilla:** lesionados, rotaciones, titulares confirmados
    - **Más targets:** BTTS (ambos equipos anotan), gol en el primer tiempo
    """)

    st.divider()
    st.markdown("""
    ---
    ### **TEL354 - Minería de Datos · UTFSM**
    Vicente Beiza · Lucas Andrade · Sebastián Vicuña
    """)
