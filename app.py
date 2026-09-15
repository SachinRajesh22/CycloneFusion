from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import DEFAULT_DATASET, load_cyclone_data, resolve_dataset_path
from fingerprint import data_quality, fingerprint_table, normalized_vector, radar_values, selected_row, vector_text
from preprocessing import FEATURE_LABELS, SIMILARITY_FEATURES, add_engineered_features, build_feature_frame
from risk_engine import historical_risk_signal, preparedness_level
from similarity import feature_contributions, top_historical_twins, weights_table
from temporal_matching import current_history, future_evolution, temporal_similarity
from visualization import cyclone_map, evolution_chart, odisha_focus_map, radar_chart, time_series


st.set_page_config(
    page_title="CycloneFusion",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_theme()
    st.sidebar.title("CycloneFusion")
    st.sidebar.caption("Historical Cyclone Twin & Early Risk Intelligence")

    df, metadata = load_data_panel()
    if df is None:
        st.markdown(
            """
            <div class="hero">
              <div>
                <p class="eyebrow">Dataset required</p>
                <h1>CycloneFusion</h1>
                <p>Place the dataset at <code>dataset/odisha_cyclone_tracks_clean.csv</code> or upload a CSV from the sidebar.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning("No historical observations are loaded, so the app will not fabricate sample cyclone data.")
        return

    df = add_engineered_features(df)
    normalized_features, _, _ = build_feature_frame(df)
    normalized_features.index = df["observation_id"]

    selected_id = selection_panel(df)
    row = selected_row(df, selected_id)
    twins = top_historical_twins(df, normalized_features, selected_id, top_k=5)

    page = st.sidebar.radio(
        "Dashboard pages",
        [
            "Overview",
            "Cyclone Explorer",
            "Cyclone Fingerprint",
            "Historical Twins",
            "What Happened Next?",
            "Odisha Impact / Preparedness",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption("Prototype boundary")
    st.sidebar.info("Decision-support demo only. Follow official IMD and government advisories for actual action.")

    if page == "Overview":
        overview_page(df, row, selected_id, metadata, twins)
    elif page == "Cyclone Explorer":
        explorer_page(df, row, selected_id)
    elif page == "Cyclone Fingerprint":
        fingerprint_page(df, row, selected_id, normalized_features)
    elif page == "Historical Twins":
        twins_page(df, row, selected_id, normalized_features, twins)
    elif page == "What Happened Next?":
        happened_next_page(df, row, selected_id, normalized_features, twins)
    else:
        preparedness_page(df, row, selected_id, twins)


@st.cache_data(show_spinner=False)
def cached_load_from_path(path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    return load_cyclone_data(Path(path))


def load_data_panel() -> tuple[pd.DataFrame | None, dict[str, object] | None]:
    uploaded = st.sidebar.file_uploader("Upload cyclone CSV", type=["csv"])
    try:
        if uploaded is not None:
            return load_cyclone_data(uploaded)
        dataset_path = resolve_dataset_path()
        if dataset_path is not None:
            st.sidebar.success(f"Loaded {dataset_path}")
            return cached_load_from_path(str(dataset_path))
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.sidebar.error(f"Could not load dataset: {exc}")
        return None, None
    return None, None


def selection_panel(df: pd.DataFrame) -> int:
    st.sidebar.divider()
    st.sidebar.subheader("Select cyclone stage")
    storm_labels = df["storm_label"].drop_duplicates().tolist()
    default_storm = _default_demo_storm(df)
    storm_label = st.sidebar.selectbox("Cyclone", storm_labels, index=storm_labels.index(default_storm))
    storm_df = df[df["storm_label"] == storm_label].sort_values("timestamp_utc")

    ids = storm_df["storm_id"].drop_duplicates().tolist()
    selected_storm_id = st.sidebar.selectbox("Storm ID", ids)
    storm_df = storm_df[storm_df["storm_id"] == selected_storm_id]

    labels = storm_df["timestamp_label"].tolist()
    default_index = max(0, min(len(labels) - 1, int(len(labels) * 0.65)))
    timestamp_label = st.sidebar.selectbox("Observation timestamp", labels, index=default_index)
    selected = storm_df[storm_df["timestamp_label"] == timestamp_label].iloc[0]
    return int(selected["observation_id"])


def overview_page(df: pd.DataFrame, row: pd.Series, selected_id: int, metadata: dict[str, object] | None, twins: pd.DataFrame) -> None:
    title_block("Overview", "Current stage → fingerprint → historical twins → observed future evolution")
    strongest = df.loc[df["wmo_wind_kt"].idxmax()] if df["wmo_wind_kt"].notna().any() else pd.Series()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Historical observations", f"{len(df):,}")
    c2.metric("Unique storms", f"{df['storm_id'].nunique():,}")
    c3.metric("Seasons", f"{df['season'].nunique():,}")
    c4.metric("Strongest storm", _storm_metric(strongest))
    c5.metric("Selected cyclone", str(row.get("name", "Unknown")))

    st.plotly_chart(cyclone_map(df, selected_id, twin_df=twins), width="stretch")

    st.subheader("Selected cyclone state")
    metrics_grid(row)

    with st.expander("Dataset quality snapshot"):
        if metadata:
            st.write(f"Loaded rows: {metadata['original_rows']}")
            if metadata["missing_columns"]:
                st.warning(f"Missing expected columns: {', '.join(metadata['missing_columns'])}")
            st.dataframe(pd.DataFrame(metadata["missing_summary"].items(), columns=["Column", "Missing %"]), width="stretch")


def explorer_page(df: pd.DataFrame, row: pd.Series, selected_id: int) -> None:
    title_block("Cyclone Explorer", "Track, state, and observation-time evolution")
    storm = df[df["storm_id"] == row["storm_id"]].sort_values("timestamp_utc")
    metrics_grid(row)
    st.plotly_chart(cyclone_map(storm, selected_id, title=f"Historical Track · {row['storm_label']}"), width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(time_series(storm, "wmo_wind_kt", "Wind vs time", "Wind (kt)"), width="stretch")
    c2.plotly_chart(time_series(storm, "wmo_pressure_mb", "Pressure vs time", "Pressure (mb)"), width="stretch")
    c3.plotly_chart(time_series(storm, "distance_to_land_km", "Distance to land vs time", "Distance (km)"), width="stretch")

    st.subheader("Current cyclone state")
    st.dataframe(_state_table(row), width="stretch", hide_index=True)


def fingerprint_page(df: pd.DataFrame, row: pd.Series, selected_id: int, normalized_features: pd.DataFrame) -> None:
    title_block("Cyclone Fingerprint", "Normalized cyclone-state vector and explainable feature quality")
    normalized = normalized_vector(normalized_features, selected_id)
    quality = data_quality(row)

    c1, c2 = st.columns([1.05, 1])
    c1.plotly_chart(radar_chart(radar_values(df, row)), width="stretch")
    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.metric("Fingerprint timestamp", row.get("timestamp_label", "Unknown"))
        st.metric("Feature completeness", f"{quality['percent']}%", f"{quality['usable']} / {quality['total']} usable")
        missing = quality["missing"]
        if missing:
            st.warning("Missing features: " + ", ".join(FEATURE_LABELS.get(item, item) for item in missing))
        else:
            st.success("All fingerprint features are present for this observation.")
        st.code(vector_text(normalized), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Feature table")
    st.dataframe(fingerprint_table(row, normalized), width="stretch", hide_index=True)


def twins_page(
    df: pd.DataFrame,
    row: pd.Series,
    selected_id: int,
    normalized_features: pd.DataFrame,
    twins: pd.DataFrame,
) -> None:
    title_block("Historical Twins", "Weighted normalized-distance search across historical cyclone stages")
    st.caption("Similarity score = 100 / (1 + weighted Euclidean distance). Exact selected observation is excluded; different storms are preferred when enough candidates exist.")

    if twins.empty:
        st.warning("No candidate historical twins are available after excluding the selected observation.")
        return

    st.dataframe(weights_table(), width="stretch", hide_index=True)
    twin_id = twin_cards(twins)

    c1, c2 = st.columns([1.25, 1])
    selected_twin = df[df["observation_id"] == twin_id].iloc[0]
    twin_storm = df[df["storm_id"] == selected_twin["storm_id"]].sort_values("timestamp_utc")
    c1.plotly_chart(cyclone_map(twin_storm, twin_id, title=f"Selected Twin Track · {selected_twin['storm_label']}"), width="stretch")
    c2.subheader("Top feature differences")
    c2.dataframe(feature_contributions(normalized_features, selected_id, twin_id).head(8), width="stretch", hide_index=True)

    evo = temporal_similarity(df, normalized_features, selected_id, twin_id)
    if evo["available"]:
        st.info(f"Evolution Similarity: {evo['score']:.1f} using the latest {evo['steps']} aligned observations. This compares historical evolution, not a validated forecast.")
    else:
        st.warning("Evolution Similarity needs at least two aligned observations for both storms.")


def happened_next_page(
    df: pd.DataFrame,
    row: pd.Series,
    selected_id: int,
    normalized_features: pd.DataFrame,
    twins: pd.DataFrame,
) -> None:
    title_block("What Happened Next?", "Observed historical evolution after the matched twin stage")
    if twins.empty:
        st.warning("No historical twin is available to analyze.")
        return

    twin_id = twin_selector(twins)
    twin_row = df[df["observation_id"] == twin_id].iloc[0]
    future = future_evolution(df, twin_id, max_hours=48)
    history = current_history(df, selected_id, max_hours=48)

    st.info("Storms with similar historical fingerprints showed the following subsequent evolution. This is not a guaranteed future prediction.")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(evolution_chart(history, future, "wmo_wind_kt", "Wind evolution", "Wind (kt)"), width="stretch")
    c2.plotly_chart(evolution_chart(history, future, "wmo_pressure_mb", "Pressure evolution", "Pressure (mb)"), width="stretch")
    c3.plotly_chart(evolution_chart(history, future, "distance_to_land_km", "Distance-to-land evolution", "Distance (km)"), width="stretch")

    st.subheader("Historical Twin Evolution")
    display = future[["hours_from_match", "timestamp_label", "name", "nature", "wmo_wind_kt", "wmo_pressure_mb", "distance_to_land_km", "latitude", "longitude"]].copy()
    display["hours_from_match"] = display["hours_from_match"].round(1)
    st.dataframe(display, width="stretch", hide_index=True)

    evo = temporal_similarity(df, normalized_features, selected_id, twin_id)
    if evo["available"]:
        st.caption(f"Evolution Similarity for this twin: {evo['score']:.1f} over {evo['steps']} aligned observations.")
    st.caption(f"Selected twin: {twin_row['storm_label']} at {twin_row['timestamp_label']}")


def preparedness_page(df: pd.DataFrame, row: pd.Series, selected_id: int, twins: pd.DataFrame) -> None:
    title_block("Odisha Impact / Preparedness", "Prototype Historical-Risk Signal and decision-support suggestions")
    risk = historical_risk_signal(df, row, twins)
    level, suggestions = preparedness_level(risk["label"])

    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.metric("Prototype Historical-Risk Signal", risk["label"], f"evidence score {risk['score']}")
    c2.metric("Preparedness Assistant", level)
    c3.metric("Distance to land", _fmt(row.get("distance_to_land_km"), "km"))

    st.plotly_chart(odisha_focus_map(df[df["storm_id"] == row["storm_id"]], selected_id), width="stretch")

    st.subheader("Why this signal?")
    for item in risk["evidence"]:
        st.markdown(f"- {item}")

    st.subheader("Preparedness Assistant")
    for item in suggestions:
        st.markdown(f"- {item}")
    st.warning("AI-generated decision-support suggestions. Follow official IMD and government advisories for actual action.")


def twin_cards(twins: pd.DataFrame) -> int:
    options = {}
    cols = st.columns(min(5, len(twins)))
    for index, (_, twin) in enumerate(twins.iterrows()):
        label = f"#{index + 1} · {twin['name']} · {twin['timestamp_label']}"
        options[label] = int(twin["observation_id"])
        with cols[index % len(cols)]:
            st.markdown(
                f"""
                <div class="twin-card">
                  <p class="eyebrow">Historical Twin #{index + 1}</p>
                  <h3>{twin['name']} ({int(twin['season']) if pd.notna(twin['season']) else 'Unknown'})</h3>
                  <p>{twin['timestamp_label']}</p>
                  <strong>{twin['similarity_score']:.1f}</strong><span> similarity</span>
                  <p>Wind {_fmt(twin.get('wmo_wind_kt'), 'kt')} · Pressure {_fmt(twin.get('wmo_pressure_mb'), 'mb')} · Land {_fmt(twin.get('distance_to_land_km'), 'km')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    selected_label = st.selectbox("Select a historical twin", list(options.keys()))
    return options[selected_label]


def twin_selector(twins: pd.DataFrame) -> int:
    labels = {
        f"#{index + 1} · {twin['name']} · {twin['timestamp_label']} · {twin['similarity_score']:.1f}": int(twin["observation_id"])
        for index, (_, twin) in enumerate(twins.iterrows())
    }
    return labels[st.selectbox("Historical twin", list(labels.keys()))]


def metrics_grid(row: pd.Series) -> None:
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Wind", _fmt(row.get("wmo_wind_kt"), "kt"))
    c2.metric("Pressure", _fmt(row.get("wmo_pressure_mb"), "mb"))
    c3.metric("Latitude", _fmt(row.get("latitude"), "°"))
    c4.metric("Longitude", _fmt(row.get("longitude"), "°"))
    c5.metric("Speed", _fmt(row.get("storm_speed_kt"), "kt"))
    c6.metric("Direction", _fmt(row.get("storm_direction_deg"), "°"))
    c7.metric("Distance to land", _fmt(row.get("distance_to_land_km"), "km"))
    c8.metric("Landfall proximity", _fmt(row.get("landfall_next3h_km"), "km"))


def title_block(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <p class="eyebrow">CycloneFusion</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _default_demo_storm(df: pd.DataFrame) -> str:
    scored = (
        df.groupby("storm_label")
        .agg(max_wind=("wmo_wind_kt", "max"), min_distance=("distance_to_land_km", "min"), count=("observation_id", "count"))
        .fillna({"max_wind": 0, "min_distance": 9999, "count": 0})
    )
    scored["score"] = scored["max_wind"] - scored["min_distance"].clip(upper=500) / 10 + scored["count"]
    return str(scored.sort_values("score", ascending=False).index[0])


def _state_table(row: pd.Series) -> pd.DataFrame:
    fields = [
        ("Name", row.get("name")),
        ("Date/time", row.get("timestamp_label")),
        ("Wind speed", _fmt(row.get("wmo_wind_kt"), "kt")),
        ("Pressure", _fmt(row.get("wmo_pressure_mb"), "mb")),
        ("Position", f"{_fmt(row.get('latitude'), '°')}, {_fmt(row.get('longitude'), '°')}"),
        ("Storm speed", _fmt(row.get("storm_speed_kt"), "kt")),
        ("Direction", _fmt(row.get("storm_direction_deg"), "°")),
        ("Distance to land", _fmt(row.get("distance_to_land_km"), "km")),
        ("Landfall proximity", _fmt(row.get("landfall_next3h_km"), "km")),
        ("Odisha role", row.get("odisha_role")),
    ]
    return pd.DataFrame(fields, columns=["Signal", "Value"])


def _storm_metric(row: pd.Series) -> str:
    if row.empty:
        return "Unknown"
    return f"{row.get('name', 'Unknown')} · {_fmt(row.get('wmo_wind_kt'), 'kt')}"


def _fmt(value: object, suffix: str = "") -> str:
    if pd.isna(value):
        return "Missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 0.01:
        return f"{int(round(number))}{suffix}"
    return f"{number:.1f}{suffix}"


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #07111f;
          --panel: rgba(15, 23, 42, .72);
          --line: rgba(148, 163, 184, .24);
          --text: #dbeafe;
          --muted: #94a3b8;
          --cyan: #22d3ee;
          --amber: #facc15;
        }
        .stApp {
          background:
            radial-gradient(circle at 20% 0%, rgba(14, 116, 144, .22), transparent 28%),
            linear-gradient(135deg, #07111f 0%, #0f172a 48%, #111827 100%);
          color: var(--text);
        }
        section[data-testid="stSidebar"] {
          background: rgba(2, 6, 23, .92);
          border-right: 1px solid var(--line);
        }
        .hero {
          border: 1px solid var(--line);
          background: linear-gradient(135deg, rgba(8, 47, 73, .78), rgba(15, 23, 42, .78));
          padding: 26px 28px;
          border-radius: 8px;
          margin-bottom: 18px;
          box-shadow: 0 24px 70px rgba(0,0,0,.25);
        }
        .hero h1 {
          margin: 0 0 6px 0;
          font-size: 2.25rem;
          letter-spacing: 0;
          color: #f8fafc;
        }
        .hero p {
          margin: 0;
          color: #bfdbfe;
          max-width: 960px;
        }
        .eyebrow {
          color: var(--cyan) !important;
          font-size: .76rem;
          text-transform: uppercase;
          letter-spacing: .12em;
          font-weight: 700;
          margin: 0 0 8px 0 !important;
        }
        .twin-card, .panel {
          border: 1px solid var(--line);
          background: var(--panel);
          border-radius: 8px;
          padding: 16px;
          min-height: 190px;
        }
        .twin-card h3 {
          color: #f8fafc;
          font-size: 1rem;
          margin: 0 0 8px 0;
        }
        .twin-card strong {
          color: var(--amber);
          font-size: 2rem;
        }
        div[data-testid="stMetric"] {
          background: rgba(15, 23, 42, .58);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px;
        }
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
