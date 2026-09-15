from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORWAY = ["#22d3ee", "#f97316", "#a7f3d0", "#facc15", "#c084fc", "#f87171"]


def cyclone_map(
    df: pd.DataFrame,
    selected_observation_id: int | None = None,
    title: str = "Cyclone Track Map",
    twin_df: pd.DataFrame | None = None,
) -> go.Figure:
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    fig = go.Figure()

    for storm_label, storm in valid.groupby("storm_label"):
        storm = storm.sort_values("timestamp_utc")
        fig.add_trace(
            go.Scattermap(
                lat=storm["latitude"],
                lon=storm["longitude"],
                mode="lines+markers",
                marker={"size": 6, "opacity": 0.65},
                line={"width": 2},
                name=str(storm_label)[:48],
                text=storm["timestamp_label"],
                hovertemplate="%{text}<br>Lat %{lat:.2f}, Lon %{lon:.2f}<extra></extra>",
                showlegend=False,
            )
        )

    if selected_observation_id is not None and not valid.empty:
        selected = valid[valid["observation_id"] == selected_observation_id]
        if not selected.empty:
            row = selected.iloc[0]
            fig.add_trace(
                go.Scattermap(
                    lat=[row["latitude"]],
                    lon=[row["longitude"]],
                    mode="markers",
                    marker={"size": 18, "color": "#facc15", "symbol": "circle"},
                    name="Selected stage",
                    text=[row["storm_label"]],
                    hovertemplate="%{text}<br>Selected stage<extra></extra>",
                )
            )

    if twin_df is not None and not twin_df.empty:
        twin_valid = twin_df.dropna(subset=["latitude", "longitude"])
        fig.add_trace(
            go.Scattermap(
                lat=twin_valid["latitude"],
                lon=twin_valid["longitude"],
                mode="markers",
                marker={"size": 13, "color": "#fb7185"},
                name="Historical twins",
                text=twin_valid["storm_label"],
                hovertemplate="%{text}<br>Twin match<extra></extra>",
            )
        )

    center = _map_center(valid)
    fig.update_layout(
        title=title,
        map={"style": "open-street-map", "center": center, "zoom": 4.7},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbeafe"},
        colorway=COLORWAY,
    )
    return fig


def time_series(storm: pd.DataFrame, y: str, title: str, label: str) -> go.Figure:
    fig = px.line(
        storm.sort_values("timestamp_utc"),
        x="timestamp_utc",
        y=y,
        markers=True,
        title=title,
        labels={"timestamp_utc": "Time", y: label},
        color_discrete_sequence=["#22d3ee"],
    )
    fig.update_layout(_dark_layout(height=300))
    return fig


def radar_chart(radar_df: pd.DataFrame) -> go.Figure:
    categories = radar_df["Axis"].tolist()
    values = radar_df["Score"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            name="Cyclone fingerprint",
            line={"color": "#22d3ee"},
            fillcolor="rgba(34, 211, 238, 0.22)",
        )
    )
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 100], "gridcolor": "rgba(148,163,184,.25)"},
            "angularaxis": {"gridcolor": "rgba(148,163,184,.2)"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=False,
        height=430,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbeafe"},
    )
    return fig


def evolution_chart(current_history: pd.DataFrame, twin_future: pd.DataFrame, y: str, title: str, label: str) -> go.Figure:
    fig = go.Figure()
    if not current_history.empty:
        fig.add_trace(
            go.Scatter(
                x=current_history["hours_from_now"],
                y=current_history[y],
                mode="lines+markers",
                name="Selected cyclone history",
                line={"color": "#22d3ee"},
            )
        )
    if not twin_future.empty:
        fig.add_trace(
            go.Scatter(
                x=twin_future["hours_from_match"],
                y=twin_future[y],
                mode="lines+markers",
                name="Historical twin future",
                line={"color": "#f97316"},
            )
        )
    fig.update_layout(_dark_layout(height=330, title=title))
    fig.update_xaxes(title="Hours from matched stage")
    fig.update_yaxes(title=label)
    return fig


def odisha_focus_map(df: pd.DataFrame, selected_observation_id: int) -> go.Figure:
    fig = cyclone_map(df, selected_observation_id, title="Odisha Coastal Context")
    fig.add_trace(
        go.Scattermap(
            lat=[20.2961],
            lon=[85.8245],
            mode="markers+text",
            marker={"size": 11, "color": "#a7f3d0"},
            text=["Bhubaneswar"],
            textposition="top right",
            name="Odisha reference",
            hovertemplate="Odisha reference point<extra></extra>",
        )
    )
    fig.update_layout(map={"style": "open-street-map", "center": {"lat": 19.8, "lon": 85.4}, "zoom": 5.5})
    return fig


def _map_center(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"lat": 20.2, "lon": 86.0}
    return {"lat": float(df["latitude"].mean()), "lon": float(df["longitude"].mean())}


def _dark_layout(height: int = 320, title: str | None = None) -> dict:
    layout = {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,23,42,.35)",
        "font": {"color": "#dbeafe"},
        "margin": {"l": 20, "r": 20, "t": 48, "b": 30},
        "legend": {"orientation": "h", "y": -0.2},
    }
    if title:
        layout["title"] = title
    return layout
