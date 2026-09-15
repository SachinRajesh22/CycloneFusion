from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd


DEFAULT_DATASET = Path("dataset/odisha_cyclone_tracks_clean.csv")
FALLBACK_DATASETS = [
    DEFAULT_DATASET,
    Path("odisha_cyclone_tracks_clean.csv"),
    Path("odisha_cyclone_tracks_clean(1).csv"),
]

EXPECTED_COLUMNS = [
    "storm_id",
    "season",
    "name",
    "timestamp_utc",
    "nature",
    "latitude",
    "longitude",
    "wmo_wind_kt",
    "wmo_pressure_mb",
    "storm_speed_kt",
    "storm_direction_deg",
    "distance_to_land_km",
    "landfall_next3h_km",
    "storm_max_wind_kt",
    "storm_min_pressure_mb",
    "landfall_window_flag",
    "on_land_flag",
    "odisha_role",
]

NUMERIC_COLUMNS = [
    "season",
    "latitude",
    "longitude",
    "wmo_wind_kt",
    "wmo_pressure_mb",
    "storm_speed_kt",
    "storm_direction_deg",
    "distance_to_land_km",
    "landfall_next3h_km",
    "storm_max_wind_kt",
    "storm_min_pressure_mb",
    "landfall_window_flag",
    "on_land_flag",
]


def resolve_dataset_path() -> Path | None:
    for path in FALLBACK_DATASETS:
        if path.exists() and path.is_file():
            return path
    return None


def dataset_exists(path: Path = DEFAULT_DATASET) -> bool:
    return path.exists() and path.is_file()


def load_cyclone_data(source: str | Path | BinaryIO) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_csv(source)
    original_rows = len(df)
    original_columns = list(df.columns)

    df.columns = [str(col).strip() for col in df.columns]
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_columns = ["storm_id", "name", "nature", "odisha_role"]
    for col in text_columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

    df = df.sort_values(["storm_id", "timestamp_utc"], na_position="last").reset_index(drop=True)
    df["observation_id"] = df.index.astype(int)
    df["storm_label"] = df.apply(_storm_label, axis=1)
    df["timestamp_label"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M UTC").fillna("Unknown time")

    missing_summary = (
        df[EXPECTED_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .sort_values(ascending=False)
        .to_dict()
    )

    metadata = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "missing_columns": missing_columns,
        "missing_summary": missing_summary,
    }
    return df, metadata


def _storm_label(row: pd.Series) -> str:
    name = row.get("name", "Unknown")
    storm_id = row.get("storm_id", "Unknown")
    season = row.get("season")
    season_label = "Unknown season" if pd.isna(season) else str(int(season))
    return f"{name} ({season_label}) · {storm_id}"
