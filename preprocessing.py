from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = [
    "wmo_wind_kt",
    "wmo_pressure_mb",
    "latitude",
    "longitude",
    "storm_speed_kt",
    "storm_direction_deg",
    "distance_to_land_km",
    "landfall_next3h_km",
]

TREND_FEATURES = [
    "wind_trend_kt",
    "pressure_trend_mb",
    "distance_trend_km",
]

SIMILARITY_FEATURES = BASE_FEATURES + TREND_FEATURES

FEATURE_LABELS = {
    "wmo_wind_kt": "Wind",
    "wmo_pressure_mb": "Pressure",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "storm_speed_kt": "Storm speed",
    "storm_direction_deg": "Direction",
    "distance_to_land_km": "Distance to land",
    "landfall_next3h_km": "Landfall proximity",
    "wind_trend_kt": "Wind trend",
    "pressure_trend_mb": "Pressure trend",
    "distance_trend_km": "Approach trend",
}

FEATURE_WEIGHTS = {
    "wmo_wind_kt": 1.25,
    "wmo_pressure_mb": 1.25,
    "latitude": 0.8,
    "longitude": 0.8,
    "storm_speed_kt": 0.85,
    "storm_direction_deg": 0.6,
    "distance_to_land_km": 1.15,
    "landfall_next3h_km": 1.0,
    "wind_trend_kt": 1.2,
    "pressure_trend_mb": 1.2,
    "distance_trend_km": 0.9,
}


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["storm_id", "timestamp_utc"], na_position="last").reset_index(drop=True)
    out["observation_id"] = out.index.astype(int)

    grouped = out.groupby("storm_id", dropna=False)
    out["wind_trend_kt"] = grouped["wmo_wind_kt"].diff()
    out["pressure_trend_mb"] = grouped["wmo_pressure_mb"].diff()
    out["distance_trend_km"] = grouped["distance_to_land_km"].diff()
    out["hours_since_previous"] = grouped["timestamp_utc"].diff().dt.total_seconds().div(3600)

    out["landfall_proximity_score"] = _inverse_minmax(out["landfall_next3h_km"])
    out["land_proximity_score"] = _inverse_minmax(out["distance_to_land_km"])
    out["intensifying_flag"] = ((out["wind_trend_kt"] > 0) | (out["pressure_trend_mb"] < 0)).fillna(False)
    return out


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = df[SIMILARITY_FEATURES].copy()
    medians = features.median(numeric_only=True).fillna(0.0)
    filled = features.fillna(medians)
    means = filled.mean()
    stds = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
    normalized = (filled - means) / stds
    return normalized, medians, stds


def quality_for_row(row: pd.Series) -> dict[str, object]:
    missing = [feature for feature in SIMILARITY_FEATURES if pd.isna(row.get(feature))]
    usable = len(SIMILARITY_FEATURES) - len(missing)
    return {
        "usable": usable,
        "total": len(SIMILARITY_FEATURES),
        "missing": missing,
        "percent": round(100 * usable / len(SIMILARITY_FEATURES), 1),
    }


def _inverse_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_val = values.min(skipna=True)
    max_val = values.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(np.nan, index=series.index)
    return 1 - ((values - min_val) / (max_val - min_val))
