from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import FEATURE_LABELS, SIMILARITY_FEATURES, quality_for_row


RADAR_AXES = [
    ("Wind intensity", "wmo_wind_kt", False),
    ("Pressure state", "wmo_pressure_mb", True),
    ("Movement speed", "storm_speed_kt", False),
    ("Land proximity", "distance_to_land_km", True),
    ("Landfall proximity", "landfall_next3h_km", True),
    ("Wind trend", "wind_trend_kt", False),
    ("Pressure fall", "pressure_trend_mb", True),
]


def selected_row(df: pd.DataFrame, observation_id: int) -> pd.Series:
    matches = df[df["observation_id"] == observation_id]
    if matches.empty:
        return df.iloc[-1]
    return matches.iloc[0]


def normalized_vector(normalized_features: pd.DataFrame, observation_id: int) -> pd.Series:
    if observation_id in normalized_features.index:
        return normalized_features.loc[observation_id]
    return normalized_features.iloc[-1]


def fingerprint_table(row: pd.Series, normalized: pd.Series) -> pd.DataFrame:
    records = []
    for feature in SIMILARITY_FEATURES:
        records.append(
            {
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Raw value": _format_value(row.get(feature)),
                "Normalized value": round(float(normalized.get(feature, 0.0)), 3),
                "Weight input": feature,
                "Missing": bool(pd.isna(row.get(feature))),
            }
        )
    return pd.DataFrame(records)


def radar_values(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    records = []
    for label, feature, inverse in RADAR_AXES:
        values = pd.to_numeric(df[feature], errors="coerce")
        value = pd.to_numeric(pd.Series([row.get(feature)]), errors="coerce").iloc[0]
        score = _minmax_score(values, value, inverse=inverse)
        records.append({"Axis": label, "Score": score})
    return pd.DataFrame(records)


def vector_text(normalized: pd.Series) -> str:
    parts = [f"{feature}={float(normalized.get(feature, 0.0)):.3f}" for feature in SIMILARITY_FEATURES]
    return "[" + ", ".join(parts) + "]"


def data_quality(row: pd.Series) -> dict[str, object]:
    return quality_for_row(row)


def _minmax_score(values: pd.Series, value: float, inverse: bool = False) -> float:
    if pd.isna(value):
        return 0.0
    min_val = values.min(skipna=True)
    max_val = values.max(skipna=True)
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return 50.0
    score = 100 * (value - min_val) / (max_val - min_val)
    if inverse:
        score = 100 - score
    return float(np.clip(score, 0, 100))


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
