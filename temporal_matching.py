from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import FEATURE_WEIGHTS, SIMILARITY_FEATURES


def observation_sequence(df: pd.DataFrame, observation_id: int, steps: int = 3) -> pd.DataFrame:
    row = df[df["observation_id"] == observation_id]
    if row.empty:
        return pd.DataFrame()
    selected = row.iloc[0]
    storm = df[df["storm_id"] == selected["storm_id"]].sort_values("timestamp_utc")
    history = storm[storm["timestamp_utc"] <= selected["timestamp_utc"]]
    return history.tail(steps)


def temporal_similarity(
    df: pd.DataFrame,
    normalized_features: pd.DataFrame,
    observation_id: int,
    candidate_observation_id: int,
    steps: int = 3,
) -> dict[str, object]:
    current_seq = observation_sequence(df, observation_id, steps)
    candidate_seq = observation_sequence(df, candidate_observation_id, steps)
    count = min(len(current_seq), len(candidate_seq))
    if count < 2:
        return {"available": False, "score": None, "steps": count}

    current_ids = current_seq.tail(count)["observation_id"].tolist()
    candidate_ids = candidate_seq.tail(count)["observation_id"].tolist()
    current_vectors = normalized_features.loc[current_ids, SIMILARITY_FEATURES].to_numpy(dtype=float)
    candidate_vectors = normalized_features.loc[candidate_ids, SIMILARITY_FEATURES].to_numpy(dtype=float)
    weights = np.array([FEATURE_WEIGHTS[feature] for feature in SIMILARITY_FEATURES], dtype=float)
    distances = np.sqrt((np.square(candidate_vectors - current_vectors) * weights).sum(axis=1) / weights.sum())
    mean_distance = float(np.mean(distances))
    return {"available": True, "score": 100 / (1 + mean_distance), "steps": count}


def future_evolution(df: pd.DataFrame, observation_id: int, max_hours: int = 48) -> pd.DataFrame:
    row = df[df["observation_id"] == observation_id]
    if row.empty:
        return pd.DataFrame()
    selected = row.iloc[0]
    storm = df[df["storm_id"] == selected["storm_id"]].sort_values("timestamp_utc")
    future = storm[storm["timestamp_utc"] >= selected["timestamp_utc"]].copy()
    if pd.notna(selected["timestamp_utc"]):
        future["hours_from_match"] = (future["timestamp_utc"] - selected["timestamp_utc"]).dt.total_seconds().div(3600)
        future = future[(future["hours_from_match"] >= 0) & (future["hours_from_match"] <= max_hours)]
    else:
        future["hours_from_match"] = np.nan
    return future


def current_history(df: pd.DataFrame, observation_id: int, max_hours: int = 48) -> pd.DataFrame:
    row = df[df["observation_id"] == observation_id]
    if row.empty:
        return pd.DataFrame()
    selected = row.iloc[0]
    storm = df[df["storm_id"] == selected["storm_id"]].sort_values("timestamp_utc")
    history = storm[storm["timestamp_utc"] <= selected["timestamp_utc"]].copy()
    if pd.notna(selected["timestamp_utc"]):
        history["hours_from_now"] = (history["timestamp_utc"] - selected["timestamp_utc"]).dt.total_seconds().div(3600)
        history = history[history["hours_from_now"] >= -max_hours]
    else:
        history["hours_from_now"] = np.nan
    return history
