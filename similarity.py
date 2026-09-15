from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import FEATURE_LABELS, FEATURE_WEIGHTS, SIMILARITY_FEATURES


def top_historical_twins(
    df: pd.DataFrame,
    normalized_features: pd.DataFrame,
    observation_id: int,
    top_k: int = 5,
) -> pd.DataFrame:
    current = df[df["observation_id"] == observation_id]
    if current.empty:
        return pd.DataFrame()

    current_row = current.iloc[0]
    candidate_df = df[df["observation_id"] != observation_id].copy()
    different_storm = candidate_df["storm_id"] != current_row["storm_id"]
    if different_storm.sum() >= top_k:
        candidate_df = candidate_df[different_storm].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    weights = np.array([FEATURE_WEIGHTS[feature] for feature in SIMILARITY_FEATURES], dtype=float)
    current_vector = normalized_features.loc[observation_id, SIMILARITY_FEATURES].to_numpy(dtype=float)
    candidate_vectors = normalized_features.loc[candidate_df["observation_id"], SIMILARITY_FEATURES].to_numpy(dtype=float)

    deltas = candidate_vectors - current_vector
    distances = np.sqrt((np.square(deltas) * weights).sum(axis=1) / weights.sum())
    similarities = 100 / (1 + distances)

    candidate_df["similarity_distance"] = distances
    candidate_df["similarity_score"] = similarities
    candidate_df["same_storm_as_selected"] = candidate_df["storm_id"] == current_row["storm_id"]
    candidate_df = candidate_df.sort_values(["similarity_score", "timestamp_utc"], ascending=[False, True])
    return candidate_df.head(top_k).reset_index(drop=True)


def feature_contributions(
    normalized_features: pd.DataFrame,
    observation_id: int,
    twin_observation_id: int,
) -> pd.DataFrame:
    current = normalized_features.loc[observation_id, SIMILARITY_FEATURES]
    twin = normalized_features.loc[twin_observation_id, SIMILARITY_FEATURES]
    records = []
    for feature in SIMILARITY_FEATURES:
        delta = abs(float(twin[feature] - current[feature]))
        records.append(
            {
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Weighted distance": round(delta * FEATURE_WEIGHTS[feature], 3),
                "Weight": FEATURE_WEIGHTS[feature],
            }
        )
    return pd.DataFrame(records).sort_values("Weighted distance", ascending=False)


def weights_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Column": feature,
                "Weight": weight,
            }
            for feature, weight in FEATURE_WEIGHTS.items()
        ]
    )
