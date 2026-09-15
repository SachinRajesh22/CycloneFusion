from __future__ import annotations

import pandas as pd

from temporal_matching import future_evolution


def historical_risk_signal(df: pd.DataFrame, current_row: pd.Series, twins: pd.DataFrame) -> dict[str, object]:
    evidence = []
    score = 0

    wind_trend = current_row.get("wind_trend_kt")
    pressure_trend = current_row.get("pressure_trend_mb")
    distance_trend = current_row.get("distance_trend_km")
    distance_to_land = current_row.get("distance_to_land_km")

    if pd.notna(wind_trend) and wind_trend > 0:
        score += 2
        evidence.append("Current wind is increasing compared with the previous observation.")
    if pd.notna(pressure_trend) and pressure_trend < 0:
        score += 2
        evidence.append("Current pressure is falling compared with the previous observation.")
    if pd.notna(distance_trend) and distance_trend < 0:
        score += 1
        evidence.append("Current cyclone position is moving closer to land in the track data.")
    if pd.notna(distance_to_land) and distance_to_land <= 200:
        score += 2
        evidence.append("Current observation is within 200 km of land.")
    elif pd.notna(distance_to_land) and distance_to_land <= 500:
        score += 1
        evidence.append("Current observation is within 500 km of land.")

    twin_outcomes = []
    high_similarity_twins = 0
    for _, twin in twins.iterrows():
        future = future_evolution(df, int(twin["observation_id"]), max_hours=48)
        if future.empty:
            continue
        start_wind = future.iloc[0].get("wmo_wind_kt")
        max_wind = future["wmo_wind_kt"].max(skipna=True)
        start_pressure = future.iloc[0].get("wmo_pressure_mb")
        min_pressure = future["wmo_pressure_mb"].min(skipna=True)
        intensified = pd.notna(start_wind) and pd.notna(max_wind) and max_wind - start_wind >= 10
        deepened = pd.notna(start_pressure) and pd.notna(min_pressure) and start_pressure - min_pressure >= 5
        if intensified or deepened:
            twin_outcomes.append(twin.get("storm_label", "Historical twin"))
        if twin.get("similarity_score", 0) >= 60:
            high_similarity_twins += 1

    if twin_outcomes:
        score += min(3, len(twin_outcomes))
        evidence.append(f"{len(twin_outcomes)} selected historical twin(s) intensified or deepened after the matched stage.")
    if high_similarity_twins:
        score += 1
        evidence.append(f"{high_similarity_twins} historical twin(s) have similarity scores at or above 60.")

    if not evidence:
        evidence.append("Available dataset evidence does not show strong intensification, land proximity, or high-similarity analogue signals.")

    if score >= 7:
        label = "HIGH"
    elif score >= 3:
        label = "MODERATE"
    else:
        label = "LOW"

    return {"label": label, "score": score, "evidence": evidence}


def preparedness_level(risk_label: str) -> tuple[str, list[str]]:
    if risk_label == "HIGH":
        return (
            "Escalate",
            [
                "Review shelter, medical, and power-backup readiness using official local protocols.",
                "Prepare public communication material that points people to IMD and government advisories.",
                "Increase monitoring frequency for landfall proximity, wind trend, and pressure trend.",
            ],
        )
    if risk_label == "MODERATE":
        return (
            "Prepare",
            [
                "Check coastal resource availability and update district-level contact chains.",
                "Monitor historical-twin evolution and current pressure/wind changes.",
                "Keep messaging generic unless official agencies issue actionable warnings.",
            ],
        )
    return (
        "Monitor",
        [
            "Continue observation-level monitoring and historical twin comparison.",
            "Maintain routine readiness checks for coastal response teams.",
            "Avoid issuing action instructions without official IMD/government confirmation.",
        ],
    )
