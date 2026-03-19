"""
CO2 Storage Site Suitability - Synthetic Data Generator
========================================================
Generates geologically-informed synthetic datasets for training
and evaluating the CO2 storage suitability ML model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ── Geological parameter ranges (literature-informed) ────────────────────────
PARAM_RANGES = {
    # Porosity (fraction): good reservoirs ~0.15–0.35
    "porosity":          {"min": 0.01, "max": 0.45},
    # Permeability (mD): good reservoirs > ~10 mD
    "permeability_md":   {"min": 0.01, "max": 2000.0},
    # Depth (m): sweet-spot 800–3000 m (supercritical CO2)
    "depth_m":           {"min": 300,  "max": 5000},
    # Reservoir pressure (MPa): supercritical threshold ~7.38 MPa
    "pressure_mpa":      {"min": 1.0,  "max": 60.0},
    # Temperature (°C): supercritical threshold ~31.1 °C
    "temperature_c":     {"min": 20,   "max": 200},
    # Cap-rock thickness (m): thicker is safer
    "caprock_thickness_m": {"min": 5,  "max": 500},
    # Cap-rock integrity score (0–1): expert/seismic proxy
    "caprock_integrity":   {"min": 0.0, "max": 1.0},
    # Total dissolved solids (g/L): high TDS → formation water is saline (good)
    "salinity_g_l":      {"min": 0.5,  "max": 350.0},
    # Structural trap area (km²)
    "trap_area_km2":     {"min": 1.0,  "max": 500.0},
    # Distance to nearest fault (km): farther is safer
    "fault_distance_km": {"min": 0.1,  "max": 50.0},
}


def _supercritical(depth, pressure, temperature):
    """
    CO2 is supercritical (dense phase) above 31.1 °C and 7.38 MPa.
    Depth > 800 m is a practical proxy used in many screening studies.
    Returns a float 0–1 representing how 'supercritical-friendly' the site is.
    """
    depth_ok   = np.clip((depth - 500) / 1000, 0, 1)
    press_ok   = np.clip((pressure - 5) / 20, 0, 1)
    temp_ok    = np.clip((temperature - 25) / 40, 0, 1)
    return (depth_ok + press_ok + temp_ok) / 3


def _compute_suitability_and_risk(df: pd.DataFrame):
    """
    Physics-informed scoring:
      - suitability_score  0–1 (continuous, used to derive binary label)
      - risk_score         0–1 (leakage / instability risk)
    """
    p = df  # alias

    # --- Storage capacity proxy ---
    storage = (
        np.clip((p["porosity"] - 0.05) / 0.30, 0, 1) * 0.4
        + np.clip(np.log10(p["permeability_md"] + 1) / np.log10(2001), 0, 1) * 0.3
        + np.clip((p["trap_area_km2"] - 1) / 499, 0, 1) * 0.3
    )

    # --- Supercritical conditions ---
    sc = _supercritical(p["depth_m"], p["pressure_mpa"], p["temperature_c"])

    # --- Containment (caprock) ---
    containment = (
        np.clip((p["caprock_thickness_m"] - 10) / 490, 0, 1) * 0.5
        + p["caprock_integrity"] * 0.5
    )

    # --- Salinity bonus (brine displacement favoured in high-TDS aquifers) ---
    salinity_bonus = np.clip(p["salinity_g_l"] / 350, 0, 1) * 0.1

    # --- Composite suitability ---
    suitability = (
        storage      * 0.35
        + sc         * 0.30
        + containment* 0.30
        + salinity_bonus * 0.05
    )

    # --- Risk (leakage + instability) ---
    fault_risk    = np.clip(1 - (p["fault_distance_km"] / 50), 0, 1)
    caprock_risk  = 1 - containment
    sc_risk       = np.clip(1 - sc, 0, 1) * 0.5   # non-supercritical → higher risk
    depth_risk    = np.where(p["depth_m"] > 3500, 0.3, 0.0)  # very deep → drilling risk

    risk = (
        fault_risk  * 0.35
        + caprock_risk * 0.35
        + sc_risk   * 0.20
        + depth_risk * 0.10
    )

    return np.clip(suitability, 0, 1), np.clip(risk, 0, 1)


def generate_dataset(n_samples: int = 5000, noise_level: float = 0.04) -> pd.DataFrame:
    """
    Generate a synthetic geological dataset with suitability labels
    and risk scores.

    Parameters
    ----------
    n_samples    : number of reservoir samples to generate
    noise_level  : Gaussian noise std added to final scores (realism)

    Returns
    -------
    pd.DataFrame with features + 'suitable' (0/1) + 'risk_score' (0–1)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    data = {}
    for feat, bounds in PARAM_RANGES.items():
        lo, hi = bounds["min"], bounds["max"]
        if feat == "permeability_md":
            # Log-normal: most rocks are tight, few are highly permeable
            log_vals = rng.uniform(np.log(lo + 0.01), np.log(hi), n_samples)
            data[feat] = np.clip(np.exp(log_vals), lo, hi)
        elif feat in ("caprock_integrity",):
            # Beta-distributed: bimodal (poor vs good cap rocks common)
            data[feat] = np.clip(rng.beta(2, 2, n_samples), 0, 1)
        else:
            data[feat] = rng.uniform(lo, hi, n_samples)

    df = pd.DataFrame(data)

    # Derived features (geothermal gradient proxy)
    df["geothermal_gradient"] = df["temperature_c"] / (df["depth_m"] / 1000 + 1e-6)
    df["pressure_depth_ratio"] = df["pressure_mpa"] / (df["depth_m"] / 1000 + 1e-6)

    # Compute scores
    suit_score, risk_score = _compute_suitability_and_risk(df)

    # Add noise
    suit_score = np.clip(suit_score + rng.normal(0, noise_level, n_samples), 0, 1)
    risk_score = np.clip(risk_score + rng.normal(0, noise_level, n_samples), 0, 1)

    # Binary label — threshold calibrated for ~35 % suitable (realistic scarcity)
    threshold = np.percentile(suit_score, 65)
    df["suitable"]         = (suit_score >= threshold).astype(int)
    df["suitability_score"] = suit_score.round(4)
    df["risk_score"]        = risk_score.round(4)

    return df


def split_and_save(df: pd.DataFrame, output_dir: str = "."):
    """Split into train/val/test and save as CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=["suitable", "suitability_score", "risk_score"])
    y_cls = df["suitable"]
    y_risk = df["risk_score"]

    # 70 / 15 / 15 split
    X_train, X_tmp, y_cls_train, y_cls_tmp, y_risk_train, y_risk_tmp = train_test_split(
        X, y_cls, y_risk, test_size=0.30, random_state=RANDOM_SEED, stratify=y_cls
    )
    X_val, X_test, y_cls_val, y_cls_test, y_risk_val, y_risk_test = train_test_split(
        X_tmp, y_cls_tmp, y_risk_tmp, test_size=0.50, random_state=RANDOM_SEED,
        stratify=y_cls_tmp
    )

    train = X_train.copy(); train["suitable"] = y_cls_train; train["risk_score"] = y_risk_train
    val   = X_val.copy();   val["suitable"]   = y_cls_val;   val["risk_score"]   = y_risk_val
    test  = X_test.copy();  test["suitable"]  = y_cls_test;  test["risk_score"]  = y_risk_test

    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv",     index=False)
    test.to_csv(f"{output_dir}/test.csv",   index=False)

    print(f"Dataset sizes  →  train: {len(train):,}  |  val: {len(val):,}  |  test: {len(test):,}")
    print(f"Class balance  →  suitable: {y_cls_train.mean():.1%}  |  unsuitable: {1-y_cls_train.mean():.1%}")
    return train, val, test


if __name__ == "__main__":
    df = generate_dataset(n_samples=5000)
    train, val, test = split_and_save(df, output_dir="data")
    print("\nSample rows:")
    print(df.head(3).T)
