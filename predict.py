"""
CO2 Storage Site Suitability — Inference Module
================================================
Load the trained models and predict on new reservoir parameter inputs.
Can be used standalone or imported by a UI/API layer.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

OUTPUT_DIR = "outputs"

FEATURES = [
    "porosity", "permeability_md", "depth_m", "pressure_mpa",
    "temperature_c", "caprock_thickness_m", "caprock_integrity",
    "salinity_g_l", "trap_area_km2", "fault_distance_km",
    "geothermal_gradient", "pressure_depth_ratio",
]

FEATURE_LABELS = {
    "porosity":             "Porosity (fraction, e.g. 0.20)",
    "permeability_md":      "Permeability (mD)",
    "depth_m":              "Reservoir Depth (m)",
    "pressure_mpa":         "Reservoir Pressure (MPa)",
    "temperature_c":        "Temperature (°C)",
    "caprock_thickness_m":  "Cap-rock Thickness (m)",
    "caprock_integrity":    "Cap-rock Integrity (0–1)",
    "salinity_g_l":         "Brine Salinity (g/L)",
    "trap_area_km2":        "Structural Trap Area (km²)",
    "fault_distance_km":    "Distance to Nearest Fault (km)",
}


def _load_models():
    clf = joblib.load(f"{OUTPUT_DIR}/classifier_pipeline.joblib")
    reg = joblib.load(f"{OUTPUT_DIR}/regressor_pipeline.joblib")
    return clf, reg


def _derive_features(params: dict) -> dict:
    """Compute engineered features from raw inputs."""
    depth = params["depth_m"]
    temp  = params["temperature_c"]
    pres  = params["pressure_mpa"]
    params["geothermal_gradient"]  = temp / (depth / 1000 + 1e-6)
    params["pressure_depth_ratio"] = pres / (depth / 1000 + 1e-6)
    return params


def predict(params: dict) -> dict:
    """
    Predict suitability and risk for a single reservoir.

    Parameters
    ----------
    params : dict  — raw geological inputs (see FEATURE_LABELS for keys)

    Returns
    -------
    dict with:
      suitable        : bool
      confidence      : float  (0–1, probability of being suitable)
      risk_score      : float  (0–1)
      risk_level      : str    (Low / Medium / High / Critical)
      verdict         : str    (human-readable summary)
    """
    clf, reg = _load_models()

    params = _derive_features(dict(params))
    X = pd.DataFrame([{f: params[f] for f in FEATURES}])

    confidence  = float(clf.predict_proba(X)[0, 1])
    suitable    = confidence >= 0.50
    risk_score  = float(np.clip(reg.predict(X)[0], 0, 1))

    if risk_score < 0.25:
        risk_level = "Low"
    elif risk_score < 0.50:
        risk_level = "Medium"
    elif risk_score < 0.75:
        risk_level = "High"
    else:
        risk_level = "Critical"

    verdict_parts = []
    if suitable:
        verdict_parts.append(f"SUITABLE for CO₂ storage (confidence: {confidence:.1%}).")
    else:
        verdict_parts.append(f"UNSUITABLE for CO₂ storage (confidence of suitability: {confidence:.1%}).")
    verdict_parts.append(f"Leakage / instability risk is {risk_level} ({risk_score:.1%}).")

    # Key driver hints
    if params["depth_m"] < 800:
        verdict_parts.append("⚠ Depth < 800 m: CO₂ may not reach supercritical state.")
    if params["caprock_integrity"] < 0.40:
        verdict_parts.append("⚠ Low cap-rock integrity: containment risk elevated.")
    if params["fault_distance_km"] < 2:
        verdict_parts.append("⚠ Proximity to fault: seismic/leakage risk elevated.")
    if params["porosity"] < 0.10:
        verdict_parts.append("⚠ Low porosity: limited storage capacity.")

    return {
        "suitable":    suitable,
        "confidence":  round(confidence, 4),
        "risk_score":  round(risk_score, 4),
        "risk_level":  risk_level,
        "verdict":     " ".join(verdict_parts),
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a DataFrame of reservoir samples.
    Must contain the raw feature columns (see FEATURE_LABELS).
    """
    results = []
    for _, row in df.iterrows():
        r = predict(row.to_dict())
        results.append(r)
    out = df.copy()
    out["predicted_suitable"] = [r["suitable"]   for r in results]
    out["confidence"]         = [r["confidence"] for r in results]
    out["predicted_risk"]     = [r["risk_score"] for r in results]
    out["risk_level"]         = [r["risk_level"] for r in results]
    return out


# ── Interactive CLI demo ─────────────────────────────────────────────────────
DEMO_SITES = {
    "High-quality saline aquifer (should be SUITABLE)": {
        "porosity": 0.28, "permeability_md": 250.0, "depth_m": 1500,
        "pressure_mpa": 18.0, "temperature_c": 65.0,
        "caprock_thickness_m": 120, "caprock_integrity": 0.88,
        "salinity_g_l": 180.0, "trap_area_km2": 85.0, "fault_distance_km": 12.0,
    },
    "Shallow, low-porosity, near-fault (should be UNSUITABLE)": {
        "porosity": 0.06, "permeability_md": 2.5, "depth_m": 450,
        "pressure_mpa": 5.5, "temperature_c": 28.0,
        "caprock_thickness_m": 15, "caprock_integrity": 0.22,
        "salinity_g_l": 12.0, "trap_area_km2": 4.0, "fault_distance_km": 0.4,
    },
    "Deep high-pressure reservoir (borderline)": {
        "porosity": 0.18, "permeability_md": 45.0, "depth_m": 3200,
        "pressure_mpa": 38.0, "temperature_c": 120.0,
        "caprock_thickness_m": 60, "caprock_integrity": 0.65,
        "salinity_g_l": 90.0, "trap_area_km2": 30.0, "fault_distance_km": 5.0,
    },
}


def run_demo():
    print("=" * 60)
    print("  CO₂ Suitability Inference — Demo Predictions")
    print("=" * 60)

    for site_name, params in DEMO_SITES.items():
        print(f"\n{'─'*60}")
        print(f"  Site: {site_name}")
        print("  Inputs:")
        for k, v in params.items():
            print(f"    {k:<25} {v}")
        result = predict(params)
        label  = "✅ SUITABLE" if result["suitable"] else "❌ UNSUITABLE"
        print(f"\n  {label}")
        print(f"  Confidence   : {result['confidence']:.1%}")
        print(f"  Risk Score   : {result['risk_score']:.1%}  ({result['risk_level']})")
        print(f"  Verdict      : {result['verdict']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run_demo()
