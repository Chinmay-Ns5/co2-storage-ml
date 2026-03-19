# 🛢️ AI-Based CO₂ Storage Site Suitability Analysis

> A machine learning pipeline that predicts whether a subsurface geological reservoir is suitable for carbon capture and storage (CCS) — and estimates its leakage/instability risk — using XGBoost trained on physics-informed synthetic data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-f89939?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Carbon Capture and Storage (CCS) is a critical tool in reducing atmospheric CO₂. The biggest bottleneck is **site screening** — identifying which underground rock formations can safely store CO₂ without leaking. Traditional methods involve seismic surveys, lab analysis, and reservoir simulations that take months and cost millions per site.

This project provides an **AI-assisted first-pass screening tool** that:
- Takes 10 basic geological parameters as input
- Classifies a site as **suitable or unsuitable** for CO₂ storage
- Estimates a **leakage/instability risk score** (0–1)
- Flags specific geological red flags with plain-language warnings

It is not a replacement for full reservoir simulation — it is a rapid filter that can narrow 500 candidate sites down to the 50 worth investigating deeply.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| ROC-AUC (test set) | **0.9387** |
| ROC-AUC (5-fold CV) | **0.9501 ± 0.006** |
| Accuracy | **85%** |
| F1 Score (suitable class) | **0.79** |
| Risk R² | **0.893** |
| Risk RMSE | **0.042** |

---

## 🗂️ Project Structure

```
co2-storage-ml/
│
├── data_generator.py       # Synthetic dataset generation (physics-informed)
├── train.py                # Model training, evaluation, dashboard
├── predict.py              # Inference engine — single site or batch
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Auto-created by data_generator.py
│   ├── train.csv           # 3,500 training samples
│   ├── val.csv             # 750 validation samples
│   └── test.csv            # 750 held-out test samples
│
└── outputs/                # Auto-created by train.py
    ├── classifier_pipeline.joblib
    ├── regressor_pipeline.joblib
    ├── metrics.json
    └── evaluation_dashboard.png
```

---

## ⚡ Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Chinmay-Ns5/co2-storage-ml.git
cd co2-storage-ml
pip install -r requirements.txt
```

### 2. Generate the dataset

```bash
python data_generator.py
```

Creates `data/train.csv`, `data/val.csv`, `data/test.csv` — 5,000 synthetic reservoir samples split 70/15/15.

### 3. Train the models

```bash
python train.py
```

Trains both models, runs cross-validation, evaluates on the test set, saves everything to `outputs/`.

### 4. Run inference on demo sites

```bash
python predict.py
```

Runs predictions on 3 pre-built demo reservoirs and prints results like:

```
Site: High-quality saline aquifer
  ✅ SUITABLE
  Confidence   : 62.2%
  Risk Score   : 42.2%  (Medium)
  Verdict      : SUITABLE for CO₂ storage. Leakage risk is Medium.

Site: Shallow, low-porosity, near-fault
  ❌ UNSUITABLE
  Confidence   : 0.0%
  Risk Score   : 69.6%  (High)
  Verdict      : ⚠ Depth < 800 m: CO₂ may not reach supercritical state.
                 ⚠ Low cap-rock integrity. ⚠ Proximity to fault. ⚠ Low porosity.
```

### 5. Predict on your own site

```python
from predict import predict

result = predict({
    "porosity": 0.22,
    "permeability_md": 120.0,
    "depth_m": 1800,
    "pressure_mpa": 20.0,
    "temperature_c": 70.0,
    "caprock_thickness_m": 90,
    "caprock_integrity": 0.75,
    "salinity_g_l": 150.0,
    "trap_area_km2": 60.0,
    "fault_distance_km": 8.0,
})

print(result)
# {suitable: True, confidence: 0.71, risk_score: 0.38, risk_level: 'Medium', verdict: '...'}
```

---

## 🤖 Model Architecture

Two separate XGBoost models are trained on the same 12 features:

```
Raw Input (10 geological parameters)
        │
        ▼
Feature Engineering → 12 features
        │
   ┌────┴────┐
   ▼         ▼
XGBClassifier    XGBRegressor
   │              │
P(suitable)    Risk score
   │              │
   └────┬─────────┘
        ▼
  Verdict + Warnings
```

**Why two models instead of one?**
Suitability (binary classification) and risk (continuous regression) have different output geometries and loss functions. Separating them allows independent tuning and replacement of either model without affecting the other.

**Why XGBoost?**
- Consistently outperforms neural networks on structured/tabular geological data
- Native feature importance — critical for geological interpretability
- Scale-invariant tree splits handle heterogeneous parameter ranges (porosity 0–1 vs permeability 0–2000 mD)
- Robust to outliers, no GPU needed, serializes to ~1 MB
- Well-calibrated probabilities via `predict_proba`

---

## 🧬 Geological Parameters

| Parameter | Unit | Good Range | Why It Matters |
|---|---|---|---|
| Porosity | fraction | 0.15 – 0.35 | More pore space = more storage volume |
| Permeability | mD | > 10 mD | Controls how easily CO₂ can be injected |
| Depth | m | 800 – 3000 m | Below 800 m CO₂ becomes supercritical (dense, safe) |
| Pressure | MPa | > 7.38 MPa | Must exceed CO₂ critical pressure |
| Temperature | °C | > 31.1 °C | Must exceed CO₂ critical temperature |
| Cap-rock thickness | m | > 50 m | Thicker seal = lower leakage probability |
| Cap-rock integrity | 0 – 1 | > 0.6 | Absence of fractures and permeable pathways in the seal |
| Salinity | g/L | > 50 g/L | CO₂ dissolves better in saline brine; protects freshwater aquifers |
| Trap area | km² | > 20 km² | Larger structural closure = more total storage capacity |
| Fault distance | km | > 5 km | Proximity to faults increases seismicity and leakage risk |

---

## 🧪 Synthetic Data — Why and How

### Why synthetic data?

Real CCS screening datasets are not publicly available — they are owned by oil and gas companies or national geological surveys, cost millions to license, and are inconsistently formatted across countries and eras. Even large global CCS programs have evaluated only dozens of sites, far too few to train a robust ML model.

Synthetic data is the **scientifically appropriate choice** for concept validation:
- Demonstrates the ML architecture is sound before real data is available
- Enables controlled experiments (vary one parameter, verify model responds correctly)
- Avoids proprietary data issues — fully open and reproducible
- Designed so `data_generator.py` can be swapped for a real data loader without touching `train.py` or `predict.py`

### How it was built

Parameters are sampled using distributions that reflect real geological behaviour:

| Parameter | Distribution | Reason |
|---|---|---|
| Permeability | Log-normal | Most rocks are tight; high-permeability rocks are rare — standard assumption in petroleum engineering |
| Cap-rock integrity | Beta (α=2, β=2) | Cap rocks tend to be either good or poor, not uniformly average |
| All others | Uniform within bounds | Full-range screening with no prior bias |

**Suitability score formula:**
```
suitability = (storage_capacity × 0.35)
            + (supercritical_score  × 0.30)
            + (containment_score    × 0.30)
            + (salinity_bonus       × 0.05)
```

**Risk score formula:**
```
risk = (fault_risk    × 0.35)
     + (caprock_risk  × 0.35)
     + (sc_risk       × 0.20)
     + (depth_risk    × 0.10)
```

Calibrated Gaussian noise (σ = 0.04) is added to both scores to simulate measurement uncertainty and prevent the model from overfitting to exact rule boundaries. The binary label is created by thresholding at the 65th percentile → **35% suitable / 65% unsuitable**, reflecting realistic scarcity of good CCS sites.

---

## 📈 Evaluation Dashboard

Running `train.py` produces `outputs/evaluation_dashboard.png` — a 7-panel visual report:

| Panel | What It Shows |
|---|---|
| ROC Curve | Classifier discrimination across all thresholds (AUC = 0.94) |
| Confusion Matrix | Normalized correct/incorrect classification breakdown |
| Probability Distribution | Separation between model confidence for suitable vs unsuitable sites |
| Feature Importance (Classifier) | Which geological parameters drive suitability prediction |
| Feature Importance (Regressor) | Which parameters drive risk prediction |
| Actual vs Predicted Risk | Regression scatter plot (R² = 0.89) |
| Metrics Summary | All key numbers in one place |

---

## 🔧 Requirements

```
xgboost>=1.7.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🚧 Limitations

- Trained on synthetic data — real-world performance requires validation against actual geological databases
- Static prediction — does not model CO₂ plume migration or pressure evolution over time
- 10-parameter simplification — real screening involves rock mineralogy, geomechanics, tectonic history, and regulatory constraints
- Treats each site independently — no spatial or regional geological context

---

## 🔮 Planned Extensions

- [ ] Interactive Streamlit UI with live parameter input and risk gauges
- [ ] Real data integration (IEAGHG, NETL CO2 Storage Resources Database)
- [ ] Uncertainty quantification with conformal prediction intervals
- [ ] Seismic attribute feature integration
- [ ] Reservoir simulation tool coupling

---

## 👤 Author

**Chinmay N S**
B.E. Artificial Intelligence & Machine Learning — PES University, Bengaluru (Class of 2028)
[github.com/Chinmay-Ns5](https://github.com/Chinmay-Ns5)

---

*Concept validation prototype. Not intended for regulatory decision-making. All training data is synthetic.*
