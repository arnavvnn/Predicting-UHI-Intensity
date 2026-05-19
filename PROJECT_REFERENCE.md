# UHI Bengaluru — MLPR End-Term Project: Full Reference

*All numbers in this document are verified by `verify_experiments.py` (2026-05-18 run). Full log: `verification_results.txt`. JSON dump: `verification_summary.json`.*

---

## Project Overview

**Title:** Predicting Urban Heat Island (UHI) Severity in Bengaluru Using Ensemble Machine Learning

**Course:** MLPR End-Term Project (Plaksha University, Spring 2026)

**Core problem:** Bengaluru's urban areas run 4–5°C hotter than surrounding rural areas. We build ML models to predict and classify UHI severity at pixel level using satellite imagery, urban morphology features, and socioeconomic data.

**Why it matters:** Heat-related mortality, energy demand spikes, urban planning decisions — the model can help BBMP / KSPCB identify hotspots and prioritize mitigation efforts.

---

## Dataset

| Property | Bengaluru | Pune (cross-city test) |
|---|---|---|
| Samples | 2,433 | 2,382 |
| Columns (raw) | 27 | 36 |
| Features (cleaned, final) | 20 | 20 (cross-city common = 18) |
| LST mean | 33.78°C | z-scored (mean = 0) |
| LST std | 2.72°C | std = 1 |
| Missing data issue | Volume_Density: 84.8% missing | Zero nulls |
| Source | GEE (Landsat thermal) + OSMnx + Meta RWI | Same pipeline |

**Data sources:**
- **Landsat 8/9** via Google Earth Engine → Land Surface Temperature (LST), NDVI, NDBI, Albedo, Soil Moisture
- **ERA5 / MODIS** → Air_Temp_C, Relative_Humidity, Wind_Speed, Surface_Pressure, MODIS_LST
- **Sentinel-5P** → NO2_Emissions
- **OSMnx** → Building_Density_Ratio, Height_Variability, SVF_Proxy, Street_Density, Dist_Water, Dist_Park, Dist_Highway
- **Meta Relative Wealth Index (RWI)** → Relative_Wealth_Index, Dist_to_RWI_Node
- **WorldPop** → Pop_Density

### Feature Cleaning Decisions

Features **dropped** from the raw 27:
- `EVI`, `SAVI`, `MNDWI` — redundant with NDVI and NDBI (high multicollinearity)
- `Volume_Density` — 84.8% missing; KNN imputation on this scale would be synthetic noise
- `Latitude`, `Longitude` — removed on mentor advice (risk of the model learning geographic coordinates rather than urban physics)

### Final 20 Features Used

| Category | Features |
|---|---|
| Remote sensing indices | NDVI, NDBI |
| Meteorological | Air_Temp_C, Relative_Humidity, Wind_Speed, Soil_Moisture, Surface_Pressure_hPa, MODIS_LST |
| Anthropogenic | NO2_Emissions, Pop_Density |
| OSM 3D morphology | Building_Density_Ratio, Height_Variability, SVF_Proxy |
| OSM proximity | Dist_Water, Dist_Park, Dist_Highway, Street_Density |
| Socioeconomic (novel) | Relative_Wealth_Index, Dist_to_RWI_Node |
| Radiative | Albedo |

---

## Classification Scheme

**Method:** Liu & Zhang (2011) Mean–SD thresholding (also used by Mansouri & Erfani 2025)

**Boundaries:** μ ± 0.5σ → 32.42°C (lower) and 35.14°C (upper)

| Class | LST Range | Count |
|---|---|---|
| Cool Island | < 32.42°C | 526 |
| Neutral | 32.42–35.14°C | 1,173 |
| Hot UHI | > 35.14°C | 734 |

### Why We Went From 4 Classes to 3

We originally tried 4 classes by adding a split at μ+σ = 36.50°C, which created a "Moderate UHI" band from 35.14°C to 36.50°C — only 1.36°C wide. Our ensemble RMSE is 1.23°C, so that band is narrower than our own prediction error, meaning the model would structurally misclassify pixels near the boundary. We merged to 3 classes, which matches the Mansouri & Erfani (2025) scheme and allows a fair head-to-head comparison.

---

## Models

| Model | Task | Notes |
|---|---|---|
| Linear Regression | Regression | Baseline only |
| KNN (k=7, distance-weighted) | Regression | Distance-based baseline |
| Random Forest | Regression + Classification | Tree-based baseline |
| XGBoost | Regression + Classification | Headline model, Optuna-tuned |
| LightGBM | Regression + Classification | Optuna-tuned |
| CatBoost | Regression + Classification | Optuna-tuned |
| **3-model Ensemble** | **Regression + Classification** | **Unweighted average of XGB + LGBM + CatBoost — final headline model** |
| MLP (2 hidden layers: 64→32 ReLU) | Regression + Classification | Neural net comparison — consistently underperforms |

### Hyperparameter Tuning

- **Framework:** Optuna (TPE sampler)
- **Trials:** 60 per model
- **CV objective during tuning:** Stratified 5-fold
- **Tuned models:** XGBoost, LightGBM, CatBoost
- Pre-tuned hyperparameters are locked in `verify_experiments.py` for reproducibility

### Validation Protocol

1. **Stratified 5-fold CV** — primary validation; all headline metrics come from this
2. **Spatial Block CV** — KMeans checkerboard partitioning into 10 geographic blocks; tests geographic generalization, not just random-split performance

### Preprocessing

- **Missing values:** KNN Imputation (k=5) applied to remaining nulls after dropping Volume_Density
- **Scaling:** StandardScaler applied for Linear / KNN / MLP only; tree-based models receive raw features
- **Cross-city pipeline:** per-city z-score standardization (each feature standardized within its own city before combining)

---

## Results — Bengaluru Regression (verified, Stratified 5-fold)

| Model / Configuration | MSE (°C²) | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|---|
| Linear Regression baseline | 4.251 | 2.062 | 1.489 | 0.4241 |
| KNN (k=7, distance) | 2.963 | 1.721 | 1.213 | 0.5985 |
| Random Forest (n=500, d=12) | 2.253 | 1.501 | 1.054 | 0.6943 |
| Paper baseline (Mansouri 6-feat, RF) | 3.398 | 1.843 | 1.314 | 0.5394 |
| XGBoost Optuna-tuned (cleaned 20-feat) | 1.572 | 1.254 | 0.904 | 0.7869 |
| LightGBM Optuna-tuned (cleaned 20-feat) | 1.583 | 1.258 | 0.911 | 0.7855 |
| CatBoost Optuna-tuned (cleaned 20-feat) | 1.568 | 1.252 | 0.900 | 0.7875 |
| **★ 3-model Ensemble (cleaned 20-feat)** | **1.520** | **1.233** | **0.887** | **0.7940** |
| MLP (64→32 ReLU) | 3.612 | 1.901 | 1.345 | 0.5066 |

**Key comparison:** Paper baseline (Mansouri 6-feature schema replicated on our Bengaluru data) → R² = 0.5394. Our ensemble → R² = 0.7940. **+0.255 R² lift.**

---

## Results — Bengaluru Classification (3-class, verified, Stratified 5-fold)

| Model / Configuration | Accuracy | F1-weighted | ROC-AUC |
|---|---|---|---|
| Paper baseline (Mansouri 6-feat, XGB) | 0.6704 | 0.6690 | 0.8420* |
| **★ 3-model Ensemble (cleaned 20-feat)** | **0.7604** | **0.7599** | **0.8946** |
| MLP (64→32 ReLU) | 0.7008 | 0.6998 | 0.8375 |
| **Mansouri & Erfani 2025 (USA, 10,795 rows)** | **0.7600** | **0.7600** | **0.9100** |

*Paper baseline AUC reported as approximate (3-class one-vs-rest)*

We match Mansouri on Accuracy and F1-weighted with **4.4× less training data** and single-city geographic spread. AUC gap is only −0.015.

### Confusion Matrix (3-class ensemble, cumulative 5-fold OOF)

| | Predicted: Cool | Predicted: Neutral | Predicted: Hot |
|---|---|---|---|
| True: Cool | **394** | 125 | 7 |
| True: Neutral | 79 | **953** | 141 |
| True: Hot | 1 | 230 | **503** |

**Per-class F1:** Cool = 0.788 | Neutral = 0.768 | Hot = 0.726

---

## Cross-City Generalization (Bengaluru ↔ Pune, verified)

| Direction | Standardization | Regression R² |
|---|---|---|
| Train Bengaluru → Test Pune | None | **−1.362** |
| Train Pune → Test Bengaluru | None | −1.010 |
| Train Bengaluru → Test Pune | Per-city z-score | +0.068 |
| Train Pune → Test Bengaluru | Per-city z-score | +0.070 |
| **Combined 5-fold CV (both cities)** | Per-city z-score | **0.7005** |

**Why naive transfer collapses:** LST absolute scale differs (Bengaluru mean = 33.78°C vs Pune z-scored = 0.0) plus feature distribution shift. The model was learning city-level offsets, not UHI physics.

**Fix:** Per-city z-score standardization of every feature within its own city before merging. The directional transfer recovers from catastrophic (−1.36) to slightly positive (+0.07). The combined model trained on both cities reaches **R² = 0.70 under 5-fold CV**, with `City` feature importance reduced to 0.092 (small relative to NDBI/RWI dominance — the model learns UHI physics but does retain a city-aware residual).

> Note: earlier handoff documents claimed City feature importance = 0.000 — that was from a different model config. The honest verified number is 0.092, still small but non-zero.

---

## Novelty Quantification — Group-Wise Ablation (verified)

*Baseline ensemble R² = 0.7940. Each feature group is dropped, model re-trained end-to-end, R² drop measured.*

| Removed group | R² after removal | ΔR² contribution |
|---|---|---|
| **All 11 Indian-context novel features** | 0.6605 | **+0.1335** |
| OSM proximity (Dist_Water, Dist_Park, Dist_Highway) | 0.7715 | +0.0224 |
| Socioeconomic RWI (RWI + Dist_to_RWI_Node) | 0.7736 | +0.0204 |
| Anthropogenic NO₂ emissions | 0.7749 | +0.0191 |
| OSM 3D morphology | 0.7945 | −0.0005 |
| Albedo | 0.7955 | −0.0015 |
| Street_Density | 0.7988 | −0.0048 |

**Headline:** Without the 11 novel features, R² drops from 0.794 to 0.660. **17% of total explained variance comes from features no prior comparable UHI ML paper jointly uses.**

The three biggest single-group contributors are **OSM proximity** (+0.022), **socioeconomic RWI** (+0.020), and **NO₂ emissions** (+0.019). OSM 3D morphology, Albedo, and Street_Density individually show near-zero ΔR² — but they contribute jointly with the others (the "ALL" row removes more than the sum of parts).

---

## TreeSHAP Feature Importance (Tuned XGBoost, full data, verified)

| Rank | Feature | Mean |SHAP| | % of Total | Type |
|---|---|---|---|---|
| 1 | NDBI | 0.7241 | 20.53% | Standard |
| 2 | Relative_Humidity | 0.4490 | 12.73% | Standard |
| **3** | **Relative_Wealth_Index** | **0.4331** | **12.28%** | **NOVEL** |
| **4** | **NO2_Emissions** | **0.3400** | **9.64%** | **NOVEL** |
| 5 | MODIS_LST | 0.2049 | 5.81% | Standard |
| 6 | Soil_Moisture | 0.1902 | 5.39% | Standard |
| **7** | **Dist_Water** | **0.1875** | **5.32%** | **NOVEL** |
| **8** | **Dist_Park** | **0.1440** | **4.08%** | **NOVEL** |
| 9 | NDVI | 0.1326 | 3.76% | Standard |
| 10 | Pop_Density | 0.1150 | 3.26% | Standard |

**Novel features total SHAP attribution: 43.40%**

RWI is the #3 most important feature in the entire model. NO₂ is #4. 4 of the top 8 features are features no prior UHI ML paper jointly uses in an Indian-city context.

---

## Residual / Error Analysis (verified, ensemble OOF predictions)

| Metric | Value |
|---|---|
| Mean bias | +0.0096°C (essentially unbiased) |
| Std of residuals | 1.2333°C |
| MAE | 0.8873°C |
| Median absolute error | 0.6579°C |
| Skewness | −0.6060 (slightly left-skewed — a handful of very low predictions) |
| Excess kurtosis | +4.6234 (leptokurtic — heavier tails than Gaussian) |

**Normality tests:** Shapiro-Wilk, Jarque-Bera, and K-S all reject normality (p < 10⁻⁹). But importantly, residuals are *tighter than Gaussian* for everything below the 95th percentile:

| Percentile | Empirical |error| | Gaussian equivalent |
|---|---|---|
| 25th | 0.322°C | 0.393°C — **tighter** |
| 50th | 0.658°C | 0.832°C — **tighter** |
| 75th | 1.202°C | 1.419°C — **tighter** |
| 90th | 1.855°C | 2.029°C — **tighter** |
| 95th | 2.392°C | 2.417°C — **tighter** |
| 99th | 4.364°C | 3.177°C — **heavier (outliers)** |

**Practical takeaway:** Use MAE (0.89°C) as the typical-case accuracy figure. Use 99th-percentile error (4.4°C) as the worst-case bound. The model is well-calibrated for most pixels but has a small number of extreme outliers (likely pixels with unusual urban morphology or land-use transitions).

---

## MLP (Neural Network) Comparison (verified)

**Architecture:** Input(20) → Dense(64, ReLU) → Dense(32, ReLU) → Output
**Training:** Adam optimizer, lr=1e-3, batch size=32, early stopping on validation loss

| Task | MLP | Best Tree-Based | Gap |
|---|---|---|---|
| Regression R² | 0.5066 ± 0.091 | 0.7940 | −0.287 |
| 3-class Accuracy | 0.7008 | 0.7604 | −0.060 |
| 3-class AUC | 0.8375 | 0.8946 | −0.057 |

**Why MLP underperforms:** 2,433 samples is well below the empirical crossover point (~10K+ rows) where neural nets start beating gradient-boosted trees on tabular data. High variance (R² std = ±0.091 across folds) confirms the MLP is too data-hungry for this dataset size. Consistent with the broader literature: Hoang's DNN < NGBoost on similar data; Tanoori's XGB > DNN; Lynda's continental study uses GBR, not deep learning.

---

## Literature Benchmarks

### Regression — Cross-Paper Comparison

| Paper | Region | Samples | CV Protocol | R² | RMSE/σ (normalized) |
|---|---|---|---|---|---|
| Hoang & Nguyen 2025 | Da Nang, Vietnam | 5,000 | Single 70/30 split | 0.90 | ~0.57 |
| Lynda 2025 | Africa (10K cities) | continental | 10-fold | 0.84 | 1.26 |
| Manna 2026 | Mumbai (nighttime) | MMR grid | In-sample | 0.80 | — |
| Kusumadewi 2025 | Malang, Indonesia | scene | Held-out | 0.81 | 0.45 |
| **Ours (Bengaluru)** | **Bengaluru** | **2,433** | **Stratified 5-fold (honest)** | **0.7940** | **0.453** |

**Note on comparisons:** Raw R² comparisons are misleading because different papers use different train/test protocols. Single held-out splits inflate R². Normalized RMSE (RMSE/σ_LST) is the only fair cross-paper metric, since it accounts for how much variance there is in LST to predict. Our RMSE/σ = 0.453 is competitive with or better than all benchmarks on this measure.

### Classification — Cross-Paper Comparison

| Paper | Region | Classes | Samples | Accuracy | F1-w | AUC |
|---|---|---|---|---|---|---|
| Mansouri & Erfani 2025 | USA Midwest, 12 states | 3 | 10,795 | 0.760 | 0.760 | 0.910 |
| **Ours (Bengaluru)** | **Bengaluru** | **3** | **2,433** | **0.7604** | **0.7599** | **0.8946** |

---

## Key Novelty Claims

1. **First ML UHI study for Bengaluru.** No prior paper in our 11-paper literature survey covers Bengaluru. Closest Indian comparables are Mumbai (coastal, nighttime) and Ghaziabad (NCR, correlation-only studies — no predictive ML).

2. **Only paper jointly combining:** Meta Relative Wealth Index + OSMnx 3D morphology (Building_Density, Height_Variability, SVF_Proxy, Street_Density) + remote sensing + meteorological features in a UHI context.

3. **Only Indian-context UHI paper demonstrating cross-city domain adaptation** — with an empirically quantified fix for negative transfer (−1.36 → +0.07 directional, 0.70 combined) using per-city z-score standardization.

4. **Quantified novelty via ablation:** Novel features contribute +0.134 R² (17% of explained variance). Confirmed by SHAP: RWI is the #3 most important feature at 12.3% attribution.

5. **Honest validation** — Stratified 5-fold CV + Spatial Block CV. Only Mansouri (2025) among the 11 surveyed papers uses comparable validation rigor.

---

## Limitations

- **Single dry-season snapshot.** One Landsat pass (peak UHI, summer). Industry norm for UHI studies (monsoon cloud cover makes Landsat thermal band unusable), but means the model captures summer-peak UHI only, not monsoon or winter dynamics.
- **Single city training data.** 2,433 samples. Cross-city transfer R² is +0.07 directional (positive but modest), 0.70 combined — not ready for zero-shot city generalization.
- **No probabilistic uncertainty.** Point predictions only. Adding conformal prediction or quantile regression would allow uncertainty-aware deployment (e.g., flag predictions in the high-error tail).

---

## Reproducibility

Every reported number in this document is produced by running:
```
python3 verify_experiments.py
```
which takes ~4 minutes on a laptop and produces `verification_results.txt` (log) + `verification_summary.json` (machine-readable).

The Optuna tuning step (60 trials × 3 models × 5-fold = ~30 min) is **not** re-run by this script; the resulting hyperparameters are locked in at the top of `verify_experiments.py`. Re-tuning would require running notebook cell 11 plus extensions for LightGBM/CatBoost.

---

## Headline Numbers (Quick Reference)

| What | Number |
|---|---|
| Regression R² (ensemble, 5-fold) | **0.7940** |
| Regression RMSE | **1.233°C** |
| Regression MAE | **0.887°C** |
| Regression R² lift over paper baseline | **+0.255** |
| Classification Accuracy (3-class) | **0.7604** |
| Classification F1-weighted | **0.7599** |
| Classification AUC | **0.8946** |
| Mansouri 2025 comparison (Acc / F1 / AUC) | 0.760 / 0.760 / 0.910 |
| Novel features ΔR² contribution (ablation) | **+0.134** |
| Novel features total SHAP attribution | **43.4%** |
| RWI SHAP rank | **#3 (12.3%)** |
| Cross-city transfer R² (naive) | −1.362 |
| Cross-city transfer R² (per-city z-scored) | +0.068 |
| Combined BLR+Pune model R² (5-fold) | 0.7005 |
| City feature importance (combined model) | 0.092 |
| MLP regression R² | 0.5066 |
