# UHI Bengaluru - MLPR End-Term Project

**Predicting Urban Heat Island severity in Bengaluru using ensemble machine learning on satellite, urban morphology, and socioeconomic data.**

Course: MLPR (Spring 2026), Plaksha University
Team: Arnav Nathani, Sahil Aleem, Vayun Gupta

---

## What this project does

Custom 2,433-point dataset built over a 10x10 km central Bengaluru ROI, combining Landsat 8/9 thermal imagery, ERA5 climate reanalysis, Sentinel-5P NO2, MODIS LST, OSMnx building/road morphology, WorldPop population density, and Meta Relative Wealth Index. After cleaning, 20 features are used to (a) regress Land Surface Temperature (LST) and (b) classify UHI severity into 3 classes (Cool Island / Neutral / Hot UHI) using the Liu & Zhang (2011) Mean-SD method.

Headline model: unweighted ensemble of XGBoost + LightGBM + CatBoost, tuned with 60-trial Optuna runs, evaluated with Stratified 5-fold CV.

---

## Results

| Task | Metric | Value |
|---|---|---|
| Regression | R2 | **0.7940** |
| | RMSE | 1.233 C |
| | MAE | 0.887 C |
| 3-class classification | Accuracy | **0.7604** |
| | F1-weighted | 0.7599 |
| | ROC-AUC | 0.8946 |
| Novelty ablation | delta-R2 (11 novel features removed) | **+0.1335** |
| SHAP | Novel-feature total attribution | **43.4%** (RWI ranks #3 at 12.3%) |
| Cross-city (BLR to Pune, naive) | R2 | -0.2443 |
| Cross-city (per-city z-score) | R2 | +0.1662 |
| Combined BLR+Pune 5-fold CV | R2 | **0.7497** |

All numbers come from a single script (`verify_experiments.py`) that runs in about 4 minutes on a laptop. Full log: `results/verification_results.txt`. Machine-readable: `results/verification_summary.json`.

---

## Repo structure

```
.
+-- *.py                         # runnable scripts (see below)
+-- *.csv                        # Bengaluru and Pune datasets
+-- README.md
+-- PROJECT_REFERENCE.md         # detailed methods and numbers reference
+-- figures/                     # maps, SHAP plots, residual analysis
+-- results/                     # JSON + TXT output from experiments
+-- notebooks/
|   +-- UHI_Bengaluru_Analysis.ipynb      # main analysis notebook
|   +-- data_collection_pipeline.ipynb    # GEE + OSMnx + RWI data build
+-- references/                  # key papers
+-- DOCX Files/                  # literature survey papers
```

---

## Dataset

- **Bengaluru:** 2,433 points, 27 columns, 10x10 km ROI in central Bengaluru, 50 m grid
- **Pune:** 2,382 points, 36 columns, same schema with LULC one-hot extras, used for cross-city transfer only
- **Sources:** Landsat 8/9 (GEE), ERA5, Sentinel-5P TROPOMI, MODIS LST, OSMnx, WorldPop, Meta RWI
- **Target:** LST in degrees Celsius. UHI severity classes use Liu & Zhang (2011) Mean-SD thresholds (mu +/- 0.5*sigma = 32.42 C / 35.14 C)

### Feature cleaning (27 to 20)

| Dropped | Reason |
|---|---|
| LST | Regression target |
| Latitude, Longitude | Prevent model memorising geography (mentor guidance) |
| EVI, SAVI, MNDWI | Redundant with NDVI/NDBI (Pearson r > 0.85) |
| Volume_Density | 84.8% missing |

### Final 20 features

| Category | Features |
|---|---|
| Remote sensing | NDVI, NDBI |
| Meteorological | Air_Temp_C, Relative_Humidity, Wind_Speed, Soil_Moisture, Surface_Pressure_hPa, MODIS_LST |
| Anthropogenic | NO2_Emissions, Pop_Density |
| OSM morphology | Building_Density_Ratio, Height_Variability, SVF_Proxy |
| OSM proximity | Dist_Water, Dist_Park, Dist_Highway, Street_Density |
| Socioeconomic | Relative_Wealth_Index, Dist_to_RWI_Node |
| Radiative | Albedo |

---

## Classification scheme

Liu & Zhang (2011) Mean-SD, 3 classes:

| Class | LST range | Count |
|---|---|---|
| Cool Island | < 32.42 C | 526 |
| Neutral | 32.42-35.14 C | 1,173 |
| Hot UHI | > 35.14 C | 734 |

4-class variant was tested but the extra "Moderate UHI" band (35.14-36.50 C) was only 1.36 C wide, narrower than our RMSE of 1.23 C, causing structural misclassification at boundaries. Merged to 3 classes to match the literature standard.

---

## Methodology

- **Models tested:** Linear, KNN, SVM, Random Forest, XGBoost, LightGBM, CatBoost, MLP (64-32 ReLU). Headline = XGB + LGB + CatBoost unweighted average.
- **Tuning:** Optuna TPE, 60 trials per tree model. Hyperparameters locked in `verify_experiments.py`.
- **Validation:**
  - Stratified 5-fold CV on LST tertile bins (primary, all reported numbers)
  - Spatial Block CV with KMeans geographic partitioning (robustness check, R2 = 0.72-0.74)
- **Preprocessing:** KNN imputation (k=5) as safety net; StandardScaler for linear/distance/MLP models only; tree ensembles use raw features. Per-city z-score applied for cross-city pipeline.

---

## Key scripts

| Script | Purpose |
|---|---|
| `verify_experiments.py` | Reproduces every reported metric end to end |
| `crosscity_20feat.py` | Cross-city transfer experiments (BLR and Pune, 20 features) |
| `within_pune_cv.py` | Within-Pune 5-fold CV (18-feature common set) |
| `generate_city_maps.py` | Generates BLR and Pune UHI/NDVI maps |
| `build_pptx.py` | Generates the 39-slide presentation |

---

## Reproducing results

```bash
pip install scikit-learn xgboost lightgbm catboost shap scipy pandas numpy matplotlib seaborn imbalanced-learn optuna geopandas contextily

# Run all experiments (Optuna skipped - hyperparameters already locked)
python3 verify_experiments.py
```

Output goes to `results/` and `figures/`. Total runtime around 4 minutes on a laptop.

Cells 0-6 in `notebooks/data_collection_pipeline.ipynb` pull data from Google Earth Engine and OSMnx; the merged CSVs are already in the repo so those cells do not need to be re-run.

---

## Novelty claims

1. First ML-based UHI study for Bengaluru (zero prior coverage in our 11-paper survey).
2. Only paper combining Meta RWI + OSMnx 3D morphology + remote sensing + meteorological features jointly.
3. Only Indian-context UHI paper with empirical cross-city domain adaptation - identified the feature scale-mismatch failure mode; fix via per-city z-score moves R2 from -0.24 to +0.17, combined model reaches 0.75.
4. Group-wise ablation shows novel features account for +0.1335 R2 (17% of total explained variance). TreeSHAP confirms RWI as #3 predictor (12.3%), NO2 as #4 (9.6%).
5. Stratified 5-fold CV + Spatial Block CV - comparable validation rigour to only Mansouri (2025) among the 11 surveyed papers.

---

## Literature comparison

| Paper | Region | Samples | Model | Best metric | Protocol |
|---|---|---|---|---|---|
| Hoang & Nguyen 2025 | Da Nang, Vietnam | 5,000 | NGBoost + DNN | R2 = 0.90 | Single 70/30 split |
| Lynda et al. 2025 | Africa, 10K cities | continental | BNN + GBR | R2 = 0.84 | 10-fold |
| Manna et al. 2026 | Mumbai, India | MMR grid | Scenario analysis | R2 = 0.80 | In-sample |
| Kusumadewi et al. 2025 | Malang, Indonesia | scene | SVR | R2 = 0.78 | Held-out |
| Mansouri & Erfani 2025 | USA Midwest | 10,795 | RF + XGBoost | Acc = 0.76, AUC = 0.91 | 3-class CV |
| **Ours** | **Bengaluru, India** | **2,433** | **XGB + LGB + CatBoost** | **R2 = 0.7940, AUC = 0.8946** | **Stratified 5-fold** |

Normalised RMSE (RMSE / sigma_LST = 0.453) is the fairest cross-paper comparison given differing protocols.

---

## Limitations

- Single dry-season snapshot; model captures summer-peak UHI only.
- Small dataset relative to global studies (5K-10K+ points).
- Directional cross-city transfer positive but modest (+0.17 R2); single-city training limits generalisation.
- Volume_Density too sparse to use (84.8% missing).

---

## References

1. A. Mansouri & A. Erfani (2025). Machine Learning Prediction of Urban Heat Island Severity in the Midwestern United States. *Sustainability*, 17(13), 6193.
2. N. D. Hoang & Q. L. Nguyen (2025). Geospatial Analysis and Machine Learning Framework for Urban Heat Island Intensity Prediction: Natural Gradient Boosting and Deep Neural Network Regressors with Multisource Remote Sensing Data. *Sustainability*, 17(10), 4287.
3. D. Lynda, G. Logeswari, K. Tamilarasi & S. Rakesh (2025). Urban Heat Islands Predictive Model Using Bayesian Neural Networks. *Scientific Reports*, 15, 31280.
4. H. Deng et al. (2025). Integrating Model Variability, Scale Effect, and Zoning Effect to Analyze Uncertainties in Surface Urban Heat Island Prediction. *Sustainable Cities and Society*, 130, 106588.
5. M. Aqdas, T. M. Usmani, R. Benhizia & G. Szabo (2025). Urban Expansion and Thermal Stress: A Remote Sensing Analysis of LULC and Urban Heat Islands in Ghaziabad, India. *Land*, 14(9), 1893.
6. T. Kusumadewi et al. (2025). Synthesizing Environmental, Social, and Urban Density Metrics to Predict Urban Heat Island Dynamics using Remote Sensing and Support Vector Regression. *Engineering, Technology and Applied Science Research (ETASR)*, 9791.
7. H. Manna, M. Pramanik, R. Ahamed & S. Sarkar (2026). Urban Heating in Mumbai: From Scenario Analysis to the Mitigation of Nighttime Thermal Footprint. *Sustainable Cities and Society*.
8. M. B. Suthar (2025). Integrating Remote Sensing and Machine Learning for Predictive Analysis of Urban Heat Island Dynamics. *Environmental Reports*, 3(1), 10.
9. Tanoori et al. (2024). Urban Heat Island Analysis using XGBoost and Deep Neural Networks. *Shiraz, Iran case study*.
10. Liu & Zhang (2011). Mean-SD classification method for urban heat island severity assessment.
11. Meta Data for Good (2022). Relative Wealth Index for low- and middle-income countries.
