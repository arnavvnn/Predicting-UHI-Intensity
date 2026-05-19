"""
verify_experiments.py
Re-runs every experiment in the UHI Bengaluru notebook EXCEPT Optuna tuning
(hyperparameters are locked-in from a previous 60-trial TPE run).

Outputs:
    - verification_results.txt : human-readable log of every metric
    - residual_analysis.png    : 3-panel residual diagnostics (re-generated)
    - shap_bar.png             : SHAP top-20 bar (re-generated)
    - shap_beeswarm.png        : SHAP beeswarm (re-generated)

Run as:  python3 verify_experiments.py
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, KFold,
                                     train_test_split, cross_val_score)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, precision_score, recall_score)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
import shap
from scipy import stats

OUTPUT_LOG = "results/verification_results.txt"
log_lines = []

def log(msg=""):
    print(msg)
    log_lines.append(msg)

def flush():
    with open(OUTPUT_LOG, "w") as f:
        f.write("\n".join(log_lines))

t0 = time.time()
log("=" * 78)
log("UHI BENGALURU — FULL EXPERIMENT VERIFICATION (Optuna skipped)")
log(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 78)

# ============================================================
# LOAD DATA
# ============================================================
log("\n--- LOADING DATA ---")
df_blr = pd.read_csv('ultimate_uhi_dataset_with_rwi.csv')
df_pune = pd.read_csv('Pune_MLPR_Standardized_Dataset.csv')
log(f"Bengaluru: {df_blr.shape}  |  Pune: {df_pune.shape}")
log(f"Bengaluru LST: mean={df_blr['LST'].mean():.3f} C  std={df_blr['LST'].std():.3f} C")

# Locked-in tuned hyperparameters (from 60-trial Optuna run — we are NOT re-tuning)
XGB_BEST = dict(n_estimators=929, max_depth=7, learning_rate=0.01944, subsample=0.8695,
                colsample_bytree=0.6321, min_child_weight=6, reg_alpha=1.4e-05, reg_lambda=2e-07)
LGB_BEST = dict(n_estimators=1244, num_leaves=51, max_depth=12, learning_rate=0.03179,
                min_child_samples=11, subsample=0.8698, colsample_bytree=0.6200,
                reg_alpha=3.3e-06, reg_lambda=3e-08)
CB_BEST  = dict(iterations=1304, depth=7, learning_rate=0.07352, l2_leaf_reg=4.285, subsample=0.8765)

# Classifier variants (same arch, multiclass objective)
XGB_CLF = dict(n_estimators=900, max_depth=7, learning_rate=0.02, subsample=0.87,
               colsample_bytree=0.65, min_child_weight=6, reg_alpha=1e-5, reg_lambda=1e-7,
               eval_metric='mlogloss', objective='multi:softprob', num_class=3)
LGB_CLF = dict(n_estimators=1200, num_leaves=51, max_depth=12, learning_rate=0.03,
               min_child_samples=11, subsample=0.87, colsample_bytree=0.62,
               reg_alpha=1e-6, reg_lambda=1e-8, verbose=-1,
               objective='multiclass', num_class=3)
CB_CLF  = dict(iterations=1300, depth=7, learning_rate=0.07, l2_leaf_reg=4.3, subsample=0.88,
               bootstrap_type='Bernoulli', loss_function='MultiClass')

# Define stratification on LST tertiles for regression CV (matches notebook)
y_lst = df_blr['LST'].values
mu, sd = y_lst.mean(), y_lst.std()
class_labels_3 = np.where(y_lst < mu - 0.5*sd, 0,
                  np.where(y_lst < mu + 0.5*sd, 1, 2))
b1, b2 = mu - 0.5*sd, mu + 0.5*sd
log(f"\n3-class boundaries: {b1:.3f} C  /  {b2:.3f} C")
log(f"Class counts: Cool={np.sum(class_labels_3==0)}, "
    f"Neutral={np.sum(class_labels_3==1)}, Hot={np.sum(class_labels_3==2)}")

# Cleaned feature set: drop redundant + high-missing + lat/lon
DROP = ['LST','Latitude','Longitude','EVI','SAVI','MNDWI','Volume_Density']
X_full = df_blr.drop(columns=DROP)
X_clean = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_full), columns=X_full.columns)
y = df_blr['LST'].values
log(f"\nCleaned feature set: {X_clean.shape[1]} features")
log(f"Features: {list(X_clean.columns)}")
flush()

# ============================================================
# EXPERIMENT 1 — Linear / RF / KNN baselines (Stratified 5-fold)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 1 — Baseline models on cleaned 20-feature set (Stratified 5-fold)")
log("=" * 78)

def cv_regression(model_factory, X_, y_, strat, name):
    """Run stratified 5-fold CV; return mean R², RMSE, MAE."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r2s, rmses, maes = [], [], []
    for tr, te in skf.split(X_, strat):
        Xt, yt = X_[tr], y_[tr]
        Xv, yv = X_[te], y_[te]
        m = model_factory()
        m.fit(Xt, yt)
        p = m.predict(Xv)
        r2s.append(r2_score(yv, p))
        rmses.append(np.sqrt(mean_squared_error(yv, p)))
        maes.append(mean_absolute_error(yv, p))
    log(f"  {name:30s}  R2={np.mean(r2s):.4f}  RMSE={np.mean(rmses):.4f}  MAE={np.mean(maes):.4f}")
    return np.mean(r2s), np.mean(rmses), np.mean(maes)

X_arr = X_clean.values
results = {}

# Standardize for distance/linear/NN models only
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_arr)

results['Linear'] = cv_regression(LinearRegression, X_scaled, y, class_labels_3, "Linear Regression")
results['KNN']    = cv_regression(lambda: KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
                                  X_scaled, y, class_labels_3, "KNN (k=7, distance)")
results['RF']     = cv_regression(lambda: RandomForestRegressor(n_estimators=500, max_depth=12,
                                                                random_state=42, n_jobs=-1),
                                  X_arr, y, class_labels_3, "Random Forest (n=500,d=12)")
flush()

# ============================================================
# EXPERIMENT 2 — Headline tuned ensemble (Cell 22 verification)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 2 — HEADLINE: Tuned 3-Model Ensemble (XGB+LGBM+CatBoost) — Cell 22")
log("=" * 78)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = {'XGB': [], 'LGB': [], 'CB': [], 'Ensemble': []}
fold_residuals = []
fold_predictions = []
fold_actuals = []

for fi, (tr, te) in enumerate(skf.split(X_arr, class_labels_3), 1):
    Xt, yt = X_clean.iloc[tr], y[tr]
    Xv, yv = X_clean.iloc[te], y[te]

    m1 = xgb.XGBRegressor(**XGB_BEST, random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
    m2 = lgb.LGBMRegressor(**LGB_BEST, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt)
    m3 = CatBoostRegressor(**CB_BEST, random_seed=42, verbose=0,
                           allow_writing_files=False, bootstrap_type='Bernoulli').fit(Xt, yt)

    p1, p2, p3 = m1.predict(Xv), m2.predict(Xv), m3.predict(Xv)
    pe = (p1 + p2 + p3) / 3

    def metrics(p):
        return (r2_score(yv, p),
                np.sqrt(mean_squared_error(yv, p)),
                mean_absolute_error(yv, p))

    fold_results['XGB'].append(metrics(p1))
    fold_results['LGB'].append(metrics(p2))
    fold_results['CB'].append(metrics(p3))
    fold_results['Ensemble'].append(metrics(pe))

    fold_residuals.extend((yv - pe).tolist())
    fold_predictions.extend(pe.tolist())
    fold_actuals.extend(yv.tolist())
    log(f"  Fold {fi}: ENS R2={metrics(pe)[0]:.4f}")

for name in ['XGB', 'LGB', 'CB', 'Ensemble']:
    arr = np.array(fold_results[name])
    log(f"  {name:10s}  R2={arr[:,0].mean():.4f}+/-{arr[:,0].std():.4f}  "
        f"RMSE={arr[:,1].mean():.4f}  MAE={arr[:,2].mean():.4f}  "
        f"MSE={(arr[:,1].mean())**2:.4f}")
    results[f'ENS_{name}'] = (arr[:,0].mean(), arr[:,1].mean(), arr[:,2].mean())

residuals = np.array(fold_residuals)
predictions = np.array(fold_predictions)
actuals = np.array(fold_actuals)
flush()

# ============================================================
# EXPERIMENT 3 — 3-class classification ensemble (Cell 10)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 3 — 3-CLASS CLASSIFICATION ENSEMBLE (Cell 10)")
log("=" * 78)

y3 = class_labels_3
skf_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds, all_probs, all_true = [], [], []

for fi, (tr, te) in enumerate(skf_c.split(X_clean, y3), 1):
    Xt, yt = X_clean.iloc[tr], y3[tr]
    Xv, yv = X_clean.iloc[te], y3[te]

    m1 = xgb.XGBClassifier(**XGB_CLF, random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
    m2 = lgb.LGBMClassifier(**LGB_CLF, random_state=42, n_jobs=-1).fit(Xt, yt)
    m3 = CatBoostClassifier(**CB_CLF, random_seed=42, verbose=0,
                            allow_writing_files=False).fit(Xt, yt)

    p1 = m1.predict_proba(Xv)
    p2 = m2.predict_proba(Xv)
    p3 = m3.predict_proba(Xv)
    pe = (p1 + p2 + p3) / 3
    pred = pe.argmax(axis=1)

    acc = accuracy_score(yv, pred)
    f1 = f1_score(yv, pred, average='weighted')
    log(f"  Fold {fi}: Acc={acc:.4f}  F1w={f1:.4f}")
    all_preds.extend(pred.tolist())
    all_probs.extend(pe.tolist())
    all_true.extend(yv.tolist())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true = np.array(all_true)

acc_total = accuracy_score(all_true, all_preds)
f1_total = f1_score(all_true, all_preds, average='weighted')
auc_total = roc_auc_score(all_true, all_probs, multi_class='ovr', average='weighted')
cm = confusion_matrix(all_true, all_preds)
f1_per_class = f1_score(all_true, all_preds, average=None)

log(f"\n  CUMULATIVE: Acc={acc_total:.4f}  F1w={f1_total:.4f}  AUC={auc_total:.4f}")
log(f"  Per-class F1: Cool={f1_per_class[0]:.4f}  Neutral={f1_per_class[1]:.4f}  Hot={f1_per_class[2]:.4f}")
log(f"  Confusion matrix (rows=true, cols=pred):")
log(f"           Cool  Neutral  Hot")
for i, lbl in enumerate(['Cool','Neutral','Hot']):
    log(f"    {lbl:8s} {cm[i,0]:5d}  {cm[i,1]:7d}  {cm[i,2]:4d}")

results['CLS_acc'] = acc_total
results['CLS_f1'] = f1_total
results['CLS_auc'] = auc_total
flush()

# ============================================================
# EXPERIMENT 4 — Paper baseline (6-feature Mansouri schema)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 4 — PAPER BASELINE (Mansouri 6-feature schema)")
log("=" * 78)

PAPER_FEATURES = ['Air_Temp_C','NDVI','Pop_Density','Building_Density_Ratio',
                  'Height_Variability','Relative_Wealth_Index']
Xp_raw = df_blr[PAPER_FEATURES].copy()
Xp = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(Xp_raw), columns=PAPER_FEATURES)

# Regression with RF+XGB on paper features
r2s_rf, rmses_rf, maes_rf = [], [], []
r2s_xgb, rmses_xgb, maes_xgb = [], [], []
skf_p = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf_p.split(Xp, class_labels_3):
    Xt, yt = Xp.iloc[tr], y[tr]
    Xv, yv = Xp.iloc[te], y[te]
    rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42, n_jobs=-1).fit(Xt, yt)
    xb = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
    pr = rf.predict(Xv); px = xb.predict(Xv)
    r2s_rf.append(r2_score(yv, pr)); rmses_rf.append(np.sqrt(mean_squared_error(yv, pr))); maes_rf.append(mean_absolute_error(yv, pr))
    r2s_xgb.append(r2_score(yv, px)); rmses_xgb.append(np.sqrt(mean_squared_error(yv, px))); maes_xgb.append(mean_absolute_error(yv, px))

log(f"  Paper-RF (6-feat):    R2={np.mean(r2s_rf):.4f}  RMSE={np.mean(rmses_rf):.4f}  MAE={np.mean(maes_rf):.4f}")
log(f"  Paper-XGB (6-feat):   R2={np.mean(r2s_xgb):.4f}  RMSE={np.mean(rmses_xgb):.4f}  MAE={np.mean(maes_xgb):.4f}")
results['Paper_RF'] = (np.mean(r2s_rf), np.mean(rmses_rf), np.mean(maes_rf))
results['Paper_XGB'] = (np.mean(r2s_xgb), np.mean(rmses_xgb), np.mean(maes_xgb))

# Paper baseline classification (3-class)
accs, f1s, aucs = [], [], []
for tr, te in skf_p.split(Xp, class_labels_3):
    Xt, yt = Xp.iloc[tr], class_labels_3[tr]
    Xv, yv = Xp.iloc[te], class_labels_3[te]
    m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                          objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                          random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
    proba = m.predict_proba(Xv); pred = proba.argmax(axis=1)
    accs.append(accuracy_score(yv, pred))
    f1s.append(f1_score(yv, pred, average='weighted'))
    aucs.append(roc_auc_score(yv, proba, multi_class='ovr', average='weighted'))
log(f"  Paper-XGB (3-cls):    Acc={np.mean(accs):.4f}  F1w={np.mean(f1s):.4f}  AUC={np.mean(aucs):.4f}")
results['Paper_CLS'] = (np.mean(accs), np.mean(f1s), np.mean(aucs))
flush()

# ============================================================
# EXPERIMENT 5 — MLP comparison
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 5 — MLP COMPARISON (2 hidden layers: 64,32 ReLU)")
log("=" * 78)

r2_mlp, rmse_mlp, mae_mlp = [], [], []
for tr, te in skf.split(X_arr, class_labels_3):
    Xt, yt = X_scaled[tr], y[tr]
    Xv, yv = X_scaled[te], y[te]
    m = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                     learning_rate_init=1e-3, max_iter=500, batch_size=32,
                     early_stopping=True, random_state=42).fit(Xt, yt)
    p = m.predict(Xv)
    r2_mlp.append(r2_score(yv, p)); rmse_mlp.append(np.sqrt(mean_squared_error(yv, p))); mae_mlp.append(mean_absolute_error(yv, p))
log(f"  MLP-Reg:  R2={np.mean(r2_mlp):.4f}+/-{np.std(r2_mlp):.4f}  RMSE={np.mean(rmse_mlp):.4f}  MAE={np.mean(mae_mlp):.4f}")
results['MLP_reg'] = (np.mean(r2_mlp), np.mean(rmse_mlp), np.mean(mae_mlp))

# MLP classifier
accs_m, f1s_m, aucs_m = [], [], []
for tr, te in skf_c.split(X_scaled, y3):
    Xt, yt = X_scaled[tr], y3[tr]
    Xv, yv = X_scaled[te], y3[te]
    m = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', solver='adam',
                      learning_rate_init=1e-3, max_iter=500, batch_size=32,
                      early_stopping=True, random_state=42).fit(Xt, yt)
    proba = m.predict_proba(Xv); pred = proba.argmax(axis=1)
    accs_m.append(accuracy_score(yv, pred))
    f1s_m.append(f1_score(yv, pred, average='weighted'))
    aucs_m.append(roc_auc_score(yv, proba, multi_class='ovr', average='weighted'))
log(f"  MLP-Cls:  Acc={np.mean(accs_m):.4f}  F1w={np.mean(f1s_m):.4f}  AUC={np.mean(aucs_m):.4f}")
results['MLP_cls'] = (np.mean(accs_m), np.mean(f1s_m), np.mean(aucs_m))
flush()

# ============================================================
# EXPERIMENT 6 — Cross-city transfer (BLR <-> Pune)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 6 — CROSS-CITY TRANSFER (BLR <-> Pune)")
log("=" * 78)

COMMON = ['EVI','NDBI','NDVI','Pop_Density','Relative_Humidity','Wind_Speed',
          'Building_Density_Ratio','Dist_Water','Dist_Park','Dist_Highway',
          'Street_Density','SVF_Proxy','Air_Temp_C','Albedo','MODIS_LST',
          'Soil_Moisture','Relative_Wealth_Index','Dist_to_RWI_Node']

df_b = df_blr.copy()
df_p = df_pune.copy()
df_b['LST_z'] = (df_b['LST'] - df_b['LST'].mean()) / df_b['LST'].std()
df_p['LST_z'] = df_p['LST']  # Pune already z-scored

Xb = df_b[COMMON].values; yb = df_b['LST_z'].values
Xp_ = df_p[COMMON].values; yp_ = df_p['LST_z'].values

# Naive transfer (no per-city standardization)
xb_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            random_state=42, n_jobs=-1, verbosity=0).fit(Xb, yb)
pred_p = xb_model.predict(Xp_)
r2_naive_bp = r2_score(yp_, pred_p)
log(f"  Naive BLR->Pune: R2={r2_naive_bp:.4f}")

xp_model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                            random_state=42, n_jobs=-1, verbosity=0).fit(Xp_, yp_)
pred_b = xp_model.predict(Xb)
r2_naive_pb = r2_score(yb, pred_b)
log(f"  Naive Pune->BLR: R2={r2_naive_pb:.4f}")

# Per-city z-score
scaler_b = StandardScaler().fit(Xb)
scaler_p = StandardScaler().fit(Xp_)
Xb_z = scaler_b.transform(Xb)
Xp_z = scaler_p.transform(Xp_)

xb_model2 = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=0).fit(Xb_z, yb)
pred_p2 = xb_model2.predict(Xp_z)
r2_std_bp = r2_score(yp_, pred_p2)
log(f"  Std BLR->Pune:   R2={r2_std_bp:.4f}")

xp_model2 = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             random_state=42, n_jobs=-1, verbosity=0).fit(Xp_z, yp_)
pred_b2 = xp_model2.predict(Xb_z)
r2_std_pb = r2_score(yb, pred_b2)
log(f"  Std Pune->BLR:   R2={r2_std_pb:.4f}")

# Combined model with City feature
X_combined = np.vstack([
    np.hstack([Xb_z, np.zeros((len(Xb_z), 1))]),
    np.hstack([Xp_z, np.ones((len(Xp_z), 1))])
])
y_combined = np.hstack([yb, yp_])
city_strata = np.hstack([np.zeros(len(yb)), np.ones(len(yp_))]).astype(int)

r2_combined = []
skf_cc = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in skf_cc.split(X_combined, city_strata):
    m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         random_state=42, n_jobs=-1, verbosity=0).fit(X_combined[tr], y_combined[tr])
    p = m.predict(X_combined[te])
    r2_combined.append(r2_score(y_combined[te], p))
log(f"  Combined CV R2:  {np.mean(r2_combined):.4f}+/-{np.std(r2_combined):.4f}")

# City feature importance
m_full = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                          random_state=42, n_jobs=-1, verbosity=0).fit(X_combined, y_combined)
fi = m_full.feature_importances_
log(f"  City feature importance: {fi[-1]:.6f}")

results['xc_naive_bp'] = r2_naive_bp
results['xc_naive_pb'] = r2_naive_pb
results['xc_std_bp'] = r2_std_bp
results['xc_std_pb'] = r2_std_pb
results['xc_combined'] = np.mean(r2_combined)
results['xc_city_fi'] = fi[-1]
flush()

# ============================================================
# EXPERIMENT 7 — Group-wise ablation (Cell 24)
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 7 — GROUP-WISE ABLATION ON TUNED ENSEMBLE (Cell 24)")
log("=" * 78)

GROUPS = {
    'NO2_Emissions (anthropogenic)':           ['NO2_Emissions'],
    'Albedo (radiative balance)':              ['Albedo'],
    'OSM 3D morphology':                        ['Building_Density_Ratio','Height_Variability','SVF_Proxy'],
    'OSM proximity (water/park/highway)':       ['Dist_Water','Dist_Park','Dist_Highway'],
    'Street_Density (OSM network)':             ['Street_Density'],
    'Socioeconomic RWI':                        ['Relative_Wealth_Index','Dist_to_RWI_Node'],
    'ALL Indian-context novel features':        ['NO2_Emissions','Albedo','Building_Density_Ratio',
        'Height_Variability','SVF_Proxy','Dist_Water','Dist_Park','Dist_Highway',
        'Street_Density','Relative_Wealth_Index','Dist_to_RWI_Node'],
}

def cv_ensemble(X_):
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc = []
    for tr, te in skf2.split(X_, class_labels_3):
        Xt, yt = X_.iloc[tr], y[tr]; Xv, yv = X_.iloc[te], y[te]
        m1 = xgb.XGBRegressor(**XGB_BEST, random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
        m2 = lgb.LGBMRegressor(**LGB_BEST, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt)
        m3 = CatBoostRegressor(**CB_BEST, random_seed=42, verbose=0,
                               allow_writing_files=False, bootstrap_type='Bernoulli').fit(Xt, yt)
        p = (m1.predict(Xv) + m2.predict(Xv) + m3.predict(Xv)) / 3
        sc.append(r2_score(yv, p))
    return np.mean(sc)

base_r2 = cv_ensemble(X_clean)
log(f"  Baseline ensemble (20 features): R2 = {base_r2:.4f}\n")
log(f"  {'Removed group':50s} {'R2':>8s}  {'dR2':>9s}  {'#feats':>7s}")
log("  " + "-" * 76)
ablation_results = {'baseline': base_r2}
for name, feats in GROUPS.items():
    Xa = X_clean.drop(columns=feats)
    r2_a = cv_ensemble(Xa)
    delta = base_r2 - r2_a
    log(f"  {name:50s} {r2_a:8.4f}  {delta:+9.4f}  {Xa.shape[1]:7d}")
    ablation_results[name] = (r2_a, delta)
flush()

# ============================================================
# EXPERIMENT 8 — TreeSHAP analysis on full data
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 8 — TREESHAP FEATURE IMPORTANCE (full data, tuned XGBoost)")
log("=" * 78)

xgb_full = xgb.XGBRegressor(**XGB_BEST, random_state=42, n_jobs=-1, verbosity=0).fit(X_clean, y)
explainer = shap.TreeExplainer(xgb_full)
shap_values = explainer.shap_values(X_clean)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
total_shap = mean_abs_shap.sum()
shap_df = pd.DataFrame({
    'feature': X_clean.columns,
    'mean_abs_shap': mean_abs_shap,
    'pct_total': 100 * mean_abs_shap / total_shap
}).sort_values('mean_abs_shap', ascending=False)

NOVEL = {'Relative_Wealth_Index','Dist_to_RWI_Node','NO2_Emissions',
         'Building_Density_Ratio','Height_Variability','SVF_Proxy',
         'Dist_Water','Dist_Park','Dist_Highway','Street_Density','Albedo'}
shap_df['type'] = shap_df['feature'].apply(lambda f: 'NOVEL' if f in NOVEL else 'standard')
novel_total_pct = shap_df[shap_df.type == 'NOVEL']['pct_total'].sum()

log(f"\n  Rank  Feature                       MeanAbsSHAP  PctTotal  Type")
for i, row in enumerate(shap_df.head(10).itertuples(), 1):
    log(f"  {i:4d}  {row.feature:28s}  {row.mean_abs_shap:10.4f}  {row.pct_total:6.2f}%  {row.type}")
log(f"\n  Novel features total SHAP attribution: {novel_total_pct:.2f}%")
shap_df.to_csv('results/shap_results.csv', index=False)

# Plot SHAP bar + beeswarm
plt.figure(figsize=(8, 8))
shap.summary_plot(shap_values, X_clean, plot_type='bar', max_display=20, show=False)
plt.title('TreeSHAP Feature Importance (top 20)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_bar.png', dpi=120, bbox_inches='tight')
plt.close()
log("  Saved: shap_bar.png")

plt.figure(figsize=(8, 8))
shap.summary_plot(shap_values, X_clean, max_display=20, show=False)
plt.title('TreeSHAP Beeswarm — feature impact on LST prediction', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/shap_beeswarm.png', dpi=120, bbox_inches='tight')
plt.close()
log("  Saved: shap_beeswarm.png")
flush()

# ============================================================
# EXPERIMENT 9 — Residual analysis with statistical tests
# ============================================================
log("\n\n" + "=" * 78)
log("EXP 9 — RESIDUAL & STATISTICAL ANALYSIS (ensemble OOF predictions)")
log("=" * 78)

mean_bias = residuals.mean()
std_resid = residuals.std()
mae = np.abs(residuals).mean()
median_abs = np.median(np.abs(residuals))
skew_v = stats.skew(residuals)
kurt_v = stats.kurtosis(residuals)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
jb_stat, jb_p = stats.jarque_bera(residuals)
ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))

log(f"  Mean bias:        {mean_bias:+.4f} C  (essentially unbiased)")
log(f"  Std of residuals: {std_resid:.4f} C")
log(f"  MAE:              {mae:.4f} C")
log(f"  Median abs error: {median_abs:.4f} C")
log(f"  std/MAE ratio:    {std_resid/mae:.4f}  (Gaussian = 1.253)")
log(f"  Skewness:         {skew_v:+.4f}")
log(f"  Excess kurtosis:  {kurt_v:+.4f}")
log(f"  Shapiro-Wilk:     stat={shapiro_stat:.4f}  p={shapiro_p:.2e}")
log(f"  Jarque-Bera:      stat={jb_stat:.4f}  p={jb_p:.2e}")
log(f"  K-S vs Normal:    stat={ks_stat:.4f}  p={ks_p:.2e}")

abs_resid = np.abs(residuals)
log(f"\n  Percentile analysis of |residual| (empirical vs Gaussian):")
for q in [25, 50, 75, 90, 95, 99]:
    emp = np.percentile(abs_resid, q)
    gauss = abs(stats.norm.ppf((q+100)/200, 0, std_resid))
    log(f"    {q:3d}th: empirical={emp:.3f} C  |  Gaussian-equiv={gauss:.3f} C  |  "
        f"{'TIGHTER' if emp < gauss else 'HEAVIER'}")

# Residual plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
ax.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black', color='steelblue')
xx = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(xx, stats.norm.pdf(xx, mean_bias, std_resid), 'r-', lw=2, label='Gaussian fit')
ax.set_xlabel('Residual (C)'); ax.set_ylabel('Density')
ax.set_title(f'Residual Distribution\nBias={mean_bias:+.3f}  Std={std_resid:.3f}  '
             f'Skew={skew_v:+.2f}  Kurt={kurt_v:+.2f}')
ax.legend()

ax = axes[1]
stats.probplot(residuals, dist='norm', plot=ax)
ax.set_title('Q-Q Plot vs Normal')

ax = axes[2]
ax.scatter(predictions, residuals, alpha=0.3, s=10, color='steelblue')
ax.axhline(0, color='r', linestyle='--', lw=1)
ax.set_xlabel('Predicted LST (C)'); ax.set_ylabel('Residual (C)')
ax.set_title(f'Residual vs Predicted  |  MAE={mae:.3f}  Median|e|={median_abs:.3f}')
plt.tight_layout()
plt.savefig('figures/residual_analysis.png', dpi=120, bbox_inches='tight')
plt.close()
log("  Saved: residual_analysis.png")
flush()

# ============================================================
# FINAL SUMMARY
# ============================================================
log("\n\n" + "=" * 78)
log("FINAL HEADLINE SUMMARY")
log("=" * 78)
ens = results['ENS_Ensemble']
log(f"\n[REGRESSION] Tuned 3-model Ensemble (20 features, Stratified 5-fold):")
log(f"   R2 = {ens[0]:.4f}  |  RMSE = {ens[1]:.4f} C  |  MAE = {ens[2]:.4f} C")
log(f"\n[CLASSIFICATION] 3-class Ensemble (Stratified 5-fold):")
log(f"   Accuracy = {results['CLS_acc']:.4f}  |  F1-weighted = {results['CLS_f1']:.4f}  |  AUC = {results['CLS_auc']:.4f}")
log(f"\n[ABLATION] Drop ALL novel features:  R2 falls from {base_r2:.4f} -> {ablation_results['ALL Indian-context novel features'][0]:.4f}  (dR2 = +{ablation_results['ALL Indian-context novel features'][1]:.4f})")
log(f"\n[SHAP] Novel-features total attribution: {novel_total_pct:.2f}%")
log(f"\n[CROSS-CITY] Naive transfer: {r2_naive_bp:+.4f}  -->  Standardized: {r2_std_bp:+.4f}  -->  Combined CV: {results['xc_combined']:.4f}  |  City FI: {results['xc_city_fi']:.6f}")
log(f"\n[MLP] R2 = {results['MLP_reg'][0]:.4f}  (vs ensemble {ens[0]:.4f})")
log(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")

# Persist as JSON too for programmatic use
with open('results/verification_summary.json', 'w') as f:
    payload = {
        'regression_ensemble': {'R2': float(ens[0]), 'RMSE': float(ens[1]), 'MAE': float(ens[2])},
        'classification_ensemble': {
            'accuracy': float(results['CLS_acc']),
            'f1_weighted': float(results['CLS_f1']),
            'auc': float(results['CLS_auc']),
            'per_class_f1': [float(x) for x in f1_per_class],
            'confusion_matrix': cm.tolist()
        },
        'paper_baseline_regression': {'R2': float(results['Paper_RF'][0]),
                                      'RMSE': float(results['Paper_RF'][1])},
        'paper_baseline_classification': {'accuracy': float(results['Paper_CLS'][0]),
                                          'f1_weighted': float(results['Paper_CLS'][1])},
        'mlp_regression_R2': float(results['MLP_reg'][0]),
        'cross_city': {
            'naive_BLR_to_Pune': float(r2_naive_bp),
            'naive_Pune_to_BLR': float(r2_naive_pb),
            'std_BLR_to_Pune': float(r2_std_bp),
            'std_Pune_to_BLR': float(r2_std_pb),
            'combined_CV_R2': float(results['xc_combined']),
            'city_feature_importance': float(results['xc_city_fi'])
        },
        'ablation_baseline_R2': float(base_r2),
        'ablation_results': {k: ([float(v[0]), float(v[1])] if isinstance(v, tuple) else float(v))
                             for k, v in ablation_results.items()},
        'shap_top10': shap_df.head(10).to_dict(orient='records'),
        'novel_features_total_shap_pct': float(novel_total_pct),
        'residual_diagnostics': {
            'mean_bias': float(mean_bias), 'std': float(std_resid),
            'mae': float(mae), 'median_abs_error': float(median_abs),
            'skewness': float(skew_v), 'excess_kurtosis': float(kurt_v),
            'std_over_mae': float(std_resid/mae)
        }
    }
    json.dump(payload, f, indent=2)
log("Saved verification_summary.json + verification_results.txt")

flush()
print("\nDONE.")
