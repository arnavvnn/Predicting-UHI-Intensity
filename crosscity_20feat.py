"""
Corrected cross-city experiment — TRUE 20-feature shared set.

Key fix vs within_pune_cv.py:
  - Original COMMON had 18 features (excluded NO2, Height_Variability, Surface_Pressure_hPa)
  - All 20 BLR features ARE present in Pune CSV with 0 nulls
  - This script uses all 20

Key fix vs my earlier broken script:
  - BLR LST must be z-scored BEFORE training (same as verify_experiments.py EXP 6)
  - Both yb and yp use z-scored targets → directional transfer is a fair comparison
  - Naive = raw features (BLR raw units vs Pune z-scored) but z-scored targets
  - Std  = per-city z-scored features + z-scored targets

Runs:
  1. Within-BLR  5-fold CV  (20 feats, per-city z-score features, z-scored LST target)
  2. Within-Pune 5-fold CV  (20 feats, same)
  3. Naive BLR→Pune         (raw features, z-scored targets both sides)
  4. Naive Pune→BLR
  5. Per-city z-score BLR→Pune  (z-scored features + z-scored targets)
  6. Per-city z-score Pune→BLR
  7. Combined 5-fold CV         (z-scored features + z-scored targets, both cities)
"""
import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ── Locked hyperparameters (same as all other experiments) ───────────────────
XGB_BEST = dict(n_estimators=929, max_depth=7, learning_rate=0.01944,
                subsample=0.8695, colsample_bytree=0.6321, min_child_weight=6,
                reg_alpha=1.4e-05, reg_lambda=2e-07)
LGB_BEST = dict(n_estimators=1244, num_leaves=51, max_depth=12,
                learning_rate=0.03179, min_child_samples=11, subsample=0.8698,
                colsample_bytree=0.6200, reg_alpha=3.3e-06, reg_lambda=3e-08)
CB_BEST  = dict(iterations=1304, depth=7, learning_rate=0.07352,
                l2_leaf_reg=4.285, subsample=0.8765)

# ── Full 20-feature shared set ────────────────────────────────────────────────
FEATURES_20 = [
    'NDBI','NDVI','NO2_Emissions','Pop_Density','Relative_Humidity','Wind_Speed',
    'Building_Density_Ratio','Dist_Water','Dist_Park','Dist_Highway','Street_Density',
    'Height_Variability','SVF_Proxy','Air_Temp_C','Albedo','MODIS_LST','Soil_Moisture',
    'Surface_Pressure_hPa','Relative_Wealth_Index','Dist_to_RWI_Node'
]

def fit_ensemble(Xt, yt):
    m1 = xgb.XGBRegressor(**XGB_BEST, random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
    m2 = lgb.LGBMRegressor(**LGB_BEST, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt)
    m3 = CatBoostRegressor(**CB_BEST, random_seed=42, verbose=0,
                           allow_writing_files=False, bootstrap_type='Bernoulli').fit(Xt, yt)
    return m1, m2, m3

def predict_ens(m1, m2, m3, X):
    return (m1.predict(X) + m2.predict(X) + m3.predict(X)) / 3

def strat_labels(y):
    mu, sd = y.mean(), y.std()
    return np.where(y < mu - 0.5*sd, 0, np.where(y < mu + 0.5*sd, 1, 2))

def within_cv(X_raw, y_z, label):
    """5-fold CV on a single city. Features are z-scored per city. Target already z-scored."""
    if pd.DataFrame(X_raw).isnull().sum().sum() > 0:
        X_raw = KNNImputer(n_neighbors=5).fit_transform(X_raw)
    X = StandardScaler().fit_transform(X_raw)
    strat = strat_labels(y_z)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r2s, rmses, maes = [], [], []
    for fi, (tr, te) in enumerate(skf.split(X, strat), 1):
        m1, m2, m3 = fit_ensemble(X[tr], y_z[tr])
        pe = predict_ens(m1, m2, m3, X[te])
        r2s.append(r2_score(y_z[te], pe))
        rmses.append(np.sqrt(mean_squared_error(y_z[te], pe)))
        maes.append(mean_absolute_error(y_z[te], pe))
        print(f"  {label} Fold {fi}: R2={r2s[-1]:.4f}")
    r2m = float(np.mean(r2s))
    print(f"  {label} → R2={r2m:.4f} +/-{np.std(r2s):.4f}  RMSE={np.mean(rmses):.4f}  MAE={np.mean(maes):.4f}")
    return r2m, float(np.std(r2s)), float(np.mean(rmses)), float(np.mean(maes))

# ── Load ──────────────────────────────────────────────────────────────────────
df_blr  = pd.read_csv('ultimate_uhi_dataset_with_rwi.csv')
df_pune = pd.read_csv('Pune_MLPR_Standardized_Dataset.csv')

# ── Z-score BLR LST; Pune LST already z-scored ───────────────────────────────
yb = ((df_blr['LST'] - df_blr['LST'].mean()) / df_blr['LST'].std()).values
yp = df_pune['LST'].values  # already z-scored (mean≈0, std≈1)

Xb_raw = df_blr[FEATURES_20].values.astype(float)
Xp_raw = df_pune[FEATURES_20].values.astype(float)

# KNN impute
Xb_raw = KNNImputer(n_neighbors=5).fit_transform(Xb_raw)
Xp_raw = KNNImputer(n_neighbors=5).fit_transform(Xp_raw)

print("=" * 70)
print("CORRECTED CROSS-CITY — 20 FEATURES — BOTH TARGETS Z-SCORED")
print("=" * 70)
print(f"BLR:  {len(yb)} pts | LST z-scored: mean={yb.mean():.3f} std={yb.std():.3f}")
print(f"Pune: {len(yp)} pts | LST z-scored: mean={yp.mean():.3f} std={yp.std():.3f}")

# ── 1. Within-BLR ─────────────────────────────────────────────────────────────
print("\n--- 1. Within-BLR 5-fold CV (20 feats, z-score feats + target) ---")
blr_r2, blr_std, blr_rmse, blr_mae = within_cv(Xb_raw.copy(), yb, 'BLR')

# ── 2. Within-Pune ────────────────────────────────────────────────────────────
print("\n--- 2. Within-Pune 5-fold CV (20 feats, z-score feats + target) ---")
pune_r2, pune_std, pune_rmse, pune_mae = within_cv(Xp_raw.copy(), yp, 'Pune')

# ── 3 & 4. Naive (raw features, z-scored targets) ────────────────────────────
print("\n--- 3. Naive BLR→Pune (raw features, z-scored targets) ---")
m1, m2, m3 = fit_ensemble(Xb_raw, yb)
naive_bp = float(r2_score(yp, predict_ens(m1, m2, m3, Xp_raw)))
print(f"  Naive BLR→Pune R2 = {naive_bp:.4f}")

print("\n--- 4. Naive Pune→BLR (raw features, z-scored targets) ---")
m1, m2, m3 = fit_ensemble(Xp_raw, yp)
naive_pb = float(r2_score(yb, predict_ens(m1, m2, m3, Xb_raw)))
print(f"  Naive Pune→BLR R2 = {naive_pb:.4f}")

# ── 5 & 6. Per-city z-score (features + targets z-scored) ───────────────────
scaler_b = StandardScaler().fit(Xb_raw)
scaler_p = StandardScaler().fit(Xp_raw)
Xb_z = scaler_b.transform(Xb_raw)
Xp_z = scaler_p.transform(Xp_raw)

print("\n--- 5. Per-city z-score BLR→Pune ---")
m1, m2, m3 = fit_ensemble(Xb_z, yb)
std_bp = float(r2_score(yp, predict_ens(m1, m2, m3, Xp_z)))
print(f"  Std BLR→Pune R2 = {std_bp:.4f}")

print("\n--- 6. Per-city z-score Pune→BLR ---")
m1, m2, m3 = fit_ensemble(Xp_z, yp)
std_pb = float(r2_score(yb, predict_ens(m1, m2, m3, Xb_z)))
print(f"  Std Pune→BLR R2 = {std_pb:.4f}")

# ── 7. Combined 5-fold CV ─────────────────────────────────────────────────────
print("\n--- 7. Combined model 5-fold CV (20 feats, z-scored both cities) ---")
X_comb = np.vstack([Xb_z, Xp_z])
y_comb = np.concatenate([yb, yp])
strat_c = strat_labels(y_comb)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
c_r2s, c_rmses, c_maes = [], [], []
for fi, (tr, te) in enumerate(skf.split(X_comb, strat_c), 1):
    m1, m2, m3 = fit_ensemble(X_comb[tr], y_comb[tr])
    pe = predict_ens(m1, m2, m3, X_comb[te])
    c_r2s.append(r2_score(y_comb[te], pe))
    c_rmses.append(np.sqrt(mean_squared_error(y_comb[te], pe)))
    c_maes.append(mean_absolute_error(y_comb[te], pe))
    print(f"  Combined Fold {fi}: R2={c_r2s[-1]:.4f}")
comb_r2   = float(np.mean(c_r2s))
comb_std  = float(np.std(c_r2s))
comb_rmse = float(np.mean(c_rmses))
print(f"  Combined → R2={comb_r2:.4f} +/-{comb_std:.4f}  RMSE={comb_rmse:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: OLD (18 feat COMMON) vs NEW (20 feat full shared)")
print("=" * 70)
old = {'blr_alone': 0.7716, 'pune_alone': 0.7178,
       'naive_bp': -1.3624, 'naive_pb': -1.0104,
       'std_bp':   +0.0681,  'std_pb':   +0.0700,
       'combined':  0.7005}
new = {'blr_alone': blr_r2, 'pune_alone': pune_r2,
       'naive_bp': naive_bp, 'naive_pb': naive_pb,
       'std_bp':   std_bp,   'std_pb':   std_pb,
       'combined': comb_r2}
for k in old:
    d = new[k] - old[k]
    print(f"  {k:20s}  old={old[k]:+.4f}  new={new[k]:+.4f}  Δ={d:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'note': 'All 20 BLR features used (NO2, Height_Var, Surface_Pressure now included). Both LST targets z-scored.',
    'n_features': 20,
    'within_blr':  {'r2_mean': blr_r2,  'r2_std': blr_std,  'rmse': blr_rmse,  'mae': blr_mae},
    'within_pune': {'r2_mean': pune_r2, 'r2_std': pune_std, 'rmse': pune_rmse, 'mae': pune_mae},
    'naive_blr_to_pune': naive_bp,
    'naive_pune_to_blr': naive_pb,
    'std_blr_to_pune':   std_bp,
    'std_pune_to_blr':   std_pb,
    'combined_cv': {'r2_mean': comb_r2, 'r2_std': comb_std, 'rmse': comb_rmse},
    'old_18feat': old,
}
with open('results/crosscity_20feat_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved → results/crosscity_20feat_results.json")
