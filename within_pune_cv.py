"""
Within-Pune 5-fold CV experiment.
Mirrors the Bengaluru headline protocol exactly:
  - Same 18 common features
  - Same tuned ensemble (XGB + LGBM + CatBoost with locked hyperparameters)
  - Stratified 5-fold on LST tertiles (within Pune)
  - Standardized features (per-city z-score, here just within Pune)

Also reports the same Bengaluru-alone numbers from the 18-feature common set
for an apples-to-apples comparison.
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

# Locked-in tuned hyperparameters (60-trial Optuna run on BLR's 20-feat set)
XGB_BEST = dict(n_estimators=929, max_depth=7, learning_rate=0.01944, subsample=0.8695,
                colsample_bytree=0.6321, min_child_weight=6, reg_alpha=1.4e-05, reg_lambda=2e-07)
LGB_BEST = dict(n_estimators=1244, num_leaves=51, max_depth=12, learning_rate=0.03179,
                min_child_samples=11, subsample=0.8698, colsample_bytree=0.6200,
                reg_alpha=3.3e-06, reg_lambda=3e-08)
CB_BEST  = dict(iterations=1304, depth=7, learning_rate=0.07352, l2_leaf_reg=4.285, subsample=0.8765)

COMMON = ['EVI','NDBI','NDVI','Pop_Density','Relative_Humidity','Wind_Speed',
          'Building_Density_Ratio','Dist_Water','Dist_Park','Dist_Highway',
          'Street_Density','SVF_Proxy','Air_Temp_C','Albedo','MODIS_LST',
          'Soil_Moisture','Relative_Wealth_Index','Dist_to_RWI_Node']

def run_within_city(df, label):
    y = df['LST'].values if label == 'BLR' else df['LST'].values  # both are 'LST'
    # Stratification: tertiles based on this city's own LST distribution
    mu, sd = y.mean(), y.std()
    strat = np.where(y < mu - 0.5*sd, 0,
                     np.where(y < mu + 0.5*sd, 1, 2))
    counts = np.bincount(strat)
    print(f"\n--- {label} within-city 5-fold CV ---")
    print(f"  Samples: {len(y)} | LST mean={mu:.3f}, std={sd:.3f}")
    print(f"  Class counts: Cool={counts[0]}, Neutral={counts[1]}, Hot={counts[2]}")
    print(f"  Boundaries: {mu-0.5*sd:.3f} / {mu+0.5*sd:.3f}")

    X_raw = df[COMMON].copy()
    # KNN impute if any nulls
    if X_raw.isnull().sum().sum() > 0:
        X_raw = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(X_raw), columns=COMMON)
    # Per-city z-score (matches cross-city protocol)
    X = pd.DataFrame(StandardScaler().fit_transform(X_raw), columns=COMMON)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2_xgb, fold_r2_lgb, fold_r2_cb, fold_r2_ens = [], [], [], []
    fold_rmse_ens, fold_mae_ens = [], []

    for fi, (tr, te) in enumerate(skf.split(X, strat), 1):
        Xt, yt = X.iloc[tr], y[tr]; Xv, yv = X.iloc[te], y[te]
        m1 = xgb.XGBRegressor(**XGB_BEST, random_state=42, n_jobs=-1, verbosity=0).fit(Xt, yt)
        m2 = lgb.LGBMRegressor(**LGB_BEST, random_state=42, n_jobs=-1, verbose=-1).fit(Xt, yt)
        m3 = CatBoostRegressor(**CB_BEST, random_seed=42, verbose=0,
                               allow_writing_files=False, bootstrap_type='Bernoulli').fit(Xt, yt)
        p1, p2, p3 = m1.predict(Xv), m2.predict(Xv), m3.predict(Xv)
        pe = (p1 + p2 + p3) / 3
        fold_r2_xgb.append(r2_score(yv, p1))
        fold_r2_lgb.append(r2_score(yv, p2))
        fold_r2_cb.append(r2_score(yv, p3))
        fold_r2_ens.append(r2_score(yv, pe))
        fold_rmse_ens.append(np.sqrt(mean_squared_error(yv, pe)))
        fold_mae_ens.append(mean_absolute_error(yv, pe))
        print(f"  Fold {fi}: XGB={fold_r2_xgb[-1]:.4f}  LGB={fold_r2_lgb[-1]:.4f}  CB={fold_r2_cb[-1]:.4f}  ENS={fold_r2_ens[-1]:.4f}")

    print(f"\n  RESULTS (mean +/- std across 5 folds):")
    print(f"  XGB        R2 = {np.mean(fold_r2_xgb):.4f} +/- {np.std(fold_r2_xgb):.4f}")
    print(f"  LGB        R2 = {np.mean(fold_r2_lgb):.4f} +/- {np.std(fold_r2_lgb):.4f}")
    print(f"  CB         R2 = {np.mean(fold_r2_cb):.4f} +/- {np.std(fold_r2_cb):.4f}")
    print(f"  ENSEMBLE   R2 = {np.mean(fold_r2_ens):.4f} +/- {np.std(fold_r2_ens):.4f}")
    print(f"           RMSE = {np.mean(fold_rmse_ens):.4f}")
    print(f"            MAE = {np.mean(fold_mae_ens):.4f}")

    return {
        'samples': int(len(y)),
        'r2_xgb_mean': float(np.mean(fold_r2_xgb)),
        'r2_lgb_mean': float(np.mean(fold_r2_lgb)),
        'r2_cb_mean': float(np.mean(fold_r2_cb)),
        'r2_ensemble_mean': float(np.mean(fold_r2_ens)),
        'r2_ensemble_std': float(np.std(fold_r2_ens)),
        'rmse_ensemble': float(np.mean(fold_rmse_ens)),
        'mae_ensemble': float(np.mean(fold_mae_ens)),
        'fold_r2_ens': [float(x) for x in fold_r2_ens],
    }


df_blr = pd.read_csv('ultimate_uhi_dataset_with_rwi.csv')
df_pune = pd.read_csv('Pune_MLPR_Standardized_Dataset.csv')

print("="*70)
print("WITHIN-CITY 5-FOLD CV (apples-to-apples comparison)")
print("="*70)
print(f"Using 18 common features (cross-city compatible set)")

blr_res = run_within_city(df_blr, 'BLR')
pune_res = run_within_city(df_pune, 'Pune')

print("\n" + "="*70)
print("HEADLINE COMPARISON")
print("="*70)
print(f"  BLR  alone (18 common feats, 2433 samples):  R2 = {blr_res['r2_ensemble_mean']:.4f}")
print(f"  Pune alone (18 common feats, 2382 samples):  R2 = {pune_res['r2_ensemble_mean']:.4f}")
print(f"  Note: BLR headline (20 features, raw LST in C) was R2 = 0.7940")
print(f"  Note: Pune LST is already z-scored (unit variance)")

# Append to verification summary
try:
    with open('results/verification_summary.json') as f:
        summary = json.load(f)
    summary['within_city_cv'] = {
        'blr_alone_18feat': blr_res,
        'pune_alone_18feat': pune_res,
    }
    with open('results/verification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n  Appended to verification_summary.json")
except Exception as e:
    print(f"  Could not update json: {e}")
