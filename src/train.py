import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import joblib
from preprocess import preprocess

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

MODELS_DIR = 'outputs'


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, 125)
    mae  = mean_absolute_error(y_test, preds)
    r    = rmse(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"  {name:20s} | MAE: {mae:.2f} | RMSE: {r:.2f} | R²: {r2:.4f}")
    return preds, mae, r, r2


def train_all():
    X_train, y_train, X_test, y_test, feature_cols, train_df = preprocess()
    print("\n--- Training Models ---\n")

    # 1. Random Forest (baseline)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds, *_ = evaluate("Random Forest", rf, X_test, y_test)
    joblib.dump(rf, f'{MODELS_DIR}/rf_model.pkl')

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_preds, *_ = evaluate("XGBoost", xgb_model, X_test, y_test)
    joblib.dump(xgb_model, f'{MODELS_DIR}/xgb_model.pkl')

    # 3. LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    lgb_preds, *_ = evaluate("LightGBM", lgb_model, X_test, y_test)
    joblib.dump(lgb_model, f'{MODELS_DIR}/lgb_model.pkl')

    # Save feature list and test data for SHAP
    joblib.dump(feature_cols, f'{MODELS_DIR}/feature_cols.pkl')
    joblib.dump((X_test, y_test), f'{MODELS_DIR}/test_data.pkl')
    joblib.dump((X_train, y_train), f'{MODELS_DIR}/train_data.pkl')

    print("\n✓ All models saved to outputs/")
    return lgb_model, X_train, X_test, y_test, feature_cols


if __name__ == '__main__':
    train_all()
