import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUTS = 'outputs'

def run_shap():
    lgb_model   = joblib.load(f'{OUTPUTS}/lgb_model.pkl')
    feature_cols = joblib.load(f'{OUTPUTS}/feature_cols.pkl')
    X_train, _   = joblib.load(f'{OUTPUTS}/train_data.pkl')
    X_test, y_test = joblib.load(f'{OUTPUTS}/test_data.pkl')

    print("Computing SHAP values...")
    explainer   = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)

    # --- Plot 1: Summary bar (mean absolute SHAP = global feature importance) ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      plot_type='bar', show=False, max_display=15)
    plt.title("Top 15 Features by SHAP Importance (LightGBM)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS}/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved shap_bar.png")

    # --- Plot 2: Beeswarm (shows direction of each feature's effect) ---
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                      show=False, max_display=15)
    plt.title("SHAP Beeswarm — Feature Impact Direction (LightGBM)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved shap_beeswarm.png")

    # --- Top 3 features ---
    mean_shap = np.abs(shap_values).mean(axis=0)
    top3_idx  = mean_shap.argsort()[::-1][:3]
    print("\n📊 Top 3 Features Influencing RUL Prediction:")
    for rank, idx in enumerate(top3_idx, 1):
        print(f"  {rank}. {feature_cols[idx]:30s}  mean|SHAP| = {mean_shap[idx]:.4f}")

    joblib.dump((shap_values, feature_cols), f'{OUTPUTS}/shap_data.pkl')
    return shap_values, feature_cols


if __name__ == '__main__':
    run_shap()
