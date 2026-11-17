import json

print("=== VERIFYING REGRESSION METRICS ===")
with open("new_research_project/data/predictions/stacking_regression_metrics.json") as f:
    stack_m = json.load(f)

for h in ["h1", "h6"]:
    test_metrics = stack_m[h]["test"]
    print(f"Stacking {h}: MAE={test_metrics['mae']:.4f}, RMSE={test_metrics['rmse']:.4f}, R2={test_metrics['r2']:.4f}, Bias={test_metrics['bias']:.4f}")

with open("new_research_project/data/predictions/regression_metrics.json") as f:
    reg_m = json.load(f)

for h in ["h24", "h48"]:
    lgb_test = reg_m[h]["models"]["lightgbm"]["test"]
    print(f"LightGBM {h}: MAE={lgb_test['mae']:.4f}, RMSE={lgb_test['rmse']:.4f}, R2={lgb_test['r2']:.4f}, Bias={lgb_test['bias']:.4f}")

print("\n=== VERIFYING CLASSIFICATION METRICS (h24) ===")
with open("new_research_project/data/predictions/classification_metrics.json") as f:
    cls_m = json.load(f)

for thresh in ["acute_excursion_60", "severe_spike_120"]:
    if thresh in cls_m["h24"]:
        for model in ["lightgbm", "xgboost"]:
            res = cls_m["h24"][thresh]["models"][model]["test"]
            print(f"Horizon h24 | {thresh} | {model}: F1={res['f1']:.4f}, ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, Recall={res['recall']:.4f}, Precision={res['precision']:.4f}")

print("\n=== VERIFYING SHAP EXTREMES AND DOMAINS ===")
with open("new_research_project/data/predictions/global_feature_importance.json") as f:
    gfi = json.load(f)

for h in ["h1", "h48"]:
    print(f"SHAP {h} total: {gfi[h]['total_shap']:.2f} ug/m3")
    print(f"  Historical PM2.5 share: {gfi[h]['domain_share_pct']['Historical PM2.5 Dynamics']:.2f}%")
    print(f"  Thermodynamics share:   {gfi[h]['domain_share_pct']['Thermodynamics & Boundary Layer']:.2f}%")
