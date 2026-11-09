"""
Phase 4.5 — Aggregate Results
Loads predictions from all 5 Task A models and compares them to the Phase 3 baselines.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Config
LABELLERS = ["qwen3.5", "gemma4", "mistral3.2", "gptoss", "granite4.2"]
VAR_IDS = [
    "D1a", "D1b", "D3+", "D3\u2212", "D4+", "D4\u2212", "D5+", "D5\u2212",
    "D6+", "D6\u2212", "D7+", "D7\u2212", "F7+", "F7\u2212", "D8", "D9", "D11",
    "A1", "A2", "A3+", "A3\u2212", "A4", "A5+", "A5\u2212", "A6", "A11",
    "A12", "A13", "F9", "E1+", "E1\u2212", "E2", "E3", "E4+", "E4\u2212", "E7",
    "I1+", "I1\u2212", "I3+", "I3\u2212", "I4", "I8", "I9", "F13",
    "P1", "P2", "P8", "P3", "P4", "P5+", "P5\u2212", "P6+", "P6\u2212", "P7", "P10", "P11",
]

BASE_DIR = Path("/data/nfs/larry/runs_time")
BASELINE_CSV = Path("./phase1_artifacts/phase3_baselines/baseline_results.csv")
OUT_CSV = Path("./phase1_artifacts/phase4_deberta_results.csv")

def main():
    results = []
    
    print("Aggregating Task A (DeBERTa) Results...")
    for lab in LABELLERS:
        csv_path = BASE_DIR / f"task_A_{lab}" / "test_predictions_author.csv"
        if not csv_path.exists():
            print(f"  ⚠️ Missing predictions for {lab} at {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        r2s, maes, pearson_rs = [], [], []
        
        # Calculate metrics per variable
        for var in VAR_IDS:
            true_col = f"{var}_true"
            pred_col = f"{var}_pred"
            
            if true_col not in df.columns:
                continue
                
            # Drop NaNs (important for gpt-oss)
            valid = df[[true_col, pred_col]].dropna()
            if len(valid) < 10:
                continue
                
            y_true = valid[true_col]
            y_pred = valid[pred_col]
            
            # R2
            r2 = r2_score(y_true, y_pred)
            r2s.append(r2)
            
            # MAE
            mae = mean_absolute_error(y_true, y_pred)
            maes.append(mae)
            
            # Pearson r (safe check for constant arrays)
            if y_true.std() > 1e-6 and y_pred.std() > 1e-6:
                pr, _ = pearsonr(y_true, y_pred)
                pearson_rs.append(pr)
            else:
                pearson_rs.append(0.0)
                
        # Average across 56 variables
        results.append({
            "Labeller": lab,
            "Model": "DeBERTa",
            "Mean_R2": np.mean(r2s),
            "Mean_MAE": np.mean(maes),
            "Mean_Pearson": np.mean(pearson_rs)
        })
        
        print(f"  ✅ {lab}: R² = {np.mean(r2s):.4f} | MAE = {np.mean(maes):.4f} | Pearson r = {np.mean(pearson_rs):.4f}")

    # Load Phase 3 Baselines
    print("\nLoading Phase 3 Baselines...")
    if BASELINE_CSV.exists():
        baselines = pd.read_csv(BASELINE_CSV)
        # Fill NA Pearson if baseline didn't calculate it
        if "Mean_Pearson" not in baselines.columns:
            baselines["Mean_Pearson"] = np.nan
            
        df_results = pd.DataFrame(results)
        # Combine
        combined = pd.concat([baselines, df_results], ignore_index=True)
    else:
        print("  ⚠️ Baseline CSV not found. Only showing DeBERTa results.")
        combined = pd.DataFrame(results)

    # Save and Print
    combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved aggregated results to {OUT_CSV}")
    
    # Print a beautiful pivot table for the thesis
    print("\n=== FINAL TASK A RESULTS (Mean R² across 56 variables) ===")
    pivot = combined.pivot(index='Labeller', columns='Model', values='Mean_R2')
    
    # Reorder columns to make sense (Baselines -> DeBERTa)
    col_order = [c for c in ["Mean", "BoW", "TFIDF", "DeBERTa"] if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.round(4))
    
    print("\n=== FINAL TASK A RESULTS (Mean Pearson r) ===")
    pivot_pr = combined.pivot(index='Labeller', columns='Model', values='Mean_Pearson')
    pivot_pr = pivot_pr[[c for c in col_order if c in pivot_pr.columns]]
    print(pivot_pr.round(4))

if __name__ == "__main__":
    main()