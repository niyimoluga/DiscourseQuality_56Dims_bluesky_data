"""
Phase 6 — Aggregate Task C (PLE) Results
Compares the unified PLE model against Task A and the Phase 3 baselines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
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
TASK_C_CSV = BASE_DIR / "task_C_PLE" / "test_predictions_author.csv"
TASK_A_RESULTS_CSV = Path("./phase1_artifacts/phase4_deberta_results.csv")

def main():
    print("Loading Task C (PLE) Predictions...")
    if not TASK_C_CSV.exists():
        print(f"Error: {TASK_C_CSV} not found.")
        return
        
    df = pd.read_csv(TASK_C_CSV)
    results = []

    for lab in LABELLERS:
        r2s, maes, pearson_rs = [], [], []
        
        for var in VAR_IDS:
            true_col = f"{var}_{lab}_true"
            pred_col = f"{var}_{lab}_pred"
            
            if true_col not in df.columns:
                continue
                
            valid = df[[true_col, pred_col]].dropna()
            if len(valid) < 10:
                continue
                
            y_true = valid[true_col]
            y_pred = valid[pred_col]
            
            # Metrics
            r2s.append(r2_score(y_true, y_pred))
            maes.append(mean_absolute_error(y_true, y_pred))
            
            if y_true.std() > 1e-6 and y_pred.std() > 1e-6:
                pr, _ = pearsonr(y_true, y_pred)
                pearson_rs.append(pr)
            else:
                pearson_rs.append(0.0)
                
        results.append({
            "Labeller": lab,
            "Model": "Task_C_PLE",
            "Mean_R2": np.mean(r2s),
            "Mean_MAE": np.mean(maes),
            "Mean_Pearson": np.mean(pearson_rs)
        })
        
        print(f"  ✅ {lab}: R² = {np.mean(r2s):.4f} | MAE = {np.mean(maes):.4f} | Pearson r = {np.mean(pearson_rs):.4f}")

    # Load Task A results
    print("\nLoading Task A Results...")
    if TASK_A_RESULTS_CSV.exists():
        task_a_df = pd.read_csv(TASK_A_RESULTS_CSV)
        # Filter only the DeBERTa rows from Task A
        task_a_deberta = task_a_df[task_a_df["Model"] == "DeBERTa"]
        
        task_c_df = pd.DataFrame(results)
        
        # Combine and pivot
        combined = pd.concat([task_a_deberta, task_c_df], ignore_index=True)
        
        print("\n=== FINAL COMPARISON: Task A vs Task C (Mean R²) ===")
        pivot = combined.pivot(index='Labeller', columns='Model', values='Mean_R2')
        
        # Reorder rows to match Labellers list
        pivot = pivot.reindex(LABELLERS)
        
        # Calculate improvement
        pivot["PLE_Lift"] = pivot["Task_C_PLE"] - pivot["DeBERTa"]
        
        print(pivot.round(4))
        
        # Save the final comparison
        out_csv = Path("./phase1_artifacts/final_task_A_vs_C.csv")
        pivot.to_csv(out_csv)
        print(f"\nSaved final comparison to {out_csv}")

if __name__ == "__main__":
    main()