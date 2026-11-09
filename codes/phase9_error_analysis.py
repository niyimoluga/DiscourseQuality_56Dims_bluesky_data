import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset

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
OUT_DIR = Path("./phase1_artifacts")

def main():
    print("Reconstructing test dataset to get raw text...")
    ds = load_dataset("idealab-cs2/bluesky-discourse-quality", data_files="bluesky-discourse-quality.parquet", token=True)
    df = ds["train"].to_pandas()
    splits = pd.read_parquet("splits_author_temporal.parquet")
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    
    df = df.merge(splits, on="comment_id").merge(lang_flags, on="comment_id")
    df = df[df["lang"] == "en"].copy()
    
    # This MUST match the exact filtering and order used in training
    test_df = df[df['split_author'] == 'test'].reset_index(drop=True)
    
    results = []

    print("Analyzing Task A predictions...")
    for lab in LABELLERS:
        # Load Task A CSV
        csv_path = BASE_DIR / f"task_A_{lab}" / "test_predictions_author.csv"
        if not csv_path.exists():
            continue
            
        pred_df = pd.read_csv(csv_path)
        
        # Calculate per-row MSE (ignoring NaNs)
        sq_errs = []
        for var in VAR_IDS:
            err = (pred_df[f"{var}_true"] - pred_df[f"{var}_pred"])**2
            sq_errs.append(err)
            
        sq_errs = pd.DataFrame(sq_errs).T
        row_mse = sq_errs.mean(axis=1, skipna=True)
        
        # Find best and worst indices
        worst_idx = row_mse.nlargest(3).index
        best_idx = row_mse.nsmallest(3).index
        
        for rank, idx in enumerate(worst_idx):
            _append_result(results, "Task A", lab, "Worst", rank+1, idx, row_mse[idx], sq_errs.iloc[idx], test_df, pred_df)
            
        for rank, idx in enumerate(best_idx):
            _append_result(results, "Task A", lab, "Best", rank+1, idx, row_mse[idx], sq_errs.iloc[idx], test_df, pred_df)

    print("Analyzing Task C (PLE) predictions...")
    csv_path_c = BASE_DIR / "task_C_PLE" / "test_predictions_author.csv"
    if csv_path_c.exists():
        pred_df_c = pd.read_csv(csv_path_c)
        
        for lab in LABELLERS:
            sq_errs = []
            for var in VAR_IDS:
                # Note the column naming difference for Task C
                err = (pred_df_c[f"{var}_{lab}_true"] - pred_df_c[f"{var}_{lab}_pred"])**2
                sq_errs.append(err)
                
            sq_errs = pd.DataFrame(sq_errs).T
            row_mse = sq_errs.mean(axis=1, skipna=True)
            
            worst_idx = row_mse.nlargest(3).index
            best_idx = row_mse.nsmallest(3).index
            
            # For Task C, we need to adapt the _append_result function slightly
            # (we'll just duplicate the logic to handle the column names)
            for rank, idx in enumerate(worst_idx):
                _append_task_c_result(results, lab, "Worst", rank+1, idx, row_mse[idx], sq_errs.iloc[idx], test_df, pred_df_c)
                
            for rank, idx in enumerate(best_idx):
                _append_task_c_result(results, lab, "Best", rank+1, idx, row_mse[idx], sq_errs.iloc[idx], test_df, pred_df_c)

    # Save to CSV
    res_df = pd.DataFrame(results)
    out_path = OUT_DIR / "error_analysis.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n✅ Saved error analysis to {out_path}")
    
    # Print a snippet
    print("\n=== SAMPLE WORST PREDICTIONS ===")
    print(res_df[res_df['Type'] == 'Worst'].head(3)[['Labeller', 'Overall_MSE', 'Top_Error_Var', 'Comment_Text']].to_string(index=False, max_colwidth=50))

def _append_result(results, model, lab, mtype, rank, idx, mse, row_sq_errs, test_df, pred_df):
    # Find which variable had the highest error
    worst_var_idx = row_sq_errs.idxmax()
    worst_var = VAR_IDS[worst_var_idx]
    true_val = pred_df.loc[idx, f"{worst_var}_true"]
    pred_val = pred_df.loc[idx, f"{worst_var}_pred"]
    
    results.append({
        "Model": model,
        "Labeller": lab,
        "Type": mtype,
        "Rank": rank,
        "Overall_MSE": mse,
        "Top_Error_Var": worst_var,
        "Var_True": true_val,
        "Var_Pred": pred_val,
        "Comment_Text": test_df.loc[idx, 'comment_text'][:300].replace("\n", " ")
    })

def _append_task_c_result(results, lab, mtype, rank, idx, mse, row_sq_errs, test_df, pred_df_c):
    worst_var_idx = row_sq_errs.idxmax()
    worst_var = VAR_IDS[worst_var_idx]
    true_val = pred_df_c.loc[idx, f"{worst_var}_{lab}_true"]
    pred_val = pred_df_c.loc[idx, f"{worst_var}_{lab}_pred"]
    
    results.append({
        "Model": "Task C",
        "Labeller": lab,
        "Type": mtype,
        "Rank": rank,
        "Overall_MSE": mse,
        "Top_Error_Var": worst_var,
        "Var_True": true_val,
        "Var_Pred": pred_val,
        "Comment_Text": test_df.loc[idx, 'comment_text'][:300].replace("\n", " ")
    })

if __name__ == "__main__":
    main()