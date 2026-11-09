"""
Phase 3 — Baseline Ladder (CPU)
Evaluates Mean, BoW+Ridge, and TF-IDF+Ridge for all 5 labellers.
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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
ARTIFACTS = Path("./phase1_artifacts/phase3_baselines")
ARTIFACTS.mkdir(exist_ok=True, parents=True)
SEED = 42

def main():
    print("Loading dataset and splits...")
    ds = load_dataset(
        "idealab-cs2/bluesky-discourse-quality",
        data_files="bluesky-discourse-quality.parquet",
        token=True,
    )
    df = ds["train"].to_pandas()
    splits = pd.read_parquet("splits_author_temporal.parquet")
    
    # Merge and filter by author split + English
    df = df.merge(splits[['comment_id', 'split_author']], on='comment_id')
    # We use the language flags from Phase 1 to only keep English
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    df = df.merge(lang_flags, on='comment_id')
    df = df[df['lang'] == 'en'].copy()

    # Build input text (we'll use the same format as the future DeBERTa model)
    print("Formatting input text...")
    df['input_text'] = (
        "[THREAD ROOT] " + df['root_text'].fillna("") + 
        " [PARENT] " + df['parent_text'].fillna("") + 
        " [TARGET POST] " + df['comment_text']
    )
    
    train_df = df[df['split_author'] == 'train']
    test_df  = df[df['split_author'] == 'test']
    
    print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")

    results = []

    # ── BASELINE 1: Mean Predictor ───────────────────────────────────────
    print("\nRunning Baseline 1: Training Mean...")
    for lab in LABELLERS:
        target_cols = [f"{v}__{lab}" for v in VAR_IDS]
        
        # Compute train means (ignoring NaN)
        train_means = train_df[target_cols].mean()
        
        # Predict that constant for all test rows
        y_pred = np.tile(train_means.values, (len(test_df), 1))
        y_true = test_df[target_cols].values
        
        # Filter out NaNs in test for metrics
        mask = ~np.isnan(y_true)
        
        r2s, maes = [], []
        for i in range(len(VAR_IDS)):
            if mask[:, i].sum() > 0:
                r2 = r2_score(y_true[mask[:, i], i], y_pred[mask[:, i], i])
                mae = mean_absolute_error(y_true[mask[:, i], i], y_pred[mask[:, i], i])
                r2s.append(r2)
                maes.append(mae)
                
        results.append({
            "Model": "Mean", "Labeller": lab, 
            "Mean_R2": np.mean(r2s), "Mean_MAE": np.mean(maes)
        })
        print(f"  {lab}: R² = {np.mean(r2s):.4f} | MAE = {np.mean(maes):.4f}")

    # ── BASELINES 2 & 3: Lexical + Ridge ─────────────────────────────────
    for vec_name, vectorizer in [
        ("BoW", CountVectorizer(max_features=50000, ngram_range=(1, 1), min_df=3)),
        ("TFIDF", TfidfVectorizer(max_features=100000, ngram_range=(1, 2), min_df=3, sublinear_tf=True))
    ]:
        print(f"\nFitting {vec_name} Vectorizer...")
        X_train = vectorizer.fit_transform(train_df['input_text'])
        X_test = vectorizer.transform(test_df['input_text'])
        
        print(f"Running Baseline: {vec_name} + Ridge...")
        for lab in LABELLERS:
            target_cols = [f"{v}__{lab}" for v in VAR_IDS]
            Y_train = train_df[target_cols].values
            Y_test = test_df[target_cols].values
            
            # Impute NaNs in train with column mean (Ridge can't handle NaNs)
            col_means = np.nanmean(Y_train, axis=0)
            Y_train_imputed = np.where(np.isnan(Y_train), col_means, Y_train)
            
            # Fit Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, Y_train_imputed)
            
            Y_pred = ridge.predict(X_test)
            
            # Evaluate
            mask = ~np.isnan(Y_test)
            r2s, maes = [], []
            for i in range(len(VAR_IDS)):
                if mask[:, i].sum() > 0:
                    r2 = r2_score(Y_test[mask[:, i], i], Y_pred[mask[:, i], i])
                    mae = mean_absolute_error(Y_test[mask[:, i], i], Y_pred[mask[:, i], i])
                    r2s.append(r2)
                    maes.append(mae)
            
            results.append({
                "Model": vec_name, "Labeller": lab, 
                "Mean_R2": np.mean(r2s), "Mean_MAE": np.mean(maes)
            })
            print(f"  {lab}: R² = {np.mean(r2s):.4f} | MAE = {np.mean(maes):.4f}")

    # ── SAVE RESULTS ─────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    res_df.to_csv(ARTIFACTS / "baseline_results.csv", index=False)
    print(f"\nAll results saved to {ARTIFACTS}/baseline_results.csv")

    # Pivot for easy viewing
    pivot = res_df.pivot(index='Labeller', columns='Model', values='Mean_R2')
    print("\n=== Mean R² by Model and Labeller ===")
    print(pivot.round(4))

if __name__ == "__main__":
    main()