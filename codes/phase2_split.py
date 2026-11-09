"""
Phase 2 — Split Construction
Creates author-DID splits and temporal splits for the English-filtered data.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path

ARTIFACTS = Path("./phase1_artifacts")
SEED = 42

def extract_author_did(comment_id):
    # at://did:plc:XXXX/app.bsky.feed.post/YYYY -> did:plc:XXXX
    return comment_id.replace("at://", "").split("/")[0]

def main():
    print("Loading dataset and language flags...")
    ds = load_dataset(
        "idealab-cs2/bluesky-discourse-quality",
        data_files="bluesky-discourse-quality.parquet",
        token=True,
    )
    df = ds["train"].to_pandas()
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    
    # Merge and filter to English only
    df = df[['comment_id', 'created_utc']].merge(lang_flags, on='comment_id')
    df = df[df['lang'] == 'en'].copy()
    print(f"Filtered to {len(df):,} English comments.")
    
    # Extract Author DIDs
    print("Extracting author DIDs...")
    df['author_did'] = df['comment_id'].apply(extract_author_did)
    
    # ── SPLIT 1: Author-DID Split ────────────────────────────────────────
    print("Building Author-DID split (80/10/10)...")
    unique_dids = df['author_did'].unique()
    np.random.seed(SEED)
    np.random.shuffle(unique_dids)
    
    n_dids = len(unique_dids)
    train_end = int(0.8 * n_dids)
    val_end = int(0.9 * n_dids)
    
    train_dids = set(unique_dids[:train_end])
    val_dids = set(unique_dids[train_end:val_end])
    test_dids = set(unique_dids[val_end:])
    
    def assign_split(did):
        if did in train_dids: return 'train'
        elif did in val_dids: return 'val'
        else: return 'test'
        
    df['split_author'] = df['author_did'].apply(assign_split)
    
    # ── SPLIT 2: Temporal Split ──────────────────────────────────────────
    print("Building Temporal split (80/10/10 by time)...")
    df = df.sort_values('created_utc').reset_index(drop=True)
    n = len(df)
    train_idx = int(0.8 * n)
    val_idx = int(0.9 * n)
    
    df['split_temporal'] = 'test'
    df.loc[:train_idx-1, 'split_temporal'] = 'train'
    df.loc[train_idx:val_idx-1, 'split_temporal'] = 'val'
    
    # ── VERIFICATION ─────────────────────────────────────────────────────
    print("\n--- Verification ---")
    
    # 1. Check DID leakage
    overlap = train_dids.intersection(test_dids)
    print(f"DID leakage (train ∩ test): {len(overlap)} (should be 0)")
    
    # 2. Print sizes
    print("\nAuthor-DID Split Sizes:")
    print(df['split_author'].value_counts())
    
    print("\nTemporal Split Sizes:")
    print(df['split_temporal'].value_counts())
    
    # ── SAVE ─────────────────────────────────────────────────────────────
    out_cols = ['comment_id', 'author_did', 'split_author', 'split_temporal']
    out = df[out_cols]
    
    out.to_parquet("splits_author_temporal.parquet", index=False)
    print("\nSaved to splits_author_temporal.parquet")

if __name__ == "__main__":
    main()