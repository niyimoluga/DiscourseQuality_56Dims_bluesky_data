import pandas as pd
from datasets import load_dataset
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 0

print("Loading dataset...")
ds = load_dataset(
    "idealab-cs2/bluesky-discourse-quality",
    data_files="bluesky-discourse-quality.parquet",
    token=True,
)
df = ds["train"].to_pandas()

print("Detecting languages (this will take 10-15 minutes on 467k rows)...")
# Using a safe wrapper in case a comment is too short or weird
def safe_detect(text):
    try:
        return detect(str(text))
    except Exception:
        return "unknown"

df['lang'] = df['comment_text'].apply(safe_detect)

print("\n--- Language Distribution ---")
counts = df['lang'].value_counts()
print(counts.head(15))

print(f"\nTotal rows: {len(df)}")
print(f"English ('en'): {counts.get('en', 0)} ({100 * counts.get('en', 0) / len(df):.2f}%)")
print(f"Non-English: {len(df) - counts.get('en', 0)}")

# Save the language flags for Phase 2
df[['comment_id', 'lang']].to_parquet("phase1_language_flags.parquet", index=False)
print("\nSaved to phase1_language_flags.parquet")