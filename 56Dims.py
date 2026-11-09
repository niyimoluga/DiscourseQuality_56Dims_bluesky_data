import json
from datasets import load_dataset

ds = load_dataset(
    "idealab-cs2/bluesky-discourse-quality",
    data_files="bluesky-discourse-quality.parquet",
    token=True,
    split="train",
)

subset = ds.select(range(50))

with open("first50.json", "w") as f:
    json.dump(subset.to_list(), f, indent=2, default=str)

print("wrote first50.json")