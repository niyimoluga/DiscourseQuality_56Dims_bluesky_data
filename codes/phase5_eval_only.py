import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from datasets import load_dataset

# Import your model and dataset classes from the training script
from phase5_train_task_c_ple_update import PLEDeBERTa, DiscourseDataset, DynamicCollator, LABELLERS, VAR_IDS

# Set the sharing strategy to avoid the "Too many open files" error
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path("/data/nfs/larry/runs_time/task_C_PLE")
    
    print("Loading data...")
    ds = load_dataset("idealab-cs2/bluesky-discourse-quality", data_files="bluesky-discourse-quality.parquet", token=True)
    df = ds["train"].to_pandas()
    splits = pd.read_parquet("splits_author_temporal.parquet")
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    df = df.merge(splits, on="comment_id").merge(lang_flags, on="comment_id")
    df = df[df["lang"] == "en"].copy()
    
    test_author_df = df[df['split_author'] == 'test']
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    
    # Set num_workers=0 to completely bypass the multiprocessing file limit error
    collator = DynamicCollator(tokenizer)
    test_loader = DataLoader(DiscourseDataset(test_author_df, tokenizer), batch_size=16, shuffle=False, collate_fn=collator, num_workers=0)

    print("Loading best model weights...")
    model = PLEDeBERTa().to(device)
    model.load_state_dict(torch.load(BASE_DIR / "best_model.pt"))
    model.eval()

    print("Generating predictions...")
    all_preds = {lab: [] for lab in LABELLERS}
    all_labels = {lab: [] for lab in LABELLERS}
    
    with torch.no_grad():
        for batch in test_loader:
            preds_dict = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            for lab in LABELLERS:
                all_preds[lab].append(preds_dict[lab].cpu().numpy())
                all_labels[lab].append(batch[f"labels_{lab}"].numpy())
                
    data = {}
    for lab in LABELLERS:
        preds = np.concatenate(all_preds[lab], axis=0)
        labels = np.concatenate(all_labels[lab], axis=0)
        for i, v in enumerate(VAR_IDS):
            data[f"{v}_{lab}_true"] = labels[:, i]
            data[f"{v}_{lab}_pred"] = preds[:, i]
            
    out_path = BASE_DIR / "test_predictions_author.csv"
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Successfully saved predictions to {out_path}")

if __name__ == "__main__":
    main()