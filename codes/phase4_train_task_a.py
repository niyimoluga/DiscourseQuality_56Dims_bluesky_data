import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from transformers import DebertaV2Tokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import wandb
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
LABELLERS = ["qwen3.5", "gemma4", "mistral3.2", "gptoss", "granite4.2"]
VAR_IDS = [
    "D1a", "D1b", "D3+", "D3\u2212", "D4+", "D4\u2212", "D5+", "D5\u2212",
    "D6+", "D6\u2212", "D7+", "D7\u2212", "F7+", "F7\u2212", "D8", "D9", "D11",
    "A1", "A2", "A3+", "A3\u2212", "A4", "A5+", "A5\u2212", "A6", "A11",
    "A12", "A13", "F9", "E1+", "E1\u2212", "E2", "E3", "E4+", "E4\u2212", "E7",
    "I1+", "I1\u2212", "I3+", "I3\u2212", "I4", "I8", "I9", "F13",
    "P1", "P2", "P8", "P3", "P4", "P5+", "P5\u2212", "P6+", "P6\u2212", "P7", "P10", "P11",
]
BACKBONE = "microsoft/deberta-v3-large"
BASE_DIR = Path("/data/nfs/larry/runs_time")
SEED = 42

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ── DATASET & COLLATOR ──────────────────────────────────────────────────────
class DiscourseDataset(Dataset):
    def __init__(self, df, tokenizer, labeller):
        self.texts = (
            "[THREAD ROOT] " + df['root_text'].fillna("") + 
            " [PARENT] " + df['parent_text'].fillna("") + 
            " [TARGET POST] " + df['comment_text']
        ).tolist()
        self.targets = df[[f"{v}__{labeller}" for v in VAR_IDS]].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # No padding here! Collator handles it dynamically.
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=512)
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(self.targets[idx], dtype=torch.float32)
        }

class DynamicCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = torch.stack([item["labels"] for item in batch])
        
        # Dynamic pad to longest sequence in THIS batch
        enc = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc

# ── MODEL ───────────────────────────────────────────────────────────────────
class PlainDeBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BACKBONE)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256),  nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 56),   nn.Sigmoid()
        )
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:, 0, :])

# ── CSV LOGGER (Your Custom Module Request) ─────────────────────────────────
def save_predictions_csv(model, dataloader, device, out_path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Create DataFrame
    cols = []
    data = {}
    for i, v in enumerate(VAR_IDS):
        cols.extend([f"{v}_true", f"{v}_pred"])
        data[f"{v}_true"] = labels[:, i]
        data[f"{v}_pred"] = preds[:, i]
    
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeller", type=str, required=True, choices=LABELLERS)
    parser.add_argument("--epochs", type=int, default=1, help="1 for sanity check, 4 for full")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup output directory
    out_dir = BASE_DIR / f"task_A_{args.labeller}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # W&B Init
    run = wandb.init(
        entity="niyimoluga85-universty-of-technology-graz",
        project="56-dims-regression",
        name=f"task_A_{args.labeller}",
        dir=str(out_dir),  # <--- ADD THIS LINE
        config=vars(args)
    )

    # Load Data
    print("Loading data and splits...")
    ds = load_dataset("idealab-cs2/bluesky-discourse-quality", data_files="bluesky-discourse-quality.parquet", token=True)
    df = ds["train"].to_pandas()
    splits = pd.read_parquet("splits_author_temporal.parquet")
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    
    df = df.merge(splits, on="comment_id").merge(lang_flags, on="comment_id")
    df = df[df["lang"] == "en"].copy()

    train_df = df[df['split_author'] == 'train']
    val_df   = df[df['split_author'] == 'val']
    test_author_df   = df[df['split_author'] == 'test']
    test_temporal_df = df[df['split_temporal'] == 'test']

    tokenizer = DebertaV2Tokenizer.from_pretrained(BACKBONE)
    
    # We compute training means for R2 here
    train_means = train_df[[f"{v}__{args.labeller}" for v in VAR_IDS]].mean().values
    train_means_tensor = torch.tensor(train_means, dtype=torch.float32).to(device)

    # Datasets & Loaders
    collator = DynamicCollator(tokenizer)
    train_loader = DataLoader(DiscourseDataset(train_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)
    val_loader   = DataLoader(DiscourseDataset(val_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
    test_a_loader = DataLoader(DiscourseDataset(test_author_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
    test_t_loader = DataLoader(DiscourseDataset(test_temporal_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    # Model, Optimizer, Loss
    model = PlainDeBERTa().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)
    scaler = GradScaler()

    # Training Loop
    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # NaN masking (for missing cells)
            mask = ~torch.isnan(labels)
            safe_labels = torch.nan_to_num(labels, nan=0.0)

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask)
                # Plain MSE with NaN masking
                loss = F.mse_loss(logits * mask, safe_labels * mask)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Epoch Validation
        model.eval()
        val_loss = 0
        val_mse_dim = torch.zeros(56, device=device)
        val_ss_tot = torch.zeros(56, device=device)
        val_n = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                mask = ~torch.isnan(labels)
                safe_labels = torch.nan_to_num(labels, nan=0.0)
                
                logits = model(input_ids, attention_mask)
                loss = F.mse_loss(logits * mask, safe_labels * mask)
                val_loss += loss.item()
                
                sq_err = (logits - safe_labels)**2 * mask
                val_mse_dim += sq_err.sum(dim=0)
                val_ss_tot += ((safe_labels - train_means_tensor)**2 * mask).sum(dim=0)
                val_n += labels.shape[0]
                
        avg_val_loss = val_loss / len(val_loader)
        val_r2 = (1.0 - (val_mse_dim / (val_ss_tot + 1e-9))).mean().item()
        
        print(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val R²: {val_r2:.4f}")
        
        # Log to W&B and CSV
        metrics = {"epoch": epoch+1, "train_loss": epoch_loss/len(train_loader), "val_loss": avg_val_loss, "val_r2": val_r2}
        run.log(metrics)
        history.append(metrics)
        pd.DataFrame(history).to_csv(out_dir / "metrics_log.csv", index=False)

        # Save epoch checkpoint
        torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print("  -> Saved new best_model.pt")

    # Final Evaluation on Test Sets
    print("\nRunning final test evaluation...")
    if os.path.exists(out_dir / "best_model.pt"):
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    
    save_predictions_csv(model, test_a_loader, device, out_dir / "test_predictions_author.csv")
    save_predictions_csv(model, test_t_loader, device, out_dir / "test_predictions_temporal.csv")
    
    run.finish()
    print("Done!")

if __name__ == "__main__":
    main()