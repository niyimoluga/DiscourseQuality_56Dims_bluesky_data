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

# Logging frequencies
LOG_STEPS = 50       # Log metrics to W&B and CSV every N steps
EVAL_STEPS = 500     # Run evaluation every N steps

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
        enc = self.tokenizer.pad(
            {"input_ids": [item["input_ids"] for item in batch],
             "attention_mask": [item["attention_mask"] for item in batch]},
            padding="longest",
            return_tensors="pt"
        )
        enc["labels"] = torch.stack([item["labels"] for item in batch])
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

# ── CSV LOGGER ──────────────────────────────────────────────────────────────
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
    
    data = {}
    for i, v in enumerate(VAR_IDS):
        data[f"{v}_true"] = labels[:, i]
        data[f"{v}_pred"] = preds[:, i]
    
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeller", type=str, required=True, choices=LABELLERS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = BASE_DIR / f"task_A_{args.labeller}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # W&B Init (saving locally to the out_dir to keep things clean)
    run = wandb.init(
        entity="niyimoluga85-universty-of-technology-graz",
        project="56-dims-regression",
        name=f"task_A_{args.labeller}",
        dir=str(out_dir),
        config=vars(args)
    )

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
    
    train_means = train_df[[f"{v}__{args.labeller}" for v in VAR_IDS]].mean().values
    train_means_tensor = torch.tensor(train_means, dtype=torch.float32).to(device)

    collator = DynamicCollator(tokenizer)
    train_loader = DataLoader(DiscourseDataset(train_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)
    val_loader   = DataLoader(DiscourseDataset(val_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
    test_a_loader = DataLoader(DiscourseDataset(test_author_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
    test_t_loader = DataLoader(DiscourseDataset(test_temporal_df, tokenizer, args.labeller), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    model = PlainDeBERTa().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler()

    best_val_loss = float("inf")
    log_history = []
    global_step = 0
    
    # Accumulators for logging
    dim_loss_acc = torch.zeros(56, device=device)
    dim_loss_steps = 0
    grad_norm_acc = 0.0
    grad_norm_count = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            mask = ~torch.isnan(labels)
            safe_labels = torch.nan_to_num(labels, nan=0.0)

            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = F.mse_loss(logits * mask, safe_labels * mask)
                
            scaler.scale(loss).backward()
            
            # Capture pre-clip grad norm
            scaler.unscale_(optimizer)
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            
            # Accumulate for step-level logging
            with torch.no_grad():
                sq_err = (logits - safe_labels)**2 * mask
                dim_loss_acc += sq_err.mean(dim=0)
                dim_loss_steps += 1
                
            grad_norm_acc += pre_clip_norm
            grad_norm_count += 1
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # ── LOGGING ──
            if global_step % LOG_STEPS == 0:
                avg_dim = dim_loss_acc / dim_loss_steps
                avg_grad = grad_norm_acc / grad_norm_count
                
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": avg_grad,
                    "train/true_avg_mse_OVERALL": avg_dim.mean().item(),
                }
                for i, v in enumerate(VAR_IDS):
                    log_entry[f"train/true_avg_mse_{v}"] = avg_dim[i].item()
                    
                wandb.log(log_entry, step=global_step)
                log_history.append(log_entry)
                
                # Save incremental CSV
                pd.DataFrame(log_history).to_csv(out_dir / f"thesis_cache_task_A_{args.labeller}.csv", index=False)
                
                dim_loss_acc.zero_()
                dim_loss_steps = 0
                grad_norm_acc = 0.0
                grad_norm_count = 0

            # ── EVALUATION ──
            if global_step % EVAL_STEPS == 0:
                model.eval()
                val_losses = []
                val_mse_dim = torch.zeros(56, device=device)
                val_ss_tot = torch.zeros(56, device=device)
                val_n = 0
                
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_ids = v_batch["input_ids"].to(device)
                        v_mask = v_batch["attention_mask"].to(device)
                        v_labels = v_batch["labels"].to(device)
                        
                        v_nan_mask = ~torch.isnan(v_labels)
                        v_safe = torch.nan_to_num(v_labels, nan=0.0)
                        
                        v_logits = model(v_ids, v_mask)
                        v_loss = F.mse_loss(v_logits * v_nan_mask, v_safe * v_nan_mask)
                        val_losses.append(v_loss.item())
                        
                        sq_err = (v_logits - v_safe)**2 * v_nan_mask
                        val_mse_dim += sq_err.sum(dim=0)
                        val_ss_tot += ((v_safe - train_means_tensor)**2 * v_nan_mask).sum(dim=0)
                        val_n += v_labels.shape[0]
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                r2_dim = 1.0 - (val_mse_dim / (val_ss_tot + 1e-9))
                avg_r2 = r2_dim.mean().item()
                
                eval_log = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "eval/loss": avg_val_loss,
                    "eval/true_avg_mse_OVERALL": (val_mse_dim / val_n).mean().item(),
                    "eval/r2_OVERALL": avg_r2,
                }
                for i, v in enumerate(VAR_IDS):
                    eval_log[f"eval/true_avg_mse_{v}"] = (val_mse_dim[i] / val_n).item()
                    eval_log[f"eval/r2_{v}"] = r2_dim[i].item()
                    
                wandb.log(eval_log, step=global_step)
                log_history.append(eval_log)
                pd.DataFrame(log_history).to_csv(out_dir / f"thesis_cache_task_A_{args.labeller}.csv", index=False)
                
                print(f"\n  ✅ Eval @ step {global_step}: loss={avg_val_loss:.4f} | R²={avg_r2:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), out_dir / "best_model.pt")
                    print("  -> Saved new best_model.pt")
                    
                model.train()

        # Save epoch checkpoint
        torch.save(model.state_dict(), out_dir / f"checkpoint_epoch_{epoch+1}.pt")

    # Final Test
    print("\nRunning final test evaluation...")
    if os.path.exists(out_dir / "best_model.pt"):
        model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    
    save_predictions_csv(model, test_a_loader, device, out_dir / "test_predictions_author.csv")
    save_predictions_csv(model, test_t_loader, device, out_dir / "test_predictions_temporal.csv")
    
    run.finish()
    print("Done!")

if __name__ == "__main__":
    main()