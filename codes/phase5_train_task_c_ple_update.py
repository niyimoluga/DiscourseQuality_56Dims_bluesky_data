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
BASE_DIR = Path("/data/nfs/larry/runs_time/task_C_PLE")
SEED = 42

LOG_STEPS = 50
EVAL_STEPS = 500

def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ── DATASET & COLLATOR ──────────────────────────────────────────────────────
class DiscourseDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = (
            "[THREAD ROOT] " + df['root_text'].fillna("") + 
            " [PARENT] " + df['parent_text'].fillna("") + 
            " [TARGET POST] " + df['comment_text']
        ).tolist()
        
        # Store targets for all 5 labellers
        self.targets = {}
        for lab in LABELLERS:
            self.targets[lab] = df[[f"{v}__{lab}" for v in VAR_IDS]].values
            
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=512)
        
        item = {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
        }
        # Add targets for each labeller
        for lab in LABELLERS:
            item[f"labels_{lab}"] = torch.tensor(self.targets[lab][idx], dtype=torch.float32)
            
        return item

class DynamicCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        
        enc = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_masks},
            padding="longest",
            return_tensors="pt"
        )
        
        # Stack labels for each labeller
        for lab in LABELLERS:
            enc[f"labels_{lab}"] = torch.stack([item[f"labels_{lab}"] for item in batch])
            
        return enc

# ── PLE ARCHITECTURE ────────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, query_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(query_dim, num_experts)
    def forward(self, query, expert_outputs):
        # query: [B, query_dim], expert_outputs: [B, num_experts, hidden_dim]
        scores = torch.softmax(self.gate(query), dim=-1)  # [B, num_experts]
        scores = scores.unsqueeze(1)                      # [B, 1, num_experts]
        # Weighted sum of experts
        return torch.bmm(scores, expert_outputs).squeeze(1) # [B, hidden_dim]

class PLELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_shared_experts=4, n_task_experts=2):
        super().__init__()
        self.n_tasks = len(LABELLERS)
        self.n_shared = n_shared_experts
        self.n_task = n_task_experts
        
        # Experts
        self.shared_experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(n_shared_experts)])
        self.task_experts = nn.ModuleList([
            nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(n_task_experts)])
            for _ in range(self.n_tasks)
        ])
        
        # Gates
        # Task gates look at: own task experts + shared experts
        self.task_gates = nn.ModuleList([
            GatingNetwork(input_dim, n_task_experts + n_shared_experts) 
            for _ in range(self.n_tasks)
        ])
        
        # Shared gate looks at: all task experts + shared experts
        self.shared_gate = GatingNetwork(input_dim, (self.n_tasks * n_task_experts) + n_shared_experts)

    def forward(self, task_inputs, shared_input):
        # task_inputs: list of 5 tensors [B, input_dim]
        # shared_input: tensor [B, input_dim]
        
        # 1. Compute all expert outputs based on the shared input for this layer
        shared_out = torch.stack([e(shared_input) for e in self.shared_experts], dim=1) # [B, n_shared, hidden_dim]
        
        task_outs = []
        for i in range(self.n_tasks):
            t_out = torch.stack([e(shared_input) for e in self.task_experts[i]], dim=1) # [B, n_task, hidden_dim]
            task_outs.append(t_out)
            
        # 2. Route Task Outputs
        new_task_outputs = []
        for i in range(self.n_tasks):
            # Gate sees: own task experts + shared experts
            available_experts = torch.cat([task_outs[i], shared_out], dim=1) # [B, n_task + n_shared, hidden_dim]
            routed = self.task_gates[i](task_inputs[i], available_experts)
            new_task_outputs.append(routed)
            
        # 3. Route Shared Output
        # Gate sees: all task experts + shared experts
        all_experts = torch.cat(task_outs + [shared_out], dim=1) # [B, (n_tasks * n_task) + n_shared, hidden_dim]
        new_shared_output = self.shared_gate(shared_input, all_experts)
        
        return new_task_outputs, new_shared_output

class PLEDeBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BACKBONE)
        emb_dim = 1024
        expert_dim = 256
        
        # 2 Progressive Layers
        self.ple_layer1 = PLELayer(input_dim=emb_dim, hidden_dim=expert_dim)
        self.ple_layer2 = PLELayer(input_dim=expert_dim, hidden_dim=expert_dim)
        
        # 5 separate prediction towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.GELU(),
                nn.Linear(64, 56),
                nn.Sigmoid()
            ) for _ in range(len(LABELLERS))
        ])

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean Pooling over sequence
        mask = attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        pooled = torch.sum(out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        
        # PLE Layer 1
        # Initially, task inputs are the same as the shared input (the pooled embedding)
        initial_tasks = [pooled] * len(LABELLERS)
        t_out1, s_out1 = self.ple_layer1(initial_tasks, pooled)
        
        # PLE Layer 2
        t_out2, s_out2 = self.ple_layer2(t_out1, s_out1)
        
        # Towers
        predictions = {}
        for i, lab in enumerate(LABELLERS):
            predictions[lab] = self.towers[i](t_out2[i])
            
        return predictions

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────
def compute_loss(predictions, batch, device):
    total_loss = 0.0
    sq_errs = {}
    
    for lab in LABELLERS:
        labels = batch[f"labels_{lab}"].to(device)
        preds = predictions[lab]
        
        mask = ~torch.isnan(labels)
        safe_labels = torch.nan_to_num(labels, nan=0.0)
        
        err = (preds - safe_labels)**2 * mask
        loss = err.sum() / (mask.sum() + 1e-8)
        
        total_loss += loss / len(LABELLERS) # <--- AVERAGE HERE
        sq_errs[lab] = err
        
    return total_loss, sq_errs

def save_predictions_csv(model, dataloader, device, out_path):
    model.eval()
    all_preds = {lab: [] for lab in LABELLERS}
    all_labels = {lab: [] for lab in LABELLERS}
    
    with torch.no_grad():
        for batch in dataloader:
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
            
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)# Was 2e-5
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(BASE_DIR / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    run = wandb.init(
        entity="niyimoluga85-universty-of-technology-graz",
        project="56-dims-regression",
        name="task_C_PLE",
        dir=str(BASE_DIR),
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
    test_author_df = df[df['split_author'] == 'test']

    tokenizer = DebertaV2Tokenizer.from_pretrained(BACKBONE)
    
    # We need training means for R2 for ALL labellers
    train_means = {}
    for lab in LABELLERS:
        train_means[lab] = torch.tensor(
            train_df[[f"{v}__{lab}" for v in VAR_IDS]].mean().values, 
            dtype=torch.float32
        ).to(device)

    collator = DynamicCollator(tokenizer)
    train_loader = DataLoader(DiscourseDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4)
    val_loader   = DataLoader(DiscourseDataset(val_df, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)
    test_loader  = DataLoader(DiscourseDataset(test_author_df, tokenizer), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=4)

    model = PLEDeBERTa().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = GradScaler()

    best_val_loss = float("inf")
    log_history = []
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            optimizer.zero_grad()
            with autocast():
                preds_dict = model(input_ids, attention_mask)
                loss, _ = compute_loss(preds_dict, batch, device)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0) # <--- Was 1.0
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % LOG_STEPS == 0:
                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0]
                }
                wandb.log(log_entry, step=global_step)
                log_history.append(log_entry)
                pd.DataFrame(log_history).to_csv(BASE_DIR / "thesis_cache_task_C_PLE.csv", index=False)

            if global_step % EVAL_STEPS == 0:
                model.eval()
                val_losses = []
                val_mse_dict = {lab: torch.zeros(56, device=device) for lab in LABELLERS}
                val_ss_tot_dict = {lab: torch.zeros(56, device=device) for lab in LABELLERS}
                val_n = 0
                
                with torch.no_grad():
                    for v_batch in val_loader:
                        v_ids = v_batch["input_ids"].to(device)
                        v_mask = v_batch["attention_mask"].to(device)
                        
                        v_preds = model(v_ids, v_mask)
                        v_loss, v_sq_errs = compute_loss(v_preds, v_batch, device)
                        val_losses.append(v_loss.item())
                        
                        for lab in LABELLERS:
                            v_labels = v_batch[f"labels_{lab}"].to(device)
                            v_nan_mask = ~torch.isnan(v_labels)
                            v_safe = torch.nan_to_num(v_labels, nan=0.0)
                            
                            val_mse_dict[lab] += (v_sq_errs[lab]).sum(dim=0)
                            val_ss_tot_dict[lab] += ((v_safe - train_means[lab])**2 * v_nan_mask).sum(dim=0)
                            
                        val_n += v_ids.shape[0]
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                
                eval_log = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "eval/loss": avg_val_loss,
                }
                
                r2_overall_list = []
                for lab in LABELLERS:
                    r2_dim = 1.0 - (val_mse_dict[lab] / (val_ss_tot_dict[lab] + 1e-9))
                    avg_r2 = r2_dim.mean().item()
                    r2_overall_list.append(avg_r2)
                    
                    eval_log[f"eval/r2_{lab}"] = avg_r2
                    eval_log[f"eval/mse_{lab}"] = (val_mse_dict[lab] / val_n).mean().item()
                    
                eval_log["eval/r2_OVERALL"] = np.mean(r2_overall_list)
                
                wandb.log(eval_log, step=global_step)
                log_history.append(eval_log)
                pd.DataFrame(log_history).to_csv(BASE_DIR / "thesis_cache_task_C_PLE.csv", index=False)
                
                print(f"\n  ✅ Eval @ step {global_step}: Val Loss={avg_val_loss:.4f}")
                for lab in LABELLERS:
                    print(f"     {lab} R²: {eval_log[f'eval/r2_{lab}']:.4f}")
                print(f"     OVERALL MEAN R²: {eval_log['eval/r2_OVERALL']:.4f}\n")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), BASE_DIR / "best_model.pt")
                    print("  -> Saved new best_model.pt")
                    
                model.train()

        torch.save(model.state_dict(), BASE_DIR / f"checkpoint_epoch_{epoch+1}.pt")

    print("\nRunning final test evaluation...")
    if os.path.exists(BASE_DIR / "best_model.pt"):
        model.load_state_dict(torch.load(BASE_DIR / "best_model.pt"))
    
    save_predictions_csv(model, test_loader, device, BASE_DIR / "test_predictions_author.csv")
    
    run.finish()
    print("Done!")

if __name__ == "__main__":
    main()