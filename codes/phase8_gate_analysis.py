import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from datasets import load_dataset

# Import your PLE model classes
from phase5_train_task_c_ple_update import PLEDeBERTa, DiscourseDataset, DynamicCollator, LABELLERS

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = Path("/data/nfs/larry/runs_time/task_C_PLE")
    OUT_DIR = Path("./phase1_artifacts")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("Loading data for gate analysis...")
    ds = load_dataset("idealab-cs2/bluesky-discourse-quality", data_files="bluesky-discourse-quality.parquet", token=True)
    df = ds["train"].to_pandas()
    splits = pd.read_parquet("splits_author_temporal.parquet")
    lang_flags = pd.read_parquet("phase1_language_flags.parquet")
    df = df.merge(splits, on="comment_id").merge(lang_flags, on="comment_id")
    df = df[df["lang"] == "en"].copy()
    
    # Use 1000 test samples for a robust average
    test_df = df[df['split_author'] == 'test'].head(1000)
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    collator = DynamicCollator(tokenizer)
    dataloader = DataLoader(DiscourseDataset(test_df, tokenizer), batch_size=16, collate_fn=collator, num_workers=0)

    print("Loading best PLE model...")
    model = PLEDeBERTa().to(device)
    model.load_state_dict(torch.load(BASE_DIR / "best_model.pt"))
    model.eval()

    gate_stats = {lab: {"L1": [], "L2": []} for lab in LABELLERS}
    hooks = []
    
    def get_hook(lab_name, layer_name):
        def hook(module, input, output):
            scores = torch.softmax(output, dim=-1)
            mean_weights = scores.mean(dim=0).detach().cpu().numpy()
            gate_stats[lab_name][layer_name].append(mean_weights)
        return hook

    # Attach hooks to the .gate Linear layer inside the GatingNetwork
    for i, lab in enumerate(LABELLERS):
        hooks.append(model.ple_layer1.task_gates[i].gate.register_forward_hook(get_hook(lab, "L1")))
        hooks.append(model.ple_layer2.task_gates[i].gate.register_forward_hook(get_hook(lab, "L2")))

    print("Running forward passes to extract gate weights...")
    with torch.no_grad():
        for batch in dataloader:
            model(batch["input_ids"].to(device), batch["attention_mask"].to(device))

    print("\n=== AVERAGE GATE WEIGHTS (Layer 2) ===")
    print("How much does each labeller rely on Shared vs Task-Specific experts?")
    
    results = []
    for lab in LABELLERS:
        l2_weights = np.mean(gate_stats[lab]["L2"], axis=0)
        
        # In PLE Layer 2, the task gate looks at: [Task Experts (n=2), Shared Experts (n=4)]
        task_specific_pct = np.sum(l2_weights[:2]) * 100
        shared_pct = np.sum(l2_weights[2:]) * 100
        
        print(f"{lab:10s} | Shared Experts: {shared_pct:5.1f}% | Task-Specific Experts: {task_specific_pct:5.1f}%")
        
        results.append({
            "Labeller": lab,
            "Shared_Expert_Pct": shared_pct,
            "Task_Specific_Expert_Pct": task_specific_pct
        })

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Save to CSV
    results_df = pd.DataFrame(results)
    csv_path = OUT_DIR / "gate_weights_task_c.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved gate weights to {csv_path}")

if __name__ == "__main__":
    main()