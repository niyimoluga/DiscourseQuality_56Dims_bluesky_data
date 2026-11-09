# Discourse Quality on Bluesky — 56 Dimensions × 5 Labellers

Code and derived artifacts for modelling **56 discourse-quality dimensions** on a 48-hour Bluesky snapshot, using **five LLM labellers** as annotation sources and **DeBERTa-v3-large** as the text encoder.

The repository compares three levels of modelling:

1. **Classical baselines** — training mean, Bag-of-Words + Ridge, TF-IDF + Ridge.
2. **Task A — independent models** — one DeBERTa-v3-large regressor per labeller.
3. **Task C — unified model** — a single shared encoder with **Progressive Layered Experts (PLE)** and five prediction towers, one per labeller.

The central question is: *can one model emulate all five labellers, or does labeller disagreement make specialised models necessary?*

---

## Highlights

- **467,125** Bluesky posts, **454** columns, **56** discourse dimensions, **5** labellers.
- **451,958** English posts after language filtering.
- Author-disjoint **80/10/10** split (plus a temporal split for robustness checks).
- DeBERTa-v3-large beats lexical baselines by a large margin, showing that context matters.
- The unified PLE model helps the hardest labeller (`qwen3.5`) but shows **negative transfer** on labellers that are easier to model independently — a "seesaw" effect.
- Gate analysis shows **gptoss/gemma4 rely mostly on task-specific experts**, while **granite4.2/mistral3.2 use more shared experts**.

---

## Dataset

The raw dataset is the gated Hugging Face dataset
[`idealab-cs2/bluesky-discourse-quality`](https://huggingface.co/datasets/idealab-cs2/bluesky-discourse-quality).
Access must be requested on Hugging Face; the code authenticates with `token=True`.

| Property | Value |
|---|---|
| Posts | 467,125 |
| Columns | 454 |
| Score columns | 280 = 56 dimensions × 5 labellers |
| Metadata columns | 174 |
| Time window | 2025-06-01 00:00:03 – 2025-06-02 23:59:59 UTC (48 hours) |
| Unique authors (DIDs) | 139,071 |
| English subset | 451,958 (96.8%) |
| Author split (train / val / test) | 359,570 / 46,989 / 45,399 |
| Temporal split (train / val / test) | 361,566 / 45,196 / 45,196 |
| Posts with a parent post | 54.1% (45.9% missing parent) |
| Median input length | 545 characters (~136 tokens) |
| Score levels | 0, 0.25, 0.50, 0.75, 1.00 |
| Mean Krippendorff's α | 0.2497 |

### Labellers

| Labeller | Mean score | Share of zero scores |
|---|---:|---:|
| `qwen3.5` | 0.148 | 72.3% |
| `gemma4` | 0.076 | 82.2% |
| `mistral3.2` | 0.214 | 57.6% |
| `gptoss` | 0.074 | 84.6% |
| `granite4.2` | 0.182 | 68.0% |

The targets are highly zero-inflated and labeller agreement is modest (mean α ≈ 0.25), which makes this a difficult multi-task regression problem.

### The 56 dimensions

The codebook is grouped into five theoretical families:

| Family | Count | Variable IDs |
|---|---:|---|
| Reasoning & argumentation | 17 | `D1a`, `D1b`, `D3+`, `D3−`, `D4+`, `D4−`, `D5+`, `D5−`, `D6+`, `D6−`, `D7+`, `D7−`, `F7+`, `F7−`, `D8`, `D9`, `D11` |
| Conflict & identity | 12 | `A1`, `A2`, `A3+`, `A3−`, `A4`, `A5+`, `A5−`, `A6`, `A11`, `A12`, `A13`, `F9` |
| Evidence & epistemic stance | 7 | `E1+`, `E1−`, `E2`, `E3`, `E4+`, `E4−`, `E7` |
| Interaction & inclusion | 8 | `I1+`, `I1−`, `I3+`, `I3−`, `I4`, `I8`, `I9`, `F13` |
| Democratic preconditions | 12 | `P1`, `P2`, `P8`, `P3`, `P4`, `P5+`, `P5−`, `P6+`, `P6−`, `P7`, `P10`, `P11` |

Every model input is formatted as:

```text
[THREAD ROOT] <root text> [PARENT] <parent text> [TARGET POST] <comment text>
```

with `max_length=512` tokens.

---

## Repository layout

```text
56_Dims/
├── README.md
├── requirements.txt
├── .gitignore
├── 56Dims.py                       # minimal dataset loader / first-50 sample
├── first50.json                    # 50-record sample of the raw dataset
├── docs/
│   └── PROJECT_CONTEXT.md          # design rationale and research narrative
└── codes/
    ├── phase1_analysis.py          # data audit: distributions, alpha, dendrogram, topics
    ├── phase1_artifacts/           # generated tables, plots and phase reports
    ├── phase1_language_flags.parquet
    ├── phase2_split.py             # author-DID + temporal splits
    ├── phase3_baseline.py          # Mean / BoW / TF-IDF baselines
    ├── phase4_train_task_a.py      # Task A (original)
    ├── phase4_train_task_a_update.py
    ├── phase4_aggregate_results.py
    ├── phase5_train_task_c_ple.py  # Task C (original)
    ├── phase5_train_task_c_ple_update.py
    ├── phase5_train_task_c_ple_update1a.py
    ├── phase5_train_task_c_ple_update1b.py   # recommended Task C script
    ├── phase5_eval_only.py         # regenerate test predictions from best_model.pt
    ├── phase6_aggregate_task_c.py  # Task C vs Task A vs baselines
    ├── phase7_final_analysis.py    # final master table
    ├── phase8_gate_analysis.py     # expert-routing / gate analysis
    ├── phase9_error_analysis.py    # best and worst test examples
    ├── splits_author_temporal.parquet
    ├── lang.py
    └── paraquet.py
```

The derived files `phase1_language_flags.parquet` and
`splits_author_temporal.parquet` are included so that the modelling phases can be
run without recomputing the data audit and split construction.

---

## Pipeline

| Phase | Script | Purpose |
|---|---|---|
| 1 | `phase1_analysis.py` | Dataset fingerprint, zero-inflation, per-labeller statistics, Krippendorff's α, correlation dendrogram, author distribution, topic clustering, length/context analysis. |
| 2 | `phase2_split.py` | Builds the author-DID 80/10/10 split and the temporal 80/10/10 split. |
| 3 | `phase3_baseline.py` | Mean predictor, BoW + Ridge, TF-IDF + Ridge for all five labellers. |
| 4 | `phase4_train_task_a_update.py` | Task A: one independent DeBERTa-v3-large model per labeller. |
| 5 | `phase5_train_task_c_ple_update1b.py` | Task C: one shared encoder + PLE layers + five towers. |
| 6 | `phase6_aggregate_task_c.py` | Evaluation of Task C predictions and comparison with Task A / baselines. |
| 7 | `phase7_final_analysis.py` | Builds `FINAL_MASTER_TABLE.csv`. |
| 8 | `phase8_gate_analysis.py` | Measures how much each labeller uses shared vs task-specific experts. |
| 9 | `phase9_error_analysis.py` | Extracts the best/worst test examples and the highest-error dimensions. |

---

## Models

### Task A — independent DeBERTa regressors

For each labeller, a separate model is trained:

```text
DeBERTa-v3-large
    → mean pooling (1024)
    → Linear(1024, 512) → GELU → Dropout(0.1)
    → Linear(512, 256)  → GELU → Dropout(0.1)
    → Linear(256, 56)   → Sigmoid
```

- One model per labeller (five models in total).
- 4 epochs, batch size 16, AdamW, learning rate `2e-5`, cosine schedule.
- Mixed-precision training; NaN-masked MSE loss.
- Evaluated on the author-held-out test split; temporal test predictions are also produced.

### Task C — unified PLE model

A single shared encoder feeds two progressive PLE layers:

- DeBERTa-v3-large, mean-pooled to 1024 dimensions.
- **PLE layer 1:** 4 shared experts + 2 task-specific experts per labeller; expert width 256.
- **PLE layer 2:** same structure on the 256-dimensional expert outputs.
- Task gates route over that task's experts plus the shared experts.
- The shared gate routes over all task experts plus the shared experts.
- Five prediction towers: `Linear(256, 64) → GELU → Linear(64, 56) → Sigmoid`.

Training configuration:

- 4 epochs, batch size 16, AdamW, learning rate `1e-5`, cosine schedule.
- NaN-masked MSE with task weights:
  `qwen3.5=0.5`, `mistral3.2=0.8`, `gemma4=1.5`, `gptoss=1.5`, `granite4.2=1.0`.
- Weights are normalised by the number of labellers.

The weighted-loss variant is intended to stop the noisier/harder labellers from being
drowned out by the easier ones.

---

## Results

Mean R² across the 56 dimensions on the **author-held-out test split**
(from `codes/phase1_artifacts/FINAL_MASTER_TABLE.csv`).
Task A and Task C were each trained for 4 epochs.

| Labeller | Mean | BoW + Ridge | TF-IDF + Ridge | Task A (independent) | Task C (PLE) | PLE lift |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3.5` | ≈0.000 | 0.054 | 0.129 | 0.194 | **0.328** | **+0.135** |
| `gemma4` | ≈0.000 | 0.035 | 0.126 | **0.485** | 0.356 | −0.129 |
| `mistral3.2` | ≈0.000 | 0.040 | 0.115 | 0.261 | **0.283** | **+0.022** |
| `gptoss` | ≈0.000 | 0.001 | 0.081 | **0.366** | 0.244 | −0.122 |
| `granite4.2` | ≈0.000 | 0.115 | 0.191 | **0.468** | 0.426 | −0.042 |

### What the results show

- **Context beats lexical overlap.** TF-IDF is a strong lexical baseline (0.081–0.191 mean R²), but DeBERTa-v3-large improves substantially (0.194–0.485), showing that discourse quality depends on context and argument structure, not just word choice.
- **The unified model is a seesaw.** PLE improves the hardest labeller (`qwen3.5`, +0.135) and slightly improves `mistral3.2` (+0.022), but it degrades the labellers that are easier to model independently (`gemma4`, `gptoss`, `granite4.2`). Shared capacity helps the noisy task but causes negative transfer for the cleaner ones.
- **Specialisation is uneven.** Gate analysis shows that `gptoss` and `gemma4` route almost all of their probability mass to task-specific experts, while `granite4.2` and `mistral3.2` use the shared experts much more heavily.

### Gate specialisation

Average expert routing over the test set (`codes/phase1_artifacts/gate_weights_task_c.csv`):

| Labeller | Shared experts | Task-specific experts |
|---|---:|---:|
| `qwen3.5` | 20.0% | 80.0% |
| `gemma4` | 14.7% | 85.3% |
| `mistral3.2` | 45.1% | 54.9% |
| `gptoss` | 4.5% | 95.5% |
| `granite4.2` | 68.4% | 31.6% |

This supports the conclusion that "one model for all labellers" is not automatically
better: labellers with different label distributions need different amounts of
specialisation.

---

## Reproducing

### 1. Clone and install

```bash
git clone git@github.com:niyimoluga/DiscourseQuality_56Dims_bluesky_data.git
cd DiscourseQuality_56Dims_bluesky_data

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Authenticate to Hugging Face

The dataset is gated. Request access, then log in:

```bash
huggingface-cli login
```

### 3. Run the pipeline

The derived language flags and splits are already included, so the modelling
phases can start directly:

```bash
cd codes

# Optional: regenerate the data audit and splits from scratch
python phase1_analysis.py
python phase2_split.py

# Classical baselines
python phase3_baseline.py

# Task A: one model per labeller
for lab in qwen3.5 gemma4 mistral3.2 gptoss granite4.2; do
  python phase4_train_task_a_update.py --labeller "$lab" --epochs 4
done

# Task C: unified PLE model
python phase5_train_task_c_ple_update1b.py --epochs 4

# Aggregation and analysis
python phase6_aggregate_task_c.py
python phase7_final_analysis.py
python phase8_gate_analysis.py
python phase9_error_analysis.py
```

If training was interrupted, regenerate the test predictions from the best
checkpoint with:

```bash
python phase5_eval_only.py
```

### Output locations

The training and aggregation scripts currently use a hard-coded output root:

```python
BASE_DIR = Path("/data/nfs/larry/runs_time")
```

Edit `BASE_DIR` in the phase 4–9 scripts to point to your own output directory, or
create the same directory layout. Task A writes to `task_A_<labeller>/`; Task C
writes to `task_C_PLE/`.

---

## Practical notes and known issues

- **Transformers v5 + fp16:** newer versions of Transformers load the DeBERTa
  checkpoint in `float16` by default. `GradScaler` cannot unscale fp16 gradients.
  The recommended `phase5_train_task_c_ple_update1b.py` forces the encoder to
  `float32` (`dtype=torch.float32`). Keep this line if you adapt the script.
- **DataLoader file-descriptor limit:** long runs with `num_workers=4` can hit
  `ulimit -n` (often 1024). The recommended script sets
  `torch.multiprocessing.set_sharing_strategy("file_system")` and uses
  `persistent_workers=True`. Also raise the limit before launching, e.g.
  `ulimit -n 65535`.
- **W&B:** the scripts initialise Weights & Biases with a hard-coded entity/project.
  Change the `wandb.init(...)` call, or set `WANDB_MODE=offline` if you do not want
  to log to W&B.
- **Paths:** parquet files are read relative to the `codes/` directory, so run the
  scripts from there.
- **Large files:** model checkpoints (`*.pt`, ~1.7 GB each), W&B runs and the
  91 MB `Deepseekchat.pdf` research-discussion export are intentionally not
  committed. See `docs/PROJECT_CONTEXT.md` for the design rationale from that
  discussion.

---

## Data and license

- **Raw data:** the Bluesky discourse-quality dataset on Hugging Face. Access and
  use are governed by the dataset's own terms.
- **Derived artifacts:** the language flags, author/temporal splits and phase 1
  audit files are included for reproducibility.
- **Code:** no license has been added yet. Please contact the maintainer before
  reusing the code in a commercial or redistributed product.
- **Third-party models:** DeBERTa-v3-large and the five labeller models retain
  their original licenses.

The phase 1 `error_analysis.csv` and `first50.json` files contain samples of
user-generated Bluesky text, including offensive or sensitive content. Handle
them according to the dataset's terms and your institution's research-ethics
guidelines.

---

## Contact

Maintainer: [@niyimoluga](https://github.com/niyimoluga)

If you use this code or the derived splits, please cite the accompanying thesis
or paper when it becomes available.
