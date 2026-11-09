# Phase 1 — Bluesky Discourse Quality Data Analysis

Generated on 2026-10-15T08:54:45.376746


## 1.1 Dataset Fingerprint

- Rows: **467,125** (expected 467,125)
- Columns: **454**
- Null `comment_id`: 0
- Null `comment_text`: 0
- Null `parent_text`: 214540 (45.93%)
- Time span: **48.0 hours**
- Missing score columns: 0

## 1.2 Per-Variable Zero-Inflation

- Sparse variables (>95% all-zero): **0**
- Active variables (<80% all-zero): **50**

### Top-15 most active variables (lowest all-zero rate)

| variable   | group       |   pct_all_zero |   mean_score |
|:-----------|:------------|---------------:|-------------:|
| I1+        | Interaction |       0.753546 |     0.445291 |
| I3+        | Interaction |       0.933797 |     0.618397 |
| I4         | Interaction |       2.11185  |     0.33138  |
| F9         | Conflict    |       2.2722   |     0.444836 |
| D1a        | Reasoning   |       2.32636  |     0.335518 |
| I1−        | Interaction |       3.30511  |     0.286256 |
| D1b        | Reasoning   |       3.98609  |     0.299604 |
| E1+        | Epistemic   |       4.80343  |     0.323439 |
| E2         | Epistemic   |      13.9188   |     0.199949 |
| A1         | Conflict    |      14.8491   |     0.262165 |
| I8         | Interaction |      15.8326   |     0.183692 |
| A2         | Conflict    |      17.0847   |     0.272027 |
| E1−        | Epistemic   |      19.4847   |     0.160049 |
| A4         | Conflict    |      20.5566   |     0.355992 |
| D7+        | Reasoning   |      23.9433   |     0.150213 |

### Top-15 sparsest variables

| variable   | group         |   pct_all_zero |   mean_score |
|:-----------|:--------------|---------------:|-------------:|
| P7         | Preconditions |        91.9613 |    0.0105365 |
| P5−        | Preconditions |        86.6881 |    0.0196373 |
| P3         | Preconditions |        84.2541 |    0.023864  |
| P8         | Preconditions |        82.6971 |    0.022758  |
| P1         | Preconditions |        82.4182 |    0.0242165 |
| I9         | Interaction   |        81.7922 |    0.0193125 |
| P6−        | Preconditions |        77.8738 |    0.0418551 |
| P6+        | Preconditions |        75.3417 |    0.0290242 |
| P2         | Preconditions |        74.7237 |    0.0288505 |
| A13        | Conflict      |        73.0128 |    0.0429369 |
| P5+        | Preconditions |        69.9063 |    0.0362918 |
| A6         | Conflict      |        69.3602 |    0.051893  |
| D11        | Reasoning     |        68.7383 |    0.0389288 |
| E3         | Epistemic     |        68.3978 |    0.0397539 |
| A3−        | Conflict      |        67.9491 |    0.0644734 |

## 1.3 Per-Labeller Statistics

| labeller   |   n_cells |   n_nan |   pct_nan |   mean_score |   pct_zero |   pct_0.25 |   pct_0.50 |   pct_0.75 |   pct_1.00 |
|:-----------|----------:|--------:|----------:|-------------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| qwen3.5    |  26157387 |       0 |         0 |     0.147954 |    72.3112 |    8.17272 |   10.2629  |    6.52938 |   2.72378  |
| gemma4     |  26158951 |       0 |         0 |     0.076408 |    82.1586 |    8.80275 |    5.90232 |    2.58945 |   0.546868 |
| mistral3.2 |  26156379 |       0 |         0 |     0.213713 |    57.587  |   17.194   |   10.859   |   10.8669  |   3.4931   |
| gptoss     |  26098683 |       0 |         0 |     0.074102 |    84.5854 |    5.51609 |    6.91467 |    1.63999 |   1.34386  |
| granite4.2 |  26156971 |       0 |         0 |     0.182015 |    68.0198 |    5.88776 |   15.4679  |    6.5155  |   4.10904  |

## 1.4 Krippendorff's Alpha

- Mean α: **0.2497**
- Variables with α ≥ 0.4: **5**
- Variables with α < 0.2: **22**

### Top-10 highest-α variables

| variable   | group       | polarity   |   krippendorff_alpha |   n_items |
|:-----------|:------------|:-----------|---------------------:|----------:|
| A4         | Conflict    | positive   |             0.616845 |    456290 |
| D8         | Reasoning   | positive   |             0.581348 |    467000 |
| F9         | Conflict    | positive   |             0.481658 |    467025 |
| D5−        | Reasoning   | negative   |             0.449181 |    466634 |
| A2         | Conflict    | positive   |             0.442479 |    466325 |
| E1+        | Epistemic   | positive   |             0.395079 |    465841 |
| D3−        | Reasoning   | negative   |             0.336758 |    464297 |
| I3+        | Interaction | positive   |             0.331797 |    464429 |
| D4−        | Reasoning   | negative   |             0.327276 |    466504 |
| F7+        | Reasoning   | positive   |             0.324657 |    467071 |

### Top-10 lowest-α variables

| variable   | group         | polarity   |   krippendorff_alpha |   n_items |
|:-----------|:--------------|:-----------|---------------------:|----------:|
| P7         | Preconditions | positive   |            0.142625  |    466965 |
| E3         | Epistemic     | positive   |            0.131853  |    467066 |
| P8         | Preconditions | positive   |            0.127546  |    467025 |
| D11        | Reasoning     | positive   |            0.123232  |    467013 |
| P1         | Preconditions | positive   |            0.119144  |    466927 |
| P6+        | Preconditions | positive   |            0.117429  |    467042 |
| A5+        | Conflict      | positive   |            0.109292  |    466956 |
| A3+        | Conflict      | positive   |            0.0926748 |    466975 |
| P2         | Preconditions | positive   |            0.0791062 |    467074 |
| I9         | Interaction   | positive   |            0.0717508 |    467044 |

## 1.5 Correlation Dendrogram

- Cluster A size: **22**
- Cluster B size: **34**
- See `05_dendrogram_*.png` for visualizations
- See `05_macro_groups_vs_theoretical.csv` for crosstab

## 1.6 Author Distribution

- n_unique_dids: 139071
- comments_per_did_min: 1
- comments_per_did_25: 1.0000
- comments_per_did_50: 1.0000
- comments_per_did_75: 3.0000
- comments_per_did_95: 11.0000
- comments_per_did_max: 3050
- pct_dids_with_1_comment: 50.1931
- pct_comments_from_top_1pct_dids: 15.7819

## 1.7 Topic Clustering

- Found **6** clusters on a 20k sample
  - Cluster 0 (n=101): que; la; en; el; se; não; los; para; es; por; um; lo; una; mas; mais
  - Cluster 1 (n=142): er; je; og; en; det; som; van; na; een; dat; på; ik; het; har; se
  - Cluster 2 (n=105): trans; pride; people; just; month; like; queer; trans people; pride month; women; gender; happy; woman; lgbtq; time
  - Cluster 3 (n=101): social; bsky social; bsky; com; thank; swifties; swifties social; post; doing; know; org; love; art; gov; bsky app
  - Cluster 4 (n=514): israel; people; jews; antisemitism; genocide; jewish; gaza; palestinians; just; like; hamas; don; palestine; zionism; an
  - Cluster 5 (n=250): ukraine; russia; putin; trump; musk; russian; war; like; drones; just; people; military; russians; drone; time

## 1.8 Length Distributions

### comment_chars
  - p50: 234.0
  - p75: 284.0
  - p90: 297.0
  - p95: 299.0
  - p99: 300.0
  - max: 1189.0
### root_chars
  - p50: 203.0
  - p75: 279.0
  - p90: 297.0
  - p95: 299.0
  - p99: 300.0
  - max: 777.0
### parent_chars
  - p50: 196.0
  - p75: 277.0
  - p90: 296.0
  - p95: 299.0
  - p99: 300.0
  - max: 468.0
- See `08_length_distributions.png`

## 1.9 Context Availability

- n_rows: 467125
- pct_has_parent: 54.0723
- pct_has_root: 100.0000
- pct_has_both: 54.0723
- pct_missing_parent: 45.9277
- full_input_chars_p50: 545.0000
- full_input_chars_p95: 880.0000
- full_input_chars_p99: 927.0000
- full_input_chars_max: 1445.0000
- full_input_tokens_p50_approx: 136.2500
- full_input_tokens_p95_approx: 220.0000
- full_input_tokens_p99_approx: 231.7500
- pct_exceed_512_tokens_approx: 0.0000
- pct_exceed_384_tokens_approx: 0.0000
- pct_exceed_256_tokens_approx: 0.0233

## 1.10 Missing Score Cells

| labeller   |   total_cells |   nan_cells |     pct_nan |   rows_with_any_nan |   pct_rows_with_any_nan |
|:-----------|--------------:|------------:|------------:|--------------------:|------------------------:|
| qwen3.5    |      26159000 |        1613 | 0.00616614  |                  31 |             0.00663634  |
| gemma4     |      26159000 |          49 | 0.000187316 |                   1 |             0.000214075 |
| mistral3.2 |      26159000 |        2621 | 0.0100195   |                  71 |             0.0151994   |
| gptoss     |      26159000 |       60317 | 0.230578    |               49641 |            10.6269      |
| granite4.2 |      26159000 |        2029 | 0.00775641  |                  45 |             0.0096334   |

## Recommendations for Downstream Phases

- **max_length**: see §1.9 for p95/p99 token estimates
- **Sparse variables to exclude**: see §1.2 top-15 sparsest
- **Split strategy**: see §1.6 — if median comments/DID is low and top-1% is not dominant, random split ≈ author-DID split; otherwise author-DID is essential.
- **Task C routing**: if §1.5 dendrogram shows clean macro-groups, consider PLE; otherwise use MMoE over labellers.