# AGENTS.md

## Project Snapshot

Project: `fakeNes_detect`

Goal: build and evaluate a multi-source fake news detection framework using text/node features, user/profile metadata, and propagation graph structure.

Environment policy:

- Use `uv` for environment and command execution.
- Do not add or use `requirements.txt`.
- Main data lives under `database/data/`.
- Large local artifacts are intentionally ignored by Git:
  - `database/data/`
  - `outputs/`
  - `.venv/`
  - `.ruff_cache/`
  - `*.egg-info/`

Current Git history:

```text
35a7ce0 Add robustness and explainability analysis
f40150d Summarize gossipcop main experiments
78066fe Add size baseline and matched evaluation
7c6eb46 Initial project baseline
```

Current working tree status at last handoff: clean.

## Data Status

Analyzed dataset path:

```text
database/data
```

Available datasets:

- `gossipcop`
- `politifact`

Confirmed label mapping:

```text
0 = real
1 = fake
```

Important data observations:

- This is a FakeNewsNet-derived graph dataset.
- Each news item is represented as a propagation graph.
- Graphs are essentially tree-like: most graphs satisfy `edges = nodes - 1`.
- `gossipcop` is the main experimental dataset because it has a much larger and more stable test set.
- `politifact` is retained only as an auxiliary small-sample dataset because its current split is heavily fake-skewed.

Known split issue:

- `politifact` custom test split is very small and fake-heavy.
- Do not use `politifact` as the primary evidence for model selection.

## Implemented Code

Core modules:

- `model/data.py`: raw graph dataset loader and batching.
- `model/models.py`: current GraphSAGE-style mean aggregation model.
- `model/train.py`: graph model training and prediction export.
- `model/train_size_control.py`: `size_only` baseline with logistic regression.
- `model/run_ablations.py`: ablation suite runner.
- `model/analyze_size_buckets.py`: graph-size bucket analysis.
- `model/analyze_matched_eval.py`: real/fake graph-size matched evaluation.
- `model/evaluate_robustness.py`: checkpoint-based robustness evaluation.
- `model/analyze_explainability.py`: occlusion delta explanation summary.
- `model/check_device.py`: CUDA/MPS environment check.

Registered commands:

```bash
uv run fakenews-train
uv run fakenews-inspect
uv run fakenews-ablations
uv run fakenews-size-control
uv run fakenews-size-buckets
uv run fakenews-matched-eval
uv run fakenews-robustness
uv run fakenews-explainability
uv run fakenews-check-device
```

Device status:

- Current project environment supports MPS.
- `--device auto` resolves to MPS on the current machine.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by project device handling.

## Completed Experiment Path

Current non-paper experiment progress:

```text
Data inspection
  -> completed

Baseline training framework
  -> completed

BERT/profile/propagation graph baselines
  -> completed

Multi-seed core ablations
  -> completed

root pooling / mean pooling controls
  -> completed

BERT-only / profile-only split experiments
  -> completed

size_only graph-size baseline
  -> completed

Graph-size bucket control
  -> completed

real/fake graph-size matched evaluation
  -> completed

caliper=0.25 strict matched evaluation
  -> completed

gossipcop main experiment report
  -> completed

Robustness analysis
  -> completed

Occlusion-style explainability analysis
  -> completed
```

Current stage:

```text
Core experiment loop is complete.
The project is now in control-analysis / robustness / explainability refinement.
```

## Main Results: gossipcop

Primary result file:

```text
outputs/ablations_pooling_splits/summary_by_experiment.csv
```

Main `gossipcop` results, averaged over seeds `42 43 44`:

| Experiment | Macro-F1 | AUC | Note |
|---|---:|---:|---|
| `bert_only_graph_mean_pool` | 0.9570 +/- 0.0016 | 0.9880 | best original-test Macro-F1 among BERT/profile split runs |
| `bert_only_graph_root_pool` | 0.9561 +/- 0.0032 | 0.9885 | best original-test AUC |
| `bert_only_no_graph_mean_pool` | 0.9531 +/- 0.0045 | 0.9842 | very strong no-graph mean-pooling baseline |
| `profile_only_graph_root_pool` | 0.9430 +/- 0.0021 | 0.9728 | profile is a strong signal |
| `profile_only_graph_mean_pool` | 0.9412 +/- 0.0027 | 0.9752 | profile remains strong with mean pooling |
| `profile_only_no_graph_mean_pool` | 0.9204 +/- 0.0126 | 0.9684 | profile distribution alone is predictive |
| `profile_only_no_graph_root_pool` | 0.8375 +/- 0.0099 | 0.9117 | root-only profile is weaker |
| `bert_only_no_graph_root_pool` | 0.7555 +/- 0.0093 | 0.8407 | root-only BERT needs graph propagation |
| `size_only` | 0.7457 +/- 0.0000 | 0.7949 | graph-size confound baseline |

Key interpretation:

- BERT is the strongest semantic signal.
- Profile/user metadata is also very strong, but should be treated as a potential bias/control signal.
- Mean pooling without graph propagation is already very strong.
- Graph propagation matters most when the readout is root-only.
- Size-only has non-trivial original-test performance, so graph size must be controlled.

## Graph-Size Control

Bucket analysis output:

```text
outputs/ablations_pooling_splits/test_size_bucket_summary_by_experiment.csv
```

`gossipcop` test split size buckets:

| Bucket | N | Fake Rate | Avg Nodes |
|---|---:|---:|---:|
| small | 365 | 0.838 | 10.9 |
| medium | 364 | 0.382 | 40.7 |
| large | 365 | 0.252 | 101.6 |

Important conclusion:

- Graph size and label are strongly correlated.
- Small graphs are mostly fake.
- Large graphs are more often real.
- Any final claim must include graph-size controls.

Strict matched evaluation output:

```text
outputs/ablations_pooling_splits/matched_caliper_0_25/test_matched_summary_by_experiment.csv
```

Strict matching setup:

```text
match variable = log1p(node_count)
matching = real/fake one-to-one nearest-neighbor
caliper = 0.25
matched pairs = 295
matched samples = 590
SMD before = 1.221
SMD after = 0.148
```

Matched results:

| Experiment | Matched Macro-F1 | Matched AUC |
|---|---:|---:|
| `bert_only_graph_root_pool` | 0.9429 | 0.9822 |
| `bert_only_graph_mean_pool` | 0.9395 | 0.9787 |
| `bert_only_no_graph_mean_pool` | 0.9328 | 0.9697 |
| `profile_only_graph_root_pool` | 0.9322 | 0.9629 |
| `profile_only_graph_mean_pool` | 0.9299 | 0.9661 |
| `profile_only_no_graph_mean_pool` | 0.9027 | 0.9554 |
| `profile_only_no_graph_root_pool` | 0.8067 | 0.8674 |
| `bert_only_no_graph_root_pool` | 0.7439 | 0.8366 |
| `size_only` | 0.5437 | 0.5416 |

Main conclusion:

- After strict graph-size matching, `size_only` drops near random.
- BERT/profile models remain strong.
- The model is not merely exploiting graph size.

## Robustness and Explainability

Robustness output:

```text
outputs/robustness_gossipcop/test_robustness_summary_by_experiment.csv
```

Explainability output:

```text
outputs/robustness_gossipcop/explainability_summary_by_experiment.csv
```

Analyzed checkpoints:

- `bert_profile_graph_mean_pool`
- `bert_profile_graph_root_pool`
- `bert_profile_no_graph_mean_pool`
- `bert_profile_no_graph_root_pool`

Robustness perturbations:

- `edge_drop_0.25`
- `edge_drop_0.5`
- `zero_bert`
- `zero_profile`
- `noise_bert`
- `noise_profile`

Main robustness conclusions:

- Small Gaussian noise on BERT/profile features has almost no effect.
- Mean pooling is nearly insensitive to edge deletion.
- Root pooling is sensitive to edge deletion because root readout depends on graph propagation.
- Occluding BERT or profile causes large drops, so the combined model strongly depends on both signals.

Representative results:

| Experiment | Clean Macro-F1 | Key Perturbation | Perturbed Macro-F1 |
|---|---:|---|---:|
| `bert_profile_graph_mean_pool` | 0.9607 | `zero_profile` | 0.3292 |
| `bert_profile_graph_mean_pool` | 0.9607 | `zero_bert` | 0.3435 |
| `bert_profile_graph_root_pool` | 0.9476 | `edge_drop_0.5` | 0.9219 |
| `bert_profile_no_graph_mean_pool` | 0.9625 | `noise_bert` | 0.9625 |
| `bert_profile_no_graph_root_pool` | 0.8540 | `zero_profile` | 0.3292 |

Interpretation warning:

- `zero_bert` and `zero_profile` are strong out-of-distribution occlusions.
- Use them as explanation/importance probes, not as ordinary deployment robustness estimates.

## Model Upgrade Decision

Current decision:

```text
Do not make GAT the next main line.
Do not enter cross-modal attention yet.
Keep current model and focus on control, robustness, and explainability.
```

Why not GAT as main line:

- Existing mean aggregation already captures the main graph propagation benefit.
- In mean pooling, explicit graph propagation adds only a small gain over no-graph mean pooling.
- Root pooling benefits strongly from graph propagation, but current simple graph propagation already solves most of that gap.
- The graph structure is mostly tree-like, where mean aggregation is a strong baseline.
- GAT can still be added later as a supplementary SOTA comparison, not as the primary path.

Why not cross-modal attention yet:

- Current data pipeline uses precomputed BERT/profile/propagation features.
- No real image/video/CLIP/Swin features are currently integrated.
- Cross-modal attention would add complexity without true multimodal evidence.

## Current Recommended Direction

Continue with:

```text
BERT/profile + propagation graph + graph-size control + robustness + explainability
```

Immediate next useful experiments:

1. Missing-modality training robustness
   - Train with modality dropout.
   - Compare against current test-time occlusion.

2. Error case analysis
   - Analyze false positives and false negatives.
   - Group by size bucket, node count, prediction confidence, and occlusion delta.

3. Profile bias audit
   - Inspect high-confidence profile-only samples.
   - Check correlation between profile signal, graph size, label, and propagation depth.

4. Optional GAT comparison only if needed
   - Dataset: `gossipcop`
   - Features: BERT-only
   - Pooling: root
   - Evaluation: original test + caliper matched evaluation
   - Keep only if it beats `bert_only_graph_root_pool` under matched evaluation.

## Useful Commands

Run the main root/mean + BERT/profile split suite with size-only control:

```bash
uv run fakenews-ablations \
  --suite pooling_splits_controls \
  --seeds 42 43 44 \
  --datasets politifact gossipcop \
  --epochs 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --patience 10 \
  --device auto \
  --output-dir outputs/ablations_pooling_splits \
  --skip-existing
```

Run graph-size bucket analysis:

```bash
uv run fakenews-size-buckets \
  --input-dir outputs/ablations_pooling_splits \
  --split test
```

Run strict matched evaluation:

```bash
uv run fakenews-matched-eval \
  --input-dir outputs/ablations_pooling_splits \
  --split test \
  --caliper 0.25 \
  --output-dir outputs/ablations_pooling_splits/matched_caliper_0_25
```

Run robustness analysis:

```bash
uv run fakenews-robustness \
  --input-dir outputs/ablations \
  --datasets gossipcop \
  --experiments bert_profile_graph_mean_pool bert_profile_graph_root_pool bert_profile_no_graph_mean_pool bert_profile_no_graph_root_pool \
  --split test \
  --device auto \
  --batch-size 64 \
  --output-dir outputs/robustness_gossipcop
```

Run explainability summary:

```bash
uv run fakenews-explainability \
  --input outputs/robustness_gossipcop/test_robustness_summary.csv \
  --output-dir outputs/robustness_gossipcop
```

Run checks:

```bash
uv run python -m compileall model
uv run ruff check model pyproject.toml
```

## Tracked Reports

Tracked project reports:

- `docs/gossipcop_main_experiment_report.md`
- `docs/gossipcop_robustness_explainability_report.md`

`outputs/` contains generated experiment artifacts and is intentionally ignored by Git.
