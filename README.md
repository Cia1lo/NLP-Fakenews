# fakenes-detect

本项目使用 `uv` 管理 Python 环境和依赖，不使用 `requirements.txt`。

## 环境初始化

```bash
uv sync
```

如需开发工具：

```bash
uv sync --group dev
```

如需使用 PyTorch Geometric 版本的 GAT / GraphSAGE：

```bash
uv sync --extra gnn
```

## 运行方式

后续脚本统一通过 `uv run` 执行，例如：

```bash
uv run python -m model.train --dataset gossipcop
```

也可以使用 `pyproject.toml` 中注册的脚本入口：

```bash
uv run fakenews-train --dataset politifact
uv run fakenews-inspect --dataset gossipcop
uv run fakenews-ablations --suite core
uv run fakenews-size-control --dataset politifact
uv run fakenews-check-device
uv run fakenews-size-buckets --input-dir outputs/ablations_pooling_splits
uv run fakenews-matched-eval --input-dir outputs/ablations_pooling_splits
```

当前实验数据目录：

```text
database/data
```

## 设备加速

检查当前 Python 环境是否可用 CUDA / MPS：

```bash
uv run fakenews-check-device
```

训练默认使用 `--device auto`，优先级为 CUDA -> MPS -> CPU。显式使用 MPS：

```bash
uv run fakenews-train --dataset politifact --device mps
```

如果当前 PyTorch 环境无法创建 MPS tensor，`--device auto` 会回退到 CPU；
`--device mps` 会直接报错并打印诊断信息。项目默认设置
`PYTORCH_ENABLE_MPS_FALLBACK=1`，用于让少数 MPS 不支持的 PyTorch 算子回退到 CPU。

注意：请通过 `uv run ...` 启动实验。直接运行系统或 Anaconda 的 `python`
可能会加载非项目环境里的 PyTorch，导致 MPS 检测失败或回退到 CPU。

## 基线实验

默认训练使用 `bert profile` 两组节点特征，严格读取 `raw/custom_train_idx.npy`、
`raw/custom_val_idx.npy`、`raw/custom_test_idx.npy`：

```bash
uv run fakenews-train --dataset politifact
```

切换图级读出方式：

```bash
uv run fakenews-train --dataset politifact --pooling root
uv run fakenews-train --dataset politifact --pooling mean
```

只用用户/profile 特征做消融：

```bash
uv run fakenews-train --dataset politifact --features profile
```

使用全部已有节点特征：

```bash
uv run fakenews-train --dataset gossipcop --features bert content spacy profile
```

注意：`gossipcop` 全量特征文件较大，建议先用 `bert profile` 跑通流程，再逐步加特征做消融。

## 自动化消融

核心消融套件包含：

- `bert_profile_graph`：默认基线
- `bert_only_graph`：去掉 profile 元数据特征
- `profile_only_graph`：去掉 BERT 内容特征
- `bert_profile_no_graph`：保留节点特征但不做图传播聚合

运行：

```bash
uv run fakenews-ablations --suite core
```

完整特征消融会额外加入 `content`、`spacy` 和全特征组合：

```bash
uv run fakenews-ablations --suite full
```

结果汇总会写到：

```text
outputs/ablations/summary.csv
outputs/ablations/summary.json
```

带控制项的核心实验会额外加入：

- `graph_size_control`：只用图规模统计，不读内容/profile 特征
- `bert_profile_graph_mean_pool`：图传播 + mean pooling
- `bert_profile_graph_root_pool`：图传播 + root-only pooling
- `bert_profile_no_graph_mean_pool`：无图传播 + mean pooling
- `bert_profile_no_graph_root_pool`：无图传播 + root-only pooling

多随机种子运行：

```bash
uv run fakenews-ablations \
  --suite core_controls \
  --seeds 42 43 44 \
  --datasets politifact gossipcop
```

多 seed 时，每次运行会写到：

```text
outputs/ablations/<experiment>/seed_<seed>/<dataset>/metrics.json
```

同时生成逐 seed 汇总和按实验聚合的 mean/std：

```text
outputs/ablations/summary.csv
outputs/ablations/summary.json
outputs/ablations/summary_by_experiment.csv
outputs/ablations/summary_by_experiment.json
```

单独运行图规模控制：

```bash
uv run fakenews-size-control --dataset politifact
```

root/mean pooling 下的 BERT/profile 拆分实验：

```bash
uv run fakenews-ablations \
  --suite pooling_splits \
  --seeds 42 43 44 \
  --datasets politifact gossipcop \
  --epochs 50 \
  --batch-size 32 \
  --hidden-dim 128 \
  --patience 10 \
  --device auto \
  --output-dir outputs/ablations_pooling_splits
```

如果要在同一套 root/mean + BERT/profile 拆分实验里同时补 `size_only`
图规模基线，使用：

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
  --output-dir outputs/ablations_pooling_splits_controls
```

其中 `size_only` 只使用图规模统计特征，不读取 BERT/profile/content/spacy：

```text
node_count
edge_count
log_node_count
log_edge_count
edge_per_node
```

按图规模分桶分析测试集表现：

```bash
uv run fakenews-size-buckets --input-dir outputs/ablations_pooling_splits --split test
```

分桶结果会写到：

```text
outputs/ablations_pooling_splits/test_size_bucket_summary.csv
outputs/ablations_pooling_splits/test_size_bucket_summary_by_experiment.csv
```

更严格的图规模匹配评估会在每个 dataset/experiment/seed 的预测结果内，
按 `log1p(node_count)` 做 real/fake 一对一最近邻匹配，然后只在匹配子集上重算指标：

```bash
uv run fakenews-matched-eval \
  --input-dir outputs/ablations_pooling_splits \
  --split test
```

如需限制最大匹配距离，可加 caliper：

```bash
uv run fakenews-matched-eval \
  --input-dir outputs/ablations_pooling_splits \
  --split test \
  --caliper 0.25
```

匹配评估结果会写到：

```text
outputs/ablations_pooling_splits/test_matched_summary.csv
outputs/ablations_pooling_splits/test_matched_summary_by_experiment.csv
```

鲁棒性评估直接加载已有 checkpoint，不重训模型。它会在测试时评估：

- `clean`
- `edge_drop_0.25`, `edge_drop_0.5`
- `zero_<feature>`
- `noise_<feature>`

例如只分析 `gossipcop` 上 BERT+profile 的 root/mean pooling 模型：

```bash
uv run fakenews-robustness \
  --input-dir outputs/ablations \
  --datasets gossipcop \
  --experiments bert_profile_graph_mean_pool bert_profile_graph_root_pool \
  --split test \
  --device auto \
  --output-dir outputs/robustness_gossipcop
```

结果会写到：

```text
outputs/robustness_gossipcop/test_robustness_summary.csv
outputs/robustness_gossipcop/test_robustness_summary_by_experiment.csv
```

解释性分析基于鲁棒性结果做 occlusion delta 汇总，即比较 clean 与删边、
置零特征、噪声扰动后的指标下降：

```bash
uv run fakenews-explainability \
  --input outputs/robustness_gossipcop/test_robustness_summary.csv \
  --output-dir outputs/robustness_gossipcop
```

结果会写到：

```text
outputs/robustness_gossipcop/explainability_summary.csv
outputs/robustness_gossipcop/explainability_summary_by_experiment.csv
```

## 当前基线结果

默认基线配置：

```text
features = bert profile
graph_layers = 2
hidden_dim = 128
batch_size = 32
early_stopping = val macro-f1, patience 10
```

| 数据集 | best epoch | Val Macro-F1 | Test Accuracy | Test F1 | Test Macro-F1 | Test AUC |
|---|---:|---:|---:|---:|---:|---:|
| `politifact` | 8 | 0.7328 | 0.8438 | 0.9123 | 0.5990 | 0.7946 |
| `gossipcop` | 6 | 0.9797 | 0.9525 | 0.9525 | 0.9525 | 0.9839 |

完整结果文件：

```text
outputs/baselines/summary.csv
outputs/baselines/summary.json
```

## Core 消融结果

核心消融配置：

```text
suite = core
features = bert/profile/bert+profile
graph_layers = 2 或 0
hidden_dim = 128
batch_size = 32
early_stopping = val macro-f1, patience 10
```

| 数据集 | 实验 | 特征 | 图层 | Test Accuracy | Test F1 | Test Macro-F1 | Test AUC |
|---|---|---|---:|---:|---:|---:|---:|
| `politifact` | `bert_only_graph` | `bert` | 2 | 0.8750 | 0.9231 | 0.7949 | 0.9598 |
| `politifact` | `bert_profile_graph` | `bert profile` | 2 | 0.8438 | 0.9123 | 0.5990 | 0.7946 |
| `politifact` | `profile_only_graph` | `profile` | 2 | 0.8125 | 0.8947 | 0.5188 | 0.6317 |
| `politifact` | `bert_profile_no_graph` | `bert profile` | 0 | 0.7500 | 0.8462 | 0.5897 | 0.7076 |
| `gossipcop` | `bert_profile_no_graph` | `bert profile` | 0 | 0.9607 | 0.9601 | 0.9607 | 0.9861 |
| `gossipcop` | `bert_profile_graph` | `bert profile` | 2 | 0.9516 | 0.9517 | 0.9516 | 0.9821 |
| `gossipcop` | `profile_only_graph` | `profile` | 2 | 0.9461 | 0.9442 | 0.9460 | 0.9780 |
| `gossipcop` | `bert_only_graph` | `bert` | 2 | 0.9433 | 0.9437 | 0.9433 | 0.9775 |

结论：

- `politifact` 上最有效的是 `bert_only_graph`，说明当前划分下 BERT 内容特征是主信号；加入 `profile` 后 Macro-F1 和 AUC 明显下降，profile 更像噪声或引入了小样本 split 偏差。
- `gossipcop` 上最有效的是 `bert_profile_no_graph`，说明 BERT+profile 的节点级统计已经很强；当前 GraphSAGE 风格的均值图聚合没有带来增益，反而略降。
- `profile` 在 `gossipcop` 上单独就很强，但在 `politifact` 上弱很多，说明用户元数据的可迁移性和稳定性不足。
- 当前图结构贡献不稳定：同特征下 `politifact` 的图层比无图略好，但 `gossipcop` 无图更好。因此不建议立刻升级到 PyG/GAT；已补充多随机种子、图规模控制、root-only/mean-pooling 对照，用于进一步确认图结构是否真的有效。
- 跨模态注意力应等待真实图像/视频数据接入后再升级；当前数据主要是预计算文本/profile/传播图特征，直接上跨模态注意力收益依据不足。

Core 消融结果文件：

```text
outputs/ablations/summary.csv
outputs/ablations/summary.json
```

## 多随机种子最终结果

最终 `core_controls` 实验配置：

```text
seeds = 42 43 44
datasets = politifact gossipcop
device = auto
```

完整分析见：

```text
outputs/ablations/final_analysis.md
outputs/ablations/summary_by_experiment.csv
```

各数据集最优结果：

| 数据集 | 最优实验 | Test Macro-F1 mean | Std | Test AUC mean |
|---|---|---:|---:|---:|
| `politifact` | `bert_profile_no_graph_root_pool` | 0.8222 | 0.0320 | 0.9754 |
| `gossipcop` | `bert_profile_no_graph_mean_pool` | 0.9625 | 0.0048 | 0.9865 |

最终判断：

- 当前最优模型都不是图传播模型，说明现有 GraphSAGE 风格聚合没有稳定贡献。
- `politifact` 更依赖 root 节点表征；`gossipcop` 更依赖全图节点 mean pooling。
- `gossipcop` 的 `profile_only_graph` 已达到 0.9576 Macro-F1，profile 元数据非常强，但存在数据集偏差风险。
- `gossipcop` 的 `graph_size_control` 达到 0.7457 Macro-F1，必须在论文中作为控制基线报告。
- 暂不建议直接升级 PyG/GAT；应先补 root/mean pooling 下的 BERT/profile 拆分，以及图规模匹配控制。

## 依赖维护规则

- 新增运行依赖：编辑 `pyproject.toml` 的 `[project].dependencies`
- 新增可选依赖：编辑 `pyproject.toml` 的 `[project.optional-dependencies]`
- 新增开发依赖：编辑 `pyproject.toml` 的 `[dependency-groups].dev`
- 不维护 `requirements.txt`
- 环境锁定文件由 `uv` 生成和更新
