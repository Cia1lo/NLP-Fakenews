# Core Controls Ablation Analysis

实验配置：

```text
suite = core_controls
datasets = politifact, gossipcop
seeds = 42, 43, 44
epochs = 50
batch_size = 32
hidden_dim = 128
patience = 10
device = auto
```

结果文件：

```text
outputs/ablations/summary.csv
outputs/ablations/summary_by_experiment.csv
outputs/ablations/summary.json
outputs/ablations/summary_by_experiment.json
```

## 完整性

- 逐 seed 结果：54 条，等于 `9 experiments * 3 seeds * 2 datasets`
- 聚合结果：18 条，等于 `9 experiments * 2 datasets`
- 主要判断指标：`test_macro_f1_mean`，同时参考 `test_auc_mean`

## Politifact

按 Test Macro-F1 均值排序：

| 实验 | Test Macro-F1 | Std | Test AUC | Test Accuracy |
|---|---:|---:|---:|---:|
| `bert_profile_no_graph_root_pool` | 0.8222 | 0.0320 | 0.9754 | 0.9115 |
| `bert_profile_graph_root_pool` | 0.7788 | 0.0366 | 0.9539 | 0.8802 |
| `bert_only_graph` | 0.6909 | 0.1944 | 0.8497 | 0.8906 |
| `bert_profile_graph` | 0.6302 | 0.1198 | 0.7820 | 0.8542 |
| `profile_only_graph` | 0.5850 | 0.0235 | 0.5030 | 0.8125 |
| `bert_profile_graph_mean_pool` | 0.5527 | 0.0415 | 0.5283 | 0.8333 |
| `bert_profile_no_graph` | 0.5248 | 0.0428 | 0.5967 | 0.7448 |
| `graph_size_control` | 0.4868 | 0.0000 | 0.6797 | 0.5781 |
| `bert_profile_no_graph_mean_pool` | 0.4787 | 0.0698 | 0.3772 | 0.7865 |

结论：

- 最强配置是 `bert_profile_no_graph_root_pool`，说明 politifact 当前 split 下 root 节点表征非常关键。
- `bert_profile_graph_root_pool` 比无图 root 低 0.0434 Macro-F1，图传播没有提升最强配置。
- 在 `mean_max` pooling 下，图传播比无图高 0.1054 Macro-F1，但仍明显低于 root-only 无图。
- `bert_only_graph` 均值可用，但标准差高达 0.1944，politifact 小样本 split 的不稳定性很强。
- `graph_size_control` Macro-F1 只有 0.4868，说明 politifact 上单靠图规模不能解释最佳模型表现。

## Gossipcop

按 Test Macro-F1 均值排序：

| 实验 | Test Macro-F1 | Std | Test AUC | Test Accuracy |
|---|---:|---:|---:|---:|
| `bert_profile_no_graph_mean_pool` | 0.9625 | 0.0048 | 0.9865 | 0.9625 |
| `bert_profile_graph_mean_pool` | 0.9607 | 0.0040 | 0.9856 | 0.9607 |
| `profile_only_graph` | 0.9576 | 0.0078 | 0.9797 | 0.9576 |
| `bert_profile_graph_root_pool` | 0.9476 | 0.0074 | 0.9782 | 0.9476 |
| `bert_profile_graph` | 0.9461 | 0.0138 | 0.9821 | 0.9461 |
| `bert_profile_no_graph` | 0.9387 | 0.0096 | 0.9777 | 0.9388 |
| `bert_only_graph` | 0.9345 | 0.0061 | 0.9749 | 0.9345 |
| `bert_profile_no_graph_root_pool` | 0.8540 | 0.0104 | 0.9335 | 0.8541 |
| `graph_size_control` | 0.7457 | 0.0000 | 0.7949 | 0.7459 |

结论：

- 最强配置是 `bert_profile_no_graph_mean_pool`，说明 gossipcop 上节点表征的平均池化已经非常强。
- `bert_profile_graph_mean_pool` 比无图 mean pooling 低 0.0018 Macro-F1，图传播没有带来有效增益。
- `profile_only_graph` 达到 0.9576 Macro-F1，profile 元数据在 gossipcop 上非常强，甚至强于 `bert_only_graph` 0.0231。
- `graph_size_control` 达到 0.7457 Macro-F1，说明 gossipcop 存在明显图规模偏差；任何图结构贡献都必须和 size-only baseline 对比。
- root-only 无图很差，加入图传播后提升 0.0936 Macro-F1，但仍低于 mean pooling 系列。

## 总体判断

### 特征有效性

- BERT 内容特征是稳定基础信号，尤其在 politifact 的 core 实验中仍强于 profile-only。
- Profile 元数据在 gossipcop 上非常强，但跨数据集不稳定。它可能包含平台或账号群体偏差，不能简单当作通用语义能力。
- BERT+profile 的效果依赖 pooling。不是简单拼接越多越好，读出方式会显著影响结论。

### 图结构有效性

当前图传播没有稳定证明有效：

- Politifact 最优模型是无图 root pooling。
- Gossipcop 最优模型是无图 mean pooling。
- 图传播只在某些较弱读出设置上有提升，例如 politifact 的 mean_max、gossipcop 的 root pooling。
- 因此当前证据不支持直接升级 PyG/GAT 作为主线。

### 图规模偏差

- Politifact size-only 接近随机 Macro-F1，但 AUC 有 0.6797，存在弱规模信号。
- Gossipcop size-only 已达 0.7457 Macro-F1，规模偏差明显。
- 后续论文必须报告 size-only control，否则图模型收益容易被质疑为图规模偏差。

## 建议

下一步不建议立即升级 PyG/GAT/跨模态注意力。更优先的是：

1. 增加 root pooling 下的特征拆分实验：
   - `bert_only_no_graph_root_pool`
   - `bert_only_graph_root_pool`
   - `profile_only_no_graph_root_pool`
   - `profile_only_graph_root_pool`
2. 增加 mean pooling 下的 BERT/profile 拆分，确认 gossipcop 最强结果来自 BERT、profile 还是二者互补。
3. 检查 root 节点语义，确认每个图第一个节点确实是新闻源节点。
4. 做图规模控制后的评估，例如按节点数分桶、加入 node_count covariate、或构造 size-matched split。
5. 若后续升级图模型，应优先做 root-aware、direction-aware、time-aware 的传播模型，而不是直接换通用 GAT。

跨模态注意力仍应等待真实图像/视频模态接入后再做。目前这些实验还不足以支持跨模态模块的必要性。
