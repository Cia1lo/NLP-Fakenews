# Root/Mean Pooling + BERT/Profile 拆分与图规模分桶控制分析

## 实验设置

本轮补充的是 `pooling_splits` suite，覆盖：

- 数据集：`politifact`, `gossipcop`
- 随机种子：`42`, `43`, `44`
- 特征拆分：`bert_only`, `profile_only`
- 图结构：`graph`, `no_graph`
- 池化方式：`root_pool`, `mean_pool`
- 训练参数：`epochs=50`, `batch_size=32`, `hidden_dim=128`, `patience=10`
- 设备：`--device auto`，实际解析为 `mps`

完整结果文件：

- `outputs/ablations_pooling_splits/summary_by_experiment.csv`
- `outputs/ablations_pooling_splits/test_size_bucket_summary_by_experiment.csv`

本轮图规模控制采用测试集节点数三分位分桶，而不是一对一匹配。`gossipcop` 每桶约 365 条样本，适合做分桶控制；`politifact` 每桶只有 21-22 条样本，且测试集极度偏 fake，因此只作为辅助观察。

## 总体结果

### gossipcop

| Experiment | Macro-F1 | AUC | Best Val Macro-F1 |
|---|---:|---:|---:|
| `bert_only_graph_mean_pool` | 0.9570 +/- 0.0016 | 0.9880 +/- 0.0014 | 0.9785 |
| `bert_only_graph_root_pool` | 0.9561 +/- 0.0032 | 0.9885 +/- 0.0009 | 0.9791 |
| `bert_only_no_graph_mean_pool` | 0.9531 +/- 0.0045 | 0.9842 +/- 0.0047 | 0.9773 |
| `profile_only_graph_root_pool` | 0.9430 +/- 0.0021 | 0.9728 +/- 0.0028 | 0.9625 |
| `profile_only_graph_mean_pool` | 0.9412 +/- 0.0027 | 0.9752 +/- 0.0026 | 0.9718 |
| `profile_only_no_graph_mean_pool` | 0.9204 +/- 0.0126 | 0.9684 +/- 0.0064 | 0.9496 |
| `profile_only_no_graph_root_pool` | 0.8375 +/- 0.0099 | 0.9117 +/- 0.0052 | 0.8705 |
| `bert_only_no_graph_root_pool` | 0.7555 +/- 0.0093 | 0.8407 +/- 0.0083 | 0.7940 |

关键差值：

| 对比 | Macro-F1 差值 |
|---|---:|
| BERT graph mean - no_graph mean | +0.0039 |
| BERT graph root - no_graph root | +0.2006 |
| BERT no_graph mean - no_graph root | +0.1976 |
| Profile graph mean - no_graph mean | +0.0207 |
| Profile graph root - no_graph root | +0.1055 |
| Profile no_graph mean - no_graph root | +0.0829 |

### politifact

| Experiment | Macro-F1 | AUC | Best Val Macro-F1 |
|---|---:|---:|---:|
| `bert_only_no_graph_root_pool` | 0.8856 +/- 0.0378 | 0.9881 +/- 0.0064 | 0.7328 |
| `bert_only_graph_root_pool` | 0.8582 +/- 0.0131 | 0.9911 +/- 0.0080 | 0.7328 |
| `profile_only_graph_mean_pool` | 0.5755 +/- 0.0374 | 0.3810 +/- 0.0763 | 0.7131 |
| `bert_only_graph_mean_pool` | 0.5529 +/- 0.0129 | 0.8921 +/- 0.0239 | 0.7438 |
| `profile_only_graph_root_pool` | 0.4951 +/- 0.0478 | 0.5067 +/- 0.1143 | 0.6540 |
| `bert_only_no_graph_mean_pool` | 0.4884 +/- 0.0376 | 0.5543 +/- 0.0468 | 0.4966 |
| `profile_only_no_graph_root_pool` | 0.4774 +/- 0.0093 | 0.3757 +/- 0.0136 | 0.6247 |
| `profile_only_no_graph_mean_pool` | 0.4353 +/- 0.0029 | 0.3214 +/- 0.0677 | 0.7328 |

`politifact` 的 split 过小且测试集 fake 占比过高，root pooling 下的 BERT 高分不应作为主要结论。这里更合理的用途是暴露小样本 split 的不稳定性。

## 图规模分桶控制

### gossipcop 分桶组成

| Bucket | N | Fake Rate | Avg Nodes |
|---|---:|---:|---:|
| small | 365 | 0.838 | 10.9 |
| medium | 364 | 0.382 | 40.7 |
| large | 365 | 0.252 | 101.6 |

图规模和标签明显相关：小图大多是 fake，大图更多是 real。因此只看总体分数容易高估模型是否真正学到了内容或传播机制。

### gossipcop 分桶 Macro-F1

| Experiment | Small | Medium | Large |
|---|---:|---:|---:|
| `bert_only_graph_mean_pool` | 0.9135 | 0.9648 | 0.9336 |
| `bert_only_graph_root_pool` | 0.9146 | 0.9541 | 0.9447 |
| `bert_only_no_graph_mean_pool` | 0.8960 | 0.9604 | 0.9315 |
| `bert_only_no_graph_root_pool` | 0.6650 | 0.7302 | 0.7290 |
| `profile_only_graph_mean_pool` | 0.8823 | 0.9532 | 0.9137 |
| `profile_only_graph_root_pool` | 0.8904 | 0.9497 | 0.9179 |
| `profile_only_no_graph_mean_pool` | 0.8946 | 0.9057 | 0.8829 |
| `profile_only_no_graph_root_pool` | 0.7638 | 0.8408 | 0.7577 |

分桶后结论仍然比较清楚：

- BERT + mean pooling 即使不显式用图，也在三个桶内保持较高性能，说明它不是只靠全局图规模先验。
- BERT + root pooling 在无图时明显失效；加图后恢复到接近 mean pooling，说明图传播主要是在 root 表征上补回邻居/评论节点信息。
- Profile-only 的表现很强，尤其是 mean pooling 或 graph root pooling，这说明用户 profile/群体组成是强信号。但这类信号也最容易带来数据集偏置或平台偏置，需要作为控制项而不是主要创新论据。
- 中等规模图上的分数最高，小图和大图下降，说明图规模/类别分布仍然影响评估。

### politifact 分桶提醒

`politifact` 测试分桶如下：

| Bucket | N | Fake Rate | Avg Nodes |
|---|---:|---:|---:|
| small | 22 | 0.818 | 18.2 |
| medium | 21 | 0.857 | 94.7 |
| large | 21 | 0.952 | 280.4 |

每桶样本太少，而且 fake 比例极高，分桶结果无法稳定支持模型选择。后续如果继续使用 `politifact`，应优先改 split 或做 repeated stratified split。

## 结论

当前最可靠的结论来自 `gossipcop`：

1. 最强基线是 `bert_only_graph_mean_pool`，但它只比 `bert_only_no_graph_mean_pool` 高 0.0039 Macro-F1。对 BERT 特征而言，显式图结构在 mean pooling 下增益很小。
2. 图结构对 root pooling 很关键。`bert_only_no_graph_root_pool` 只有 0.7555，加入图传播后到 0.9561，说明图传播主要帮助 root 节点吸收整棵传播树的信息。
3. Profile 特征本身很强。`profile_only_no_graph_mean_pool` 已达到 0.9204，加入图结构后到 0.9412/0.9430。这是有效信号，但也可能是数据集偏置来源。
4. 图规模是明显混杂因素。小图 fake rate 为 0.838，大图 fake rate 为 0.252。分桶后模型仍有效，但后续论文实验必须保留图规模控制。

## 下一步建议

暂时不急着升级到 GAT、跨模态注意力或更复杂 PyG 模型。更稳的下一步是继续做控制实验：

1. 加一个 `size_only` baseline：只用 `num_nodes`, `log_num_nodes`, `edge_count`, 简单深度/宽度统计，量化图规模先验本身能拿多少分。
2. 对 `gossipcop` 做 matched evaluation：按 `log(num_nodes)` 在 real/fake 间做最近邻或 propensity matching，再重新评估当前八组模型。
3. 把 `profile_only` 作为偏置控制项固定纳入论文实验，避免把用户群体偏差误写成模型创新。
4. 只有在 size-only 和 matched evaluation 之后，BERT/profile/graph 的增益仍然稳定，再升级 GAT 或跨模态注意力。
