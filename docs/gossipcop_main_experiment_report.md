# gossipcop 主实验整理与模型升级决策

## 实验设置

本阶段将 `gossipcop` 作为主实验集，`politifact` 仅作为小样本辅助观察。原因是 `gossipcop` 测试集规模更大、三分桶后每桶约 365 条样本，更适合做稳定的消融和图规模控制。

基础设置：

- 数据集：`gossipcop`
- 随机种子：`42`, `43`, `44`
- 训练轮数：`epochs=50`
- Early stopping：`patience=10`, 监控 validation Macro-F1
- Batch size：`32`
- Hidden dim：`128`
- Device：`auto`，实际使用 MPS
- 主指标：Macro-F1，辅助指标为 Accuracy、Precision、Recall、F1、AUC

结果来源：

- `outputs/ablations_pooling_splits/summary_by_experiment.csv`
- `outputs/ablations_pooling_splits/test_size_bucket_summary_by_experiment.csv`
- `outputs/ablations_pooling_splits/matched_caliper_0_25/test_matched_summary_by_experiment.csv`

## 正式实验表

| 实验 | 特征 | 图层 | 池化 | Acc | Precision | Recall | F1 | Macro-F1 | AUC |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `bert_only_graph_mean_pool` | BERT | 2 | mean | 0.9570 | 0.9643 | 0.9479 | 0.9559 | **0.9570 +/- 0.0016** | 0.9880 |
| `bert_only_graph_root_pool` | BERT | 2 | root | 0.9561 | 0.9589 | 0.9516 | 0.9552 | 0.9561 +/- 0.0032 | **0.9885** |
| `bert_only_no_graph_mean_pool` | BERT | 0 | mean | 0.9531 | 0.9441 | 0.9615 | 0.9527 | 0.9531 +/- 0.0045 | 0.9842 |
| `profile_only_graph_root_pool` | profile | 2 | root | 0.9430 | 0.9387 | 0.9460 | 0.9422 | 0.9430 +/- 0.0021 | 0.9728 |
| `profile_only_graph_mean_pool` | profile | 2 | mean | 0.9412 | 0.9455 | 0.9342 | 0.9397 | 0.9412 +/- 0.0027 | 0.9752 |
| `profile_only_no_graph_mean_pool` | profile | 0 | mean | 0.9205 | 0.9241 | 0.9137 | 0.9186 | 0.9204 +/- 0.0126 | 0.9684 |
| `profile_only_no_graph_root_pool` | profile | 0 | root | 0.8376 | 0.8121 | 0.8709 | 0.8404 | 0.8375 +/- 0.0099 | 0.9117 |
| `bert_only_no_graph_root_pool` | BERT | 0 | root | 0.7590 | 0.8203 | 0.6524 | 0.7265 | 0.7555 +/- 0.0093 | 0.8407 |
| `size_only` | graph size | 0 | size_only | 0.7459 | 0.7448 | 0.7337 | 0.7392 | 0.7457 +/- 0.0000 | 0.7949 |

主结论：

- 最强模型是 `bert_only_graph_mean_pool`，Macro-F1 为 `0.9570`。
- `bert_only_no_graph_mean_pool` 已达到 `0.9531`，与最强模型只差 `0.0039`，说明 BERT 节点特征的 mean pooling 已经是很强的基线。
- `size_only` 有 `0.7457` Macro-F1，说明图规模确实是一个强混杂因素，但仍明显低于 BERT/profile 模型。

## 消融表

### 图结构贡献

| 对比 | Macro-F1 差值 | 解释 |
|---|---:|---|
| BERT graph mean - BERT no_graph mean | +0.0039 | 在 mean pooling 下，显式图传播增益很小 |
| BERT graph root - BERT no_graph root | +0.2006 | root-only 必须依赖图传播吸收回复/转发节点信息 |
| profile graph mean - profile no_graph mean | +0.0207 | profile 特征在图传播后有稳定增益 |
| profile graph root - profile no_graph root | +0.1055 | profile root-only 同样明显依赖传播结构 |

### 池化方式贡献

| 对比 | Macro-F1 差值 | 解释 |
|---|---:|---|
| BERT no_graph mean - BERT no_graph root | +0.1976 | 不用图传播时，mean pooling 远优于只看 root |
| profile no_graph mean - profile no_graph root | +0.0829 | 用户群体分布比单个 root 用户更有判别力 |
| BERT graph mean - BERT graph root | +0.0009 | 加图传播后，mean/root 基本持平 |
| profile graph root - profile graph mean | +0.0018 | 加图传播后，profile 的 root/mean 差异也基本消失 |

### 特征贡献

| 结论 | 证据 |
|---|---|
| BERT 是当前主信号 | BERT 最强 `0.9570`，profile 最强 `0.9430` |
| profile 是强辅助/偏置控制信号 | profile-only graph 已超过 `0.94`，需要作为控制项保留 |
| 图规模不能作为主要解释 | `size_only` 原测试集 `0.7457`，低于所有有效 BERT/profile 模型 |

## 图规模控制表

### 三分桶控制

`gossipcop` 测试集按节点数三分桶：

| Bucket | N | Fake Rate | Avg Nodes |
|---|---:|---:|---:|
| small | 365 | 0.838 | 10.9 |
| medium | 364 | 0.382 | 40.7 |
| large | 365 | 0.252 | 101.6 |

标签和图规模高度相关：小图大多为 fake，大图更多为 real。因此必须报告 size control。

| 实验 | Small Macro-F1 | Medium Macro-F1 | Large Macro-F1 |
|---|---:|---:|---:|
| `bert_only_graph_mean_pool` | 0.9135 | **0.9648** | 0.9336 |
| `bert_only_graph_root_pool` | **0.9146** | 0.9541 | **0.9447** |
| `bert_only_no_graph_mean_pool` | 0.8960 | 0.9604 | 0.9315 |
| `profile_only_graph_mean_pool` | 0.8823 | 0.9532 | 0.9137 |
| `profile_only_graph_root_pool` | 0.8904 | 0.9497 | 0.9179 |
| `profile_only_no_graph_mean_pool` | 0.8946 | 0.9057 | 0.8829 |
| `size_only` | 0.4560 | 0.6626 | 0.4292 |

分桶结论：

- BERT 模型在 small/medium/large 三个桶内均明显优于 `size_only`。
- `size_only` 在分桶内表现较弱，说明它主要利用的是全局图规模和标签分布相关性，而不是桶内稳定判别能力。
- 中等规模图上所有有效模型表现最好；小图和大图仍是更难的评估区域。

### 严格匹配控制

使用 `log1p(node_count)` 做 real/fake 一对一最近邻匹配，并设置 `caliper=0.25`。

匹配效果：

| 指标 | 数值 |
|---|---:|
| matched pairs | 295 |
| matched samples | 590 |
| SMD before | 1.221 |
| SMD after | 0.148 |

严格匹配后，图规模差异从严重不平衡降到较低水平。

| 实验 | Matched Macro-F1 | Matched AUC |
|---|---:|---:|
| `bert_only_graph_root_pool` | **0.9429** | **0.9822** |
| `bert_only_graph_mean_pool` | 0.9395 | 0.9787 |
| `bert_only_no_graph_mean_pool` | 0.9328 | 0.9697 |
| `profile_only_graph_root_pool` | 0.9322 | 0.9629 |
| `profile_only_graph_mean_pool` | 0.9299 | 0.9661 |
| `profile_only_no_graph_mean_pool` | 0.9027 | 0.9554 |
| `profile_only_no_graph_root_pool` | 0.8067 | 0.8674 |
| `bert_only_no_graph_root_pool` | 0.7439 | 0.8366 |
| `size_only` | 0.5437 | 0.5416 |

严格匹配结论：

- `size_only` 从原测试集 Macro-F1 `0.7457` 降到 `0.5437`，AUC 从 `0.7949` 降到 `0.5416`，接近随机。
- BERT/profile 模型匹配后仍保持高性能，说明当前有效模型不是单纯依赖图规模。
- 图传播在严格匹配后仍有小幅作用：`bert_only_graph_mean_pool` 比 `bert_only_no_graph_mean_pool` 高 `0.0067` Macro-F1。
- 最强匹配结果变为 `bert_only_graph_root_pool`，说明在控制图规模后，图传播对 root 表征的聚合价值更清晰。

## 是否进入 GAT / 跨模态注意力升级

### GAT / PyG 升级决策

结论：暂不把 GAT 作为下一阶段主线；可以作为小规模对照实验实现，但不应替代当前主线。

理由：

- 当前 GraphSAGE-style mean aggregation 已经让 root pooling 从 `0.7555` 提升到 `0.9561`，说明基础图传播机制已经捕获主要收益。
- 在最强的 mean pooling 设定下，显式图传播只带来 `+0.0039` 原测试 Macro-F1 和 `+0.0067` 严格匹配 Macro-F1，增益很小。
- GAT 的计算成本和调参复杂度更高，但现有证据不足以说明注意力式邻居加权会带来稳定收益。

建议：

- 将 GAT 放在“补充 SOTA 对照”位置，而不是当前模型主线。
- 若实现 GAT，应只在 `gossipcop` 上先做小范围对照：BERT-only、root pooling、matched evaluation。
- GAT 是否保留，取决于严格匹配后是否稳定超过当前 `bert_only_graph_root_pool`。

### 跨模态注意力升级决策

结论：暂不进入跨模态注意力升级。

理由：

- 当前正式实验只使用预计算文本节点特征 BERT、profile 和传播图结构。
- 还没有接入真实图像、视频或 CLIP/Swin 特征。
- 在没有真实多模态输入前，上跨模态注意力会变成结构复杂但证据不足的模块，论文说服力不强。

建议：

- 先完成当前文本/profile/传播图实验闭环。
- 若后续能稳定接入图像或视频特征，再设计 text-image/profile-propagation 的跨模态注意力。
- 在此之前，论文创新点应聚焦为：多源节点特征、传播结构、图规模控制、严格匹配评估和可解释消融。

## 当前阶段结论

可以将 `gossipcop` 作为主实验集进入正式结果撰写。当前推荐主模型/主表述为：

```text
BERT node features + propagation graph aggregation + root/mean pooling controls
```

论文中的核心论点建议写成：

1. BERT 节点语义特征是主要判别信号。
2. 传播图结构对 root-only 表征非常关键，对 mean pooling 是小幅增益。
3. profile/user 群体特征很强，但必须作为偏置控制项谨慎解释。
4. 图规模是重要混杂因素；经过严格匹配后，size-only 近似失效，而 BERT/profile 模型仍保持高性能。
5. 当前证据支持先完成可解释的强基线和控制实验；GAT 可作为补充对照，跨模态注意力等待真实图像/视频特征接入后再做。
