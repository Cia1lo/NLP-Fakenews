# gossipcop 鲁棒性与解释性分析

## 实验设置

本阶段保留当前 GraphSAGE-style mean aggregation 模型，不进入 GAT 或跨模态注意力升级。分析重点转为：

- 图规模控制后的模型稳定性
- 测试时扰动鲁棒性
- occlusion 风格解释性分析

本次鲁棒性分析直接加载已有 checkpoint，不重训模型。

分析对象：

- `bert_profile_graph_mean_pool`
- `bert_profile_graph_root_pool`
- `bert_profile_no_graph_mean_pool`
- `bert_profile_no_graph_root_pool`

数据集：

- `gossipcop`

扰动方式：

- `clean`
- `edge_drop_0.25`
- `edge_drop_0.5`
- `zero_bert`
- `zero_profile`
- `noise_bert`
- `noise_profile`

结果文件：

- `outputs/robustness_gossipcop/test_robustness_summary.csv`
- `outputs/robustness_gossipcop/test_robustness_summary_by_experiment.csv`
- `outputs/robustness_gossipcop/explainability_summary.csv`
- `outputs/robustness_gossipcop/explainability_summary_by_experiment.csv`

## 鲁棒性结果

| 实验 | 扰动 | Macro-F1 | AUC |
|---|---|---:|---:|
| `bert_profile_graph_mean_pool` | clean | 0.9607 +/- 0.0040 | 0.9856 |
| `bert_profile_graph_mean_pool` | edge_drop_0.25 | 0.9646 +/- 0.0005 | 0.9856 |
| `bert_profile_graph_mean_pool` | edge_drop_0.5 | 0.9619 +/- 0.0026 | 0.9856 |
| `bert_profile_graph_mean_pool` | noise_bert | 0.9600 +/- 0.0051 | 0.9856 |
| `bert_profile_graph_mean_pool` | noise_profile | 0.9610 +/- 0.0028 | 0.9857 |
| `bert_profile_graph_mean_pool` | zero_bert | 0.3435 +/- 0.0246 | 0.7508 |
| `bert_profile_graph_mean_pool` | zero_profile | 0.3292 +/- 0.0000 | 0.5031 |
| `bert_profile_graph_root_pool` | clean | 0.9476 +/- 0.0074 | 0.9782 |
| `bert_profile_graph_root_pool` | edge_drop_0.25 | 0.9369 +/- 0.0090 | 0.9719 |
| `bert_profile_graph_root_pool` | edge_drop_0.5 | 0.9219 +/- 0.0096 | 0.9667 |
| `bert_profile_graph_root_pool` | noise_bert | 0.9467 +/- 0.0062 | 0.9783 |
| `bert_profile_graph_root_pool` | noise_profile | 0.9464 +/- 0.0073 | 0.9779 |
| `bert_profile_graph_root_pool` | zero_bert | 0.7754 +/- 0.0179 | 0.8892 |
| `bert_profile_graph_root_pool` | zero_profile | 0.3502 +/- 0.0362 | 0.7342 |
| `bert_profile_no_graph_mean_pool` | clean | 0.9625 +/- 0.0048 | 0.9865 |
| `bert_profile_no_graph_mean_pool` | noise_bert | 0.9625 +/- 0.0048 | 0.9865 |
| `bert_profile_no_graph_mean_pool` | noise_profile | 0.9631 +/- 0.0043 | 0.9865 |
| `bert_profile_no_graph_mean_pool` | zero_bert | 0.3356 +/- 0.0031 | 0.7741 |
| `bert_profile_no_graph_mean_pool` | zero_profile | 0.4520 +/- 0.0696 | 0.4715 |
| `bert_profile_no_graph_root_pool` | clean | 0.8540 +/- 0.0104 | 0.9335 |
| `bert_profile_no_graph_root_pool` | noise_bert | 0.8531 +/- 0.0125 | 0.9338 |
| `bert_profile_no_graph_root_pool` | noise_profile | 0.8473 +/- 0.0106 | 0.9245 |
| `bert_profile_no_graph_root_pool` | zero_bert | 0.7511 +/- 0.0196 | 0.8898 |
| `bert_profile_no_graph_root_pool` | zero_profile | 0.3292 +/- 0.0000 | 0.6855 |

## 关键观察

### 1. 小幅特征噪声影响很小

`noise_bert` 和 `noise_profile` 几乎不降低 Macro-F1：

| 实验 | noise_bert Macro-F1 drop | noise_profile Macro-F1 drop |
|---|---:|---:|
| `bert_profile_graph_mean_pool` | +0.0006 | -0.0003 |
| `bert_profile_graph_root_pool` | +0.0009 | +0.0012 |
| `bert_profile_no_graph_mean_pool` | +0.0000 | -0.0006 |
| `bert_profile_no_graph_root_pool` | +0.0009 | +0.0067 |

这说明当前模型对 0.1 倍标准差级别的特征扰动较稳定。

### 2. mean pooling 对删边不敏感，root pooling 对删边敏感

| 实验 | edge_drop_0.25 drop | edge_drop_0.5 drop |
|---|---:|---:|
| `bert_profile_graph_mean_pool` | -0.0040 | -0.0012 |
| `bert_profile_graph_root_pool` | +0.0107 | +0.0257 |

这里的 drop 定义为 `clean Macro-F1 - perturbed Macro-F1`。负值表示扰动后没有下降。

解释：

- mean pooling 本身直接聚合全图节点，因此删边对读出影响很小。
- root pooling 依赖图传播把邻居信息传到 root，删边会阻断传播路径，所以性能下降更明显。

### 3. occlusion 显示 BERT 和 profile 都是强依赖信号

| 实验 | occlude_bert drop | occlude_profile drop |
|---|---:|---:|
| `bert_profile_graph_mean_pool` | 0.6172 | 0.6314 |
| `bert_profile_graph_root_pool` | 0.1722 | 0.5974 |
| `bert_profile_no_graph_mean_pool` | 0.6269 | 0.5105 |
| `bert_profile_no_graph_root_pool` | 0.1029 | 0.5248 |

解释：

- mean pooling 模型同时依赖 BERT 和 profile，任一模态被置零都会造成大幅下降。
- root pooling 下 profile occlusion 的影响明显大于 BERT occlusion，说明 root 用户/profile 信息是 root-only 判别的重要来源。
- 置零特征属于较强 out-of-distribution 扰动，因此更适合作为 occlusion 解释，而不是常规鲁棒性结论。

## 对当前路线的影响

当前结果继续支持“不升级主模型，先做控制和解释”的路线：

1. 图结构的价值主要体现在 root pooling。
   - root pooling 删边后明显下降。
   - mean pooling 删边后基本不变。

2. profile 是强判别信号，但也需要谨慎解释。
   - profile occlusion 造成大幅下降。
   - profile-only 模型此前也表现很强。
   - 因此 profile 可能既是有效元数据，也可能携带平台/用户群体偏置。

3. BERT 仍是主语义信号。
   - mean pooling 下 occlude BERT 的下降非常大。
   - BERT-only 模型此前在严格匹配后仍显著强于 size-only。

4. GAT 暂时没有成为主线的必要。
   - 当前鲁棒性结果显示，mean pooling 对边扰动不敏感。
   - root pooling 对边敏感，但基础图传播已经提供主要收益。
   - 继续升级 GAT 前，应优先完成解释性和偏置控制分析。

## 下一步实验建议

继续保持当前模型，建议补三类分析：

1. 缺失模态训练对照
   - 当前是测试时 occlusion。
   - 后续可训练 `bert_profile` 模型的 modality dropout 版本，提升对缺失 BERT/profile 的鲁棒性。

2. 错误案例解释
   - 按 false positive / false negative 分组。
   - 对每组统计 node_count、prob_fake、occlusion delta、size bucket。

3. profile 偏置审查
   - 分析 profile-only 高置信样本。
   - 检查 profile 与图规模、标签、传播深度之间的相关性。
   - 将 profile 作为偏置控制项，而不是直接作为主要创新点。
