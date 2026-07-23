# ResNet 宽泛摘要问题运行报告

## 1. 运行信息

- 时间：2026-06-15
- 命令：`python -m paper_rag ask "Resnet讲了哪些主要内容" --debug --evidence`
- 退出码：`0`
- 总墙钟时间：`34355.36 ms`
- 最终状态：成功
- 回答模式：`llm`
- 路由：`content`
- 意图：`summary`

## 2. 最终回答

ResNet（《Deep Residual Learning for Image Recognition》）的主要内容包括：

1. **残差学习框架**：提出了一种新的深度神经网络训练方法，通过引入“残差函数”来简化深层网络的优化问题。传统的网络要求每一层直接拟合目标映射 $\mathcal{H}(x)$，而 ResNet 则让网络学习残差函数 $\mathcal{F}(x) = \mathcal{H}(x) - x$，最终输出为 $\mathcal{F}(x) + x$。这种设计使得网络更容易优化，并缓解了梯度消失/爆炸的问题。
2. **解决退化问题**：在传统网络中，随着网络深度增加，训练误差和测试误差可能反而上升，这被称为“退化问题”。ResNet 通过残差学习结构有效解决了这一问题，证明了更深的网络可以更轻松地训练并获得更高的精度。
3. **实验验证**：论文在多个数据集上进行了广泛的实验验证，包括 ImageNet、CIFAR-10 等。例如，在 ImageNet 上，ResNet 实现了 152 层的极深模型，其复杂度低于 VGG 网络，且在测试集上的 Top-5 错误率为 3.57%，获得了 ILSVRC 2015 分类竞赛的第一名。
4. **实现细节**：论文详细描述了 ResNet 的实现方式，包括使用批量归一化（BN）、随机缩放增强、SGD 优化器等技术。此外，还探讨了不同网络深度对性能的影响，并展示了超过 1000 层的模型的可行性。

这些内容主要来源于论文的摘要（Abstract）、引言（Introduction）、第 3 章“Deep Residual Learning”以及第 4 章“Experiments”部分。

## 3. Planner 中间结果

### 3.1 顶层解析

```json
{
  "route": "content",
  "status": "ok",
  "intent": "summary",
  "parse_status": "ok"
}
```

问题被正确识别为论文内容摘要问题，没有被误判为 metadata 或 reference。

### 3.2 Scope 解析

```json
{
  "scope": [
    "paper=Deep Residual Learning for Image Recognition"
  ],
  "scope_records": [
    "Deep Residual Learning for Image Recognition"
  ],
  "resolved": {
    "aliases": [
      {
        "alias": "ResNet",
        "canonical": "Deep Residual Learning for Image Recognition"
      }
    ]
  }
}
```

`Resnet` 成功通过别名解析到论文 `Deep Residual Learning for Image Recognition`。本轮只有一个论文对象，因此没有触发 `per/and` group 检索。

### 3.3 检索 query

```json
{
  "dense_query": "Resnet讲了哪些主要内容",
  "bm25_queries": [
    "讲了",
    "主要内容",
    "told",
    "It was explained.",
    "main content"
  ]
}
```

Dense query 保留了原问题。BM25 query 对宽泛摘要问题帮助较小，其中 `讲了`、`told` 和 `It was explained.` 不具备论文主题区分能力。

## 4. 检索与排序

本轮返回固定的 8 个 context。所有结果都只有 Dense 来源，没有 BM25 命中进入融合结果。

| 排名 | chunk_id | 章节 | 页码 | RRF score | Dense score |
| --- | --- | --- | --- | ---: | ---: |
| 1 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0009` | 3 Deep Residual Learning | 3 | 0.016393 | 0.637637 |
| 2 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0017` | 4 Experiments | 4 | 0.016129 | 0.610337 |
| 3 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0006` | 1 Introduction | 2 | 0.015873 | 0.602907 |
| 4 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0000` | Abstract | 1 | 0.015625 | 0.594448 |
| 5 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0010` | 3.1 Residual Learning | 3 | 0.015385 | 0.583124 |
| 6 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0012` | 3.2 Identity Mapping by Shortcuts | 3 | 0.015152 | 0.580148 |
| 7 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0016` | 3.4 Implementation | 4 | 0.014925 | 0.578206 |
| 8 | `Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0001` | 1 Introduction | 1 | 0.014706 | 0.577037 |

下面的 chunk_id、章节、页码和分数是本轮原始输出；为控制报告长度，context 正文仅保留内容摘要。

### 4.1 Context 1 内容摘要

```text
3. Deep Residual Learning
```

### 4.2 Context 2 内容摘要

```text
4. Experiments
```

### 4.3 Context 3 内容摘要

```text
We present comprehensive experiments on ImageNet to show the degradation
problem and evaluate our method. The residual nets are easier to optimize
than corresponding plain nets and gain accuracy from increased depth.

The paper also reports CIFAR-10 experiments with more than 100 layers and
exploration of models with more than 1000 layers. The 152-layer ImageNet
model has lower complexity than VGG, reaches 3.57% top-5 test error, and
wins ILSVRC 2015 classification. The representations also generalize to
detection, localization and segmentation tasks.
```

### 4.4 Context 4 内容摘要

```text
Abstract

The paper introduces a residual learning framework for training substantially
deeper networks. Layers learn residual functions relative to their inputs
instead of unreferenced functions. Experiments show that residual networks
are easier to optimize and benefit from increased depth.

It evaluates networks up to 152 layers on ImageNet, reports 3.57% test error,
and analyzes networks with 100 and 1000 layers on CIFAR-10.
```

### 4.5 Context 5 内容摘要

```text
3.1 Residual Learning

Instead of directly approximating H(x), the stacked layers approximate
F(x) := H(x) - x, so the original function becomes F(x) + x. This
reformulation is motivated by the degradation problem: if identity mappings
are optimal, residual layers can drive their weights toward zero more easily
than stacked nonlinear layers can learn an identity mapping directly.
```

### 4.6 Context 6 内容摘要

```text
3.2 Identity Mapping by Shortcuts

The residual building block is y = F(x, {Wi}) + x. A shortcut connection
performs element-wise addition without adding parameters or meaningful
computational complexity. When dimensions differ, a linear projection Ws
can be used so that y = F(x, {Wi}) + Ws x.
```

### 4.7 Context 7 内容摘要

```text
3.4 Implementation

For ImageNet, the implementation uses scale augmentation, random 224 x 224
crops, horizontal flips, color augmentation, batch normalization after each
convolution, SGD with batch size 256, an initial learning rate of 0.1,
weight decay 0.0001 and momentum 0.9.
```

### 4.8 Context 8 内容摘要

```text
1. Introduction

The introduction explains why network depth matters and distinguishes the
degradation problem from vanishing or exploding gradients. Even after
normalization and initialization make deep networks converge, adding layers
to a suitably deep plain network can still increase training error.
```

## 5. 分阶段耗时

| 阶段 | 耗时（ms） | 占墙钟时间 |
| --- | ---: | ---: |
| top_parser | 1459.83 | 4.25% |
| domain_parser | 2010.88 | 5.85% |
| scope | 0.11 | <0.01% |
| retrieval_query | 1533.58 | 4.46% |
| load_chunks | 29.69 | 0.09% |
| dense | 2785.19 | 8.11% |
| bm25 | 52.89 | 0.15% |
| fusion_context | 7.75 | 0.02% |
| plan 合计 | 7880.08 | 22.94% |
| answer | 26146.58 | 76.11% |
| 其他 CLI 开销 | 328.70 | 0.96% |
| 总墙钟 | 34355.36 | 100% |

主要耗时来自 LLM answer，占总时间约 76%。本地 scope、chunk 加载、BM25 和融合本身都很快。

## 6. 本轮判断

### 正常部分

- `Resnet` 大小写形式成功解析，没有出现上一轮的偶发解析失败。
- route、intent、论文 scope 和别名都正确。
- 证据覆盖摘要、引言、残差学习、shortcut、实现和实验，足以支持“主要内容”回答。
- 最终答案结构完整，核心贡献和实验结果基本有证据支撑。

### 暴露的问题

1. **宽泛摘要仍按普通 query 生成 BM25 关键词。** 这些词缺乏区分度，最终没有一个 BM25 结果进入融合。
2. **前两个 context 只有章节标题。** 它们占用了 top 8 中的两个位置，却没有向 LLM 提供实质内容。
3. **摘要问题依赖 Dense 偶然覆盖关键章节。** 本轮覆盖良好，但没有显式保证 Abstract、Introduction、Method 和 Experiments 都进入 context。
4. **答案有一处表述偏强。** 证据明确说明残差学习解决的是 degradation problem，而梯度消失/爆炸主要已由初始化和归一化缓解；回答中的“残差设计缓解梯度消失/爆炸”不够严谨。
5. **回答生成比检索慢得多。** 检索和规划约 7.88 秒，答案生成约 26.15 秒，当前延迟主要不是 RRF、BM25 或本地数据加载造成的。

## 7. 结论

本轮能够正常回答，而且回答质量总体可用。真正需要改进的不是让用户补关键词，而是为 `summary` 意图增加更确定的章节选择策略：优先保证 Abstract、Introduction、Method 和 Experiments 的实质文本进入 context，同时过滤只有标题的 chunk。BM25 在这类问题中可以降为补充，不应承担主要召回职责。
