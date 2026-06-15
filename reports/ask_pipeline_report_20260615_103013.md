# Paper RAG Ask Pipeline 检查报告

生成时间：2026-06-15 10:31:24
原始 payload：`reports/ask_pipeline_payloads_20260615_103013.json`

## 测试用例总览

| ID | 类型 | 问题 | route | intent | status | answer_mode | 结果摘要 | wall ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q1 | metadata-count | CVPR 有几篇论文？ | metadata | count | ok | local | count = 5; items: Deep Residual Learning for Image Recognition, Aggregated Residual Transformations for Deep Neural Networks, ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, Squeeze-and-Excitation Networks, Going deeper with convolutions | 6499.55 |
| Q2 | reference-list | 哪些论文引用了 ResNet？ | reference | list | ok | local | papers: A Discriminative Feature Learning Approach for Deep Face Recognition, Aggregated Residual Transformations for Deep Neural Networks, An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Attention is All you Need, ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks, EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits, Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning, NormFace: L2 Hypersphere Embedding for Face Verification, Squeeze-and-Excitation Networks, Supervised Contrastive Learning; edges = 11 | 5660.24 |
| Q3 | content-single | LSTM 的网络结构是什么？ | content | lookup | ok | llm | contexts = 8 | 30869.06 |
| Q4 | content-per-3-groups | ResNet、Transformer 和 ViT 的网络结构分别是什么样的？ | content | lookup | ok | llm | contexts = 9; groups = 3 | 28480.42 |

## 结论摘要

- `metadata` 和 `reference` 用本地回答链路，`answer_mode=local`。
- `content` 使用一次 LLM 组织答案；group 场景是多组分别检索后合并 evidence，不是每组调用一次 LLM。
- 三对象 group 用例中，最终 `results.contexts` 应扩展到至少 `3 * group_count = 9` 条，用于降低多对象信息缺失风险。
- 下方 context 表只保留 `section/pages/chunk_id` 等回源定位，不展开原文。

## Q1 metadata-count

问题：CVPR 有几篇论文？
目的：验证 metadata 路由、本地 count 回答、元数据过滤。
路由：`metadata` / intent：`count` / status：`ok` / answer_mode：`local`

### 阶段用时

| 阶段 | 耗时 ms |
| --- | --- |
| wall_observed | 6499.55 |
| top_parser | 3149.13 |
| domain_parser | 3336.36 |
| scope | 13.97 |
| plan | 6499.51 |
| answer | 0.02 |

### Planner / 中间过程

```json
{
  "scope": [
    "venue=CVPR"
  ]
}
```

### Answer

共找到 5 篇符合条件的论文。
[1] Deep Residual Learning for Image Recognition
[2] Aggregated Residual Transformations for Deep Neural Networks
[3] ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
[4] Squeeze-and-Excitation Networks
[5] Going deeper with convolutions

## Q2 reference-list

问题：哪些论文引用了 ResNet？
目的：验证 reference 路由、citation graph 查询和本地列表回答。
路由：`reference` / intent：`list` / status：`ok` / answer_mode：`local`

### 阶段用时

| 阶段 | 耗时 ms |
| --- | --- |
| wall_observed | 5660.24 |
| top_parser | 1621.06 |
| domain_parser | 4037.97 |
| scope | 1.12 |
| plan | 5660.19 |
| answer | 0.03 |

### Planner / 中间过程

```json
{
  "return_side": "source",
  "object_scope": [
    "paper=Deep Residual Learning for Image Recognition"
  ]
}
```

Resolved aliases：

| alias | canonical |
| --- | --- |
| ResNet | Deep Residual Learning for Image Recognition |

### Answer

[1] A Discriminative Feature Learning Approach for Deep Face Recognition
[2] Aggregated Residual Transformations for Deep Neural Networks
[3] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
[4] Attention is All you Need
[5] ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
[6] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
[7] Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits
[8] Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
[9] NormFace: L2 Hypersphere Embedding for Face Verification
[10] Squeeze-and-Excitation Networks
[11] Supervised Contrastive Learning
引用证据：
- A Discriminative Feature Learning Approach for Deep Face Recognition -> ResNet，页码 15，block b000131
- Aggregated Residual Transformations for Deep Neural Networks -> ResNet，页码 9，block b000140
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale -> ResNet，页码 10，block b000107
- Attention is All you Need -> ResNet，页码 10，block b000133
- ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks -> ResNet，页码 9，block b000121
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks -> ResNet，页码 9，block b000118
- Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits -> ResNet，页码 13，block b000115
- Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning -> ResNet，页码 7，block b000088
- NormFace: L2 Hypersphere Embedding for Face Verification -> ResNet，页码 10，block b000206
- Squeeze-and-Excitation Networks -> ResNet，页码 10，block b000122
- Supervised Contrastive Learning -> ResNet，页码 11，block b000111

## Q3 content-single

问题：LSTM 的网络结构是什么？
目的：验证单对象 content 检索、Dense/BM25/RRF、block window 和 LLM 回答。
路由：`content` / intent：`lookup` / status：`ok` / answer_mode：`llm`

### 阶段用时

| 阶段 | 耗时 ms |
| --- | --- |
| wall_observed | 30869.06 |
| top_parser | 1593.35 |
| domain_parser | 4060.68 |
| scope | 0.12 |
| retrieval_query | 3694.2 |
| load_chunks | 26.28 |
| dense | 2451.25 |
| bm25 | 62.28 |
| fusion_context | 17.99 |
| plan | 11906.33 |
| answer | 18962.69 |

### Planner / 中间过程

```json
{
  "scope": [
    "paper=Long Short-Term Memory"
  ],
  "content_objects": [
    "网络结构"
  ],
  "retrieval_query": {
    "dense_query": "查找论文中关于网络结构的相关内容",
    "bm25_queries": [
      "网络结构",
      "network structure",
      "Network architecture"
    ]
  }
}
```

Resolved aliases：

| alias | canonical |
| --- | --- |
| LSTM | Long Short-Term Memory |

### Context 定位

| # | title | section | pages | chunk_id |
| --- | --- | --- | --- | --- |
| 1 | Long Short-Term Memory | 1 INTRODUCTION | 1 > 2 | Long_Short_Term_Memory_ceb9e53d::chunk_0002 |
| 2 | Long Short-Term Memory | 2 PREVIOUS WORK | 2 | Long_Short_Term_Memory_ceb9e53d::chunk_0005 |
| 3 | Long Short-Term Memory | Abstract | 1 | Long_Short_Term_Memory_ceb9e53d::chunk_0000 |
| 4 | Long Short-Term Memory | 4 LONG SHORT-TERM MEMORY | 9 | Long_Short_Term_Memory_ceb9e53d::chunk_0030 |
| 5 | Long Short-Term Memory | 4 LONG SHORT-TERM MEMORY | 8 | Long_Short_Term_Memory_ceb9e53d::chunk_0026 |
| 6 | Long Short-Term Memory | 4 LONG SHORT-TERM MEMORY | 7 | Long_Short_Term_Memory_ceb9e53d::chunk_0023 |
| 7 | Long Short-Term Memory | 5 EXPERIMENTS | 10 > 11 | Long_Short_Term_Memory_ceb9e53d::chunk_0036 |
| 8 | Long Short-Term Memory | 1 INTRODUCTION | 2 | Long_Short_Term_Memory_ceb9e53d::chunk_0003 |

Scope records：

- Long Short-Term Memory

Debug context_units 摘要：

```json
[
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0002",
    "score": 0.031009615384615385,
    "sources": {
      "dense": {
        "rank": 5,
        "score": 0.3788454830646515
      },
      "bm25": {
        "rank": 4,
        "score": 0.030886196246139225
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0005",
    "score": 0.030886196246139225,
    "sources": {
      "dense": {
        "rank": 1,
        "score": 0.39281049370765686
      },
      "bm25": {
        "rank": 9,
        "score": 0.029631255487269532
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0000",
    "score": 0.0304147465437788,
    "sources": {
      "dense": {
        "rank": 2,
        "score": 0.3819049000740051
      },
      "bm25": {
        "rank": 10,
        "score": 0.02919863597612958
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0030",
    "score": 0.030303030303030304,
    "sources": {
      "dense": {
        "rank": 6,
        "score": 0.37663716077804565
      },
      "bm25": {
        "rank": 6,
        "score": 0.03021353930031804
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0026",
    "score": 0.02919863597612958,
    "sources": {
      "dense": {
        "rank": 9,
        "score": 0.35725679993629456
      },
      "bm25": {
        "rank": 8,
        "score": 0.029726775956284153
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0023",
    "score": 0.029116045245077504,
    "sources": {
      "dense": {
        "rank": 17,
        "score": 0.331688791513443
      },
      "bm25": {
        "rank": 2,
        "score": 0.03149801587301587
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0036",
    "score": 0.02862400327131466,
    "sources": {
      "dense": {
        "rank": 7,
        "score": 0.36772096157073975
      },
      "bm25": {
        "rank": 13,
        "score": 0.0261986301369863
      }
    }
  },
  {
    "chunk_id": "Long_Short_Term_Memory_ceb9e53d::chunk_0003",
    "score": 0.028612012987012988,
    "sources": {
      "dense": {
        "rank": 4,
        "score": 0.3791273832321167
      },
      "bm25": {
        "rank": 17,
        "score": 0.013157894736842105
      }
    }
  }
]
```

### Answer

LSTM（Long Short-Term Memory，长短期记忆网络）是一种特殊的循环神经网络（RNN）结构，旨在解决传统RNN在处理长期依赖问题时的梯度消失或爆炸问题。其核心设计是通过引入**门控机制**和**常量误差车轮（Constant Error Carousels, CEC）**来实现对信息的长期存储与选择性更新。

### LSTM 的主要组成部分：
1. **记忆单元（Memory Cell）**  
   - 记忆单元负责存储长期信息。它的内部状态 $ s_{c_j}(t) $ 通过自反馈连接保持不变，除非被输入门或输出门修改。
   - 公式：$ s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t)) $，其中 $ g $ 是一个可微函数（如Sigmoid），用于压缩输入信号。

2. **输入门（Input Gate）**  
   - 输入门控制哪些新信息可以进入记忆单元。它由一个Sigmoid激活函数生成的值 $ y^{in_j}(t) $ 决定是否允许当前输入影响记忆单元的状态。
   - 公式：$ net_{in_j}(t) = \sum_u w_{in_ju} y^u(t-1) $，然后通过Sigmoid函数计算 $ y^{in_j}(t) = \sigma(net_{in_j}(t)) $。

3. **输出门（Output Gate）**  
   - 输出门决定记忆单元中的信息何时被输出到其他部分。它同样由Sigmoid函数生成的值 $ y^{out_j}(t) $ 控制。
   - 公式：$ net_{out_j}(t) = \sum_u w_{out_ju} y^u(t-1) $，然后通过Sigmoid函数计算 $ y^{out_j}(t) = \sigma(net_{out_j}(t)) $。

4. **常量误差车轮（CEC）**  
   - CEC 是 LSTM 的关键特性之一，它确保了误差在反向传播过程中不会衰减或爆炸。通过截断梯度的方式，CEC 可以在不损失长期依赖能力的情况下高效地学习跨越数百甚至上千时间步的信息。

5. **网络拓扑**  
   - LSTM 网络通常包含一个输入层、一个隐藏层（包含记忆单元和门控单元）和一个输出层。隐藏层中的每个记忆单元都有对应的输入门和输出门。
   - 所有单元（除了门控单元）都通过全连接的方式传递信息给上一层的所有单元。

6. **计算复杂度**  
   - LSTM 的更新复杂度为 $ O(W) $，其中 $ W $ 是权重的数量。这使得 LSTM 在处理大规模数据时具有较高的效率。

### 总结
LSTM 通过引入输入门、输出门和记忆单元，结合 CEC 机制，能够有效地解决传统 RNN 在处理长期依赖问题时的梯度消失/爆炸问题。这种结构使得 LSTM 能够在处理序列数据（如语音识别、自然语言处理等）时表现出色，并且能够在噪声环境中稳定地学习长期依赖关系。

参考文献：《Long Short-Term Memory》论文第4章详细描述了 LSTM 的架构及其工作原理。

## Q4 content-per-3-groups

问题：ResNet、Transformer 和 ViT 的网络结构分别是什么样的？
目的：验证 per group 独立检索、三对象 context 配额和一次性 LLM 整合回答。
路由：`content` / intent：`lookup` / status：`ok` / answer_mode：`llm`

### 阶段用时

| 阶段 | 耗时 ms |
| --- | --- |
| wall_observed | 28480.42 |
| top_parser | 2644.42 |
| domain_parser | 5701.24 |
| scope | 0.35 |
| retrieval_query | 2982.98 |
| load_chunks | 0.13 |
| dense | 527.96 |
| bm25 | 2.46 |
| fusion_context | 22.21 |
| plan | 11882.02 |
| answer | 16598.36 |

### Planner / 中间过程

```json
{
  "groups": [
    {
      "scope": [
        "paper=Deep Residual Learning for Image Recognition"
      ]
    },
    {
      "scope": [
        "paper=Attention is All you Need"
      ]
    },
    {
      "scope": [
        "paper=An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
      ]
    }
  ],
  "group_mode": "per",
  "content_objects": [
    "网络结构"
  ],
  "retrieval_query": {
    "dense_query": "查找论文中关于网络结构的相关内容",
    "bm25_queries": [
      "网络结构",
      "network structure",
      "Network architecture"
    ]
  }
}
```

Resolved aliases：

| alias | canonical |
| --- | --- |
| ResNet | Deep Residual Learning for Image Recognition |
| Transformer | Attention is All you Need |
| ViT | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale |

### Group 效果

| # | scope | papers | count | exists | context_refs | refs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | paper=Deep Residual Learning for Image Recognition | Deep Residual Learning for Image Recognition | 1 | True | 8 | Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0001 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0014 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0015 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0029 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0002 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0005 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0006 > Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0008 |
| 2 | paper=Attention is All you Need | Attention is All you Need | 1 | True | 8 | Attention_is_All_you_Need_d87d482d::chunk_0007 > Attention_is_All_you_Need_d87d482d::chunk_0015 > Attention_is_All_you_Need_d87d482d::chunk_0006 > Attention_is_All_you_Need_d87d482d::chunk_0019 > Attention_is_All_you_Need_d87d482d::chunk_0000 > Attention_is_All_you_Need_d87d482d::chunk_0002 > Attention_is_All_you_Need_d87d482d::chunk_0003 > Attention_is_All_you_Need_d87d482d::chunk_0021 |
| 3 | paper=An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 1 | True | 8 | An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0000 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0030 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0029 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0013 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0023 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0032 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0014 > An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0031 |

### Context 定位

| # | title | section | pages | chunk_id |
| --- | --- | --- | --- | --- |
| 1 | Deep Residual Learning for Image Recognition | 1 Introduction | 1 | Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0001 |
| 2 | Attention is All you Need | 3 Model Architecture > 3.1 Encoder and Decoder Stacks | 2 > 3 | Attention_is_All_you_Need_d87d482d::chunk_0007 |
| 3 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | Abstract | 1 | An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0000 |
| 4 | Deep Residual Learning for Image Recognition | 3 Deep Residual Learning > 3.3 Network Architectures | 3 | Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0014 |
| 5 | Attention is All you Need | 3 Model Architecture > 3.3 Position-wise Feed-Forward Networks | 5 | Attention_is_All_you_Need_d87d482d::chunk_0015 |
| 6 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 4 EXPERIMENTS > 4.5 INSPECTING VISION TRANSFORMER | 8 | An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0030 |
| 7 | Deep Residual Learning for Image Recognition | 3 Deep Residual Learning > 3.3 Network Architectures | 4 | Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0015 |
| 8 | Attention is All you Need | 3 Model Architecture | 2 | Attention_is_All_you_Need_d87d482d::chunk_0006 |
| 9 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 4 EXPERIMENTS > 4.5 INSPECTING VISION TRANSFORMER | 8 | An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0029 |

Scope records：

- Deep Residual Learning for Image Recognition
- Attention is All you Need
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Debug context_units 摘要：

```json
[
  {
    "chunk_id": "Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0001",
    "score": 0.03177805800756621,
    "sources": {
      "dense": {
        "rank": 1,
        "score": 0.444548636674881
      },
      "bm25": {
        "rank": 5,
        "score": 0.030776515151515152
      }
    }
  },
  {
    "chunk_id": "Attention_is_All_you_Need_d87d482d::chunk_0007",
    "score": 0.032266458495966696,
    "sources": {
      "dense": {
        "rank": 3,
        "score": 0.34307539463043213
      },
      "bm25": {
        "rank": 1,
        "score": 0.03177805800756621
      }
    }
  },
  {
    "chunk_id": "An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0000",
    "score": 0.032266458495966696,
    "sources": {
      "dense": {
        "rank": 1,
        "score": 0.33163273334503174
      },
      "bm25": {
        "rank": 3,
        "score": 0.03125763125763126
      }
    }
  },
  {
    "chunk_id": "Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0014",
    "score": 0.03128054740957967,
    "sources": {
      "dense": {
        "rank": 6,
        "score": 0.3941091299057007
      },
      "bm25": {
        "rank": 2,
        "score": 0.031754032258064516
      }
    }
  },
  {
    "chunk_id": "Attention_is_All_you_Need_d87d482d::chunk_0015",
    "score": 0.03200204813108039,
    "sources": {
      "dense": {
        "rank": 2,
        "score": 0.3431965112686157
      },
      "bm25": {
        "rank": 3,
        "score": 0.03125
      }
    }
  },
  {
    "chunk_id": "An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0030",
    "score": 0.032018442622950824,
    "sources": {
      "dense": {
        "rank": 4,
        "score": 0.2995017468929291
      },
      "bm25": {
        "rank": 1,
        "score": 0.03278688524590164
      }
    }
  },
  {
    "chunk_id": "Deep_Residual_Learning_for_Image_Recognition_1e0651b6::chunk_0015",
    "score": 0.030886196246139225,
    "sources": {
      "dense": {
        "rank": 9,
        "score": 0.3921066224575043
      },
      "bm25": {
        "rank": 1,
        "score": 0.032266458495966696
      }
    }
  },
  {
    "chunk_id": "Attention_is_All_you_Need_d87d482d::chunk_0006",
    "score": 0.030536130536130537,
    "sources": {
      "dense": {
        "rank": 5,
        "score": 0.327728807926178
      },
      "bm25": {
        "rank": 6,
        "score": 0.030621785881252923
      }
    }
  },
  {
    "chunk_id": "An_Image_is_Worth_16x16_Words_Transformers_for_Image_Recognition_at_Scale_8ce7b839::chunk_0029",
    "score": 0.0315136476426799,
    "sources": {
      "dense": {
        "rank": 2,
        "score": 0.324748158454895
      },
      "bm25": {
        "rank": 5,
        "score": 0.016129032258064516
      }
    }
  }
]
```

### Answer

ResNet、Transformer 和 ViT 的网络结构如下：

1. **ResNet（Deep Residual Learning for Image Recognition）**：
   - ResNet 采用残差学习框架，通过引入“快捷连接”（shortcut connections）来解决深度神经网络中的退化问题。
   - 其基本结构由多个卷积层组成，通常使用 3×3 卷积核，并遵循两个设计规则：(i) 对于相同输出特征图尺寸的层，滤波器数量相同；(ii) 当特征图尺寸减半时，滤波器数量加倍以保持每层的时间复杂度。例如，34 层的 ResNet 模型包含 34 个参数层和 3.6 亿次浮点运算（FLOPs）。
   - 网络最后通过全局平均池化层和一个全连接层进行分类。

2. **Transformer（Attention is All you Need）**：
   - Transformer 架构基于自注意力机制（self-attention），其编码器和解码器均由多个相同的层堆叠而成。
   - 每个编码器层包括两个子层：多头自注意力机制和位置前馈网络（position-wise feed-forward network）。每个子层后都接有残差连接和层归一化。
   - 编码器共有 6 层，每层的维度为 $ d_{\text{model}} = 512 $，而前馈网络的中间层维度为 $ d_{\text{ff}} = 2048 $。

3. **ViT（An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale）**：
   - ViT 将图像划分为固定大小的块（如 16×16 像素），并将这些块展平后输入到标准的 Transformer 编码器中。
   - 在第一个层中，展平的图像块被线性投影到低维空间，并添加了可学习的位置嵌入（position embeddings）以保留图像的空间信息。
   - 自注意力机制允许 ViT 在最低层就整合整个图像的信息。实验表明，某些注意力头在较低层就能关注大部分图像区域，而其他注意力头则表现出局部化的特性。

以上内容分别参考了以下论文：
- ResNet：*Deep Residual Learning for Image Recognition*（第 3.3 节）
- Transformer：*Attention is All you Need*（第 3.1 节和 3.3 节）
- ViT：*An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*（第 4.5 节）
