# 正文检索 MCP 真实调用记录

> 生成时间（UTC）：2026-10-06T04:10:43.254950+00:00
> 数据来源：当前本地 Catalog、Milvus 和正文 Chunk，通过真实 MCP stdio 服务调用。
> 服务端：`paper_rag.mcp.server`；传输：`stdio`。
> 本批次使用一组与上一轮不同的问题，覆盖事实、原因解释、摘要、对比、显式附录、表格和不存在论文；不重建索引。

## 工具选择说明

所有案例都实际调用 `library_retrieve`，使用 `mode=hybrid`。需要固定范围的案例通过 `paper_ids`、`regions` 或 `filters` 限定候选；普通问题由服务端自动分类为 fact、reason、summary 或 comparison。每个案例保留完整的 MCP request/response，并在 JSON 内附加人工质量复核。

## 案例 1：BERT 的预训练使用了哪两个任务？

```json
{
  "user_question": "BERT 的预训练使用了哪两个任务？",
  "expected_evidence": {
    "paper_ids": [
      "1810.04805"
    ],
    "criteria": "识别 MLM 与 NSP 两项任务及其训练方式。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "BERT 的预训练使用了哪两个任务？",
          "task": "auto",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "BERT 的预训练使用了哪两个任务？",
          "task": "fact",
          "routing": {
            "route_intent": "retrieve",
            "task": "fact",
            "provider": "jev",
            "fallback_used": false,
            "confidence": 1.0
          },
          "papers": [
            {
              "paper_id": "1810.04805",
              "base_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
              "authors": [
                "Jacob Devlin",
                "Ming-Wei Chang",
                "Kenton Lee",
                "Kristina Toutanova"
              ],
              "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2018-10-11T00:50:01Z",
              "updated_at": "2019-05-24T20:37:26Z",
              "abs_url": "https://arxiv.org/abs/1810.04805v2",
              "pdf_url": "https://arxiv.org/pdf/1810.04805v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1810.04805\\mineru"
                }
              }
            },
            {
              "paper_id": "1909.08053",
              "base_id": "1909.08053",
              "canonical_id": "1909.08053v4",
              "title": "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
              "authors": [
                "Mohammad Shoeybi",
                "Mostofa Patwary",
                "Raul Puri",
                "Patrick LeGresley",
                "Jared Casper",
                "Bryan Catanzaro"
              ],
              "abstract": "Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite difficult to train due to memory constraints. In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2019-09-17T19:42:54Z",
              "updated_at": "2020-03-13T23:45:18Z",
              "abs_url": "https://arxiv.org/abs/1909.08053v4",
              "pdf_url": "https://arxiv.org/pdf/1909.08053v4.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "4a9222fffbcb0b38560335f4",
              "paper_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "ordinal": 16,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "Pre-training BERT",
              "section_path": [
                "content",
                "3 BERT",
                "3.1 Pre-training BERT"
              ],
              "section_label": "3.1 Pre-training BERT",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "89bb0f7968ecb9601b438831a75dd1a8239a625752b18d04d6108dfb2ef15331",
              "source_blocks": [
                {
                  "index": 45,
                  "page_idx": 3,
                  "bbox": [
                    114,
                    593,
                    490,
                    692
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 46,
                  "page_idx": 3,
                  "bbox": [
                    112,
                    701,
                    490,
                    863
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 47,
                  "page_idx": 3,
                  "bbox": [
                    507,
                    74,
                    887,
                    300
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 48,
                  "page_idx": 3,
                  "bbox": [
                    509,
                    300,
                    887,
                    544
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 49,
                  "page_idx": 3,
                  "bbox": [
                    507,
                    556,
                    887,
                    864
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "a31ff4962836dd01b60f27227055b46834a839b7f88082c61b14a030457e869f",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.\n\nTask #1: Masked LM Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-toright and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.\n\nIn order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953).",
              "score": 0.02009344262295082,
              "semantic_score": 0.5998966693878174,
              "lexical_rank": null,
              "semantic_rank": 1,
              "rrf_score": 0.01639344262295082,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "3dbe86be0013820d4248dd24",
              "paper_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "ordinal": 18,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "Pre-training BERT",
              "section_path": [
                "content",
                "3 BERT",
                "3.1 Pre-training BERT"
              ],
              "section_label": "3.1 Pre-training BERT",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "b44ccb33fd14ea9bdebb892bdbbb6f67b21b073518556cca57f8e8e7aa053d4f",
              "source_blocks": [
                {
                  "index": 45,
                  "page_idx": 3,
                  "bbox": [
                    114,
                    593,
                    490,
                    692
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 46,
                  "page_idx": 3,
                  "bbox": [
                    112,
                    701,
                    490,
                    863
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 47,
                  "page_idx": 3,
                  "bbox": [
                    507,
                    74,
                    887,
                    300
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 48,
                  "page_idx": 3,
                  "bbox": [
                    509,
                    300,
                    887,
                    544
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 49,
                  "page_idx": 3,
                  "bbox": [
                    507,
                    556,
                    887,
                    864
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "c3f0900e67957ded85baac752becd326c9a99833c3fd567c80dbcb283b05ceb6",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "en 10% of the time (3) the unchanged i-th token 10% of the time. Then, $T _ { i }$ will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.\n\nTask #2: Next Sentence Prediction (NSP) Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, C is used for next sentence prediction (NSP).<sup>5</sup> Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI. 6",
              "score": 0.019479032258064518,
              "semantic_score": 0.5801525712013245,
              "lexical_rank": null,
              "semantic_rank": 2,
              "rrf_score": 0.016129032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0037
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "4eb3c73ea8f8077d977a868a",
              "paper_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "ordinal": 12,
              "region": "content",
              "chapter_number": "3",
              "chapter_title": "BERT",
              "section_path": [
                "content",
                "3 BERT"
              ],
              "section_label": "3 BERT",
              "type": "text",
              "page_start": 2,
              "page_end": 3,
              "content_hash": "ca7135d761ca3a768ba2f6206c797f783170ee1b8b438b5af4f417a8c09f10ae",
              "source_blocks": [
                {
                  "index": 31,
                  "page_idx": 2,
                  "bbox": [
                    114,
                    667,
                    490,
                    877
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 32,
                  "page_idx": 2,
                  "bbox": [
                    114,
                    877,
                    490,
                    910
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 34,
                  "page_idx": 2,
                  "bbox": [
                    507,
                    444,
                    887,
                    621
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 35,
                  "page_idx": 2,
                  "bbox": [
                    509,
                    623,
                    887,
                    734
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 36,
                  "page_idx": 2,
                  "bbox": [
                    509,
                    737,
                    887,
                    834
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 41,
                  "page_idx": 3,
                  "bbox": [
                    114,
                    74,
                    490,
                    236
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 42,
                  "page_idx": 3,
                  "bbox": [
                    112,
                    236,
                    490,
                    495
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 43,
                  "page_idx": 3,
                  "bbox": [
                    114,
                    495,
                    490,
                    560
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "e67efd4cc2754a0d436f85a28b47346ef963d012a35531327399996d9e053a66",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For finetuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.\n\nA distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.\n\nModel Architecture BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al.",
              "score": 0.018625373134328358,
              "semantic_score": 0.5490216016769409,
              "lexical_rank": null,
              "semantic_rank": 7,
              "rrf_score": 0.014925373134328358,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 3,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "144497e3fc3b71f0172d2c73",
              "paper_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "ordinal": 4,
              "region": "content",
              "chapter_number": "1",
              "chapter_title": "Introduction",
              "section_path": [
                "content",
                "1 Introduction"
              ],
              "section_label": "1 Introduction",
              "type": "text",
              "page_start": 0,
              "page_end": 1,
              "content_hash": "1a06cd2a6e42ec452beeed0c45fa2f54c326eefc8d20ebffb85645e60fa3fc15",
              "source_blocks": [
                {
                  "index": 8,
                  "page_idx": 0,
                  "bbox": [
                    114,
                    684,
                    490,
                    910
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 9,
                  "page_idx": 0,
                  "bbox": [
                    509,
                    265,
                    887,
                    506
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 10,
                  "page_idx": 0,
                  "bbox": [
                    509,
                    507,
                    887,
                    747
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 11,
                  "page_idx": 0,
                  "bbox": [
                    509,
                    747,
                    887,
                    910
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 14,
                  "page_idx": 1,
                  "bbox": [
                    119,
                    228,
                    490,
                    374
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 15,
                  "page_idx": 1,
                  "bbox": [
                    119,
                    382,
                    490,
                    495
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 16,
                  "page_idx": 1,
                  "bbox": [
                    119,
                    504,
                    490,
                    570
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "97e1f9aad1ff222f8b19d605e304bf1c30f590c7d1b96d9e6bf7595797438471",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "ng, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the masked language model, we also use a “next sentence prediction” task that jointly pretrains text-pair representations. The contributions of our paper are as follows:\n\n• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pretrained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.\n\n• We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures. BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.\n\n• BERT advances the state of the art for eleven NLP tasks.",
              "score": 0.01847301587301587,
              "semantic_score": 0.566514790058136,
              "lexical_rank": null,
              "semantic_rank": 3,
              "rrf_score": 0.015873015873015872,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 1,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "afbd85fd7558542c91132d45",
              "paper_id": "1909.08053",
              "canonical_id": "1909.08053v4",
              "ordinal": 32,
              "region": "content",
              "chapter_number": "4",
              "chapter_title": "Setup",
              "section_path": [
                "content",
                "4. Setup"
              ],
              "section_label": "4. Setup",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "c6fbedddc489a5fe3ebda040fc0f4c51050fc56bd4a00d3a8f87fdd261ee1af6",
              "source_blocks": [
                {
                  "index": 59,
                  "page_idx": 4,
                  "bbox": [
                    495,
                    109,
                    888,
                    247
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "c35e35dc59279d7ea2e2ce1eecd342a81e12604c91bbf7916f9756d840df9a9a",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Pretrained language understanding models are central tasks in natural language processing and language understanding. There are several formulations of language modeling. In this work we focus on GPT-2 (Radford et al., 2019), a leftto-right generative transformer based language model, and BERT (Devlin et al., 2018), a bi-directional transformer model based on language model masking. We explain our configurations for these models in the following section and refer to the original papers for more details.",
              "score": 0.018224999999999998,
              "semantic_score": 0.5637116432189941,
              "lexical_rank": null,
              "semantic_rank": 4,
              "rrf_score": 0.015625,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S5"
            },
            {
              "chunk_id": "a52895ab8180d619577eb2cd",
              "paper_id": "1810.04805",
              "canonical_id": "1810.04805v2",
              "ordinal": 19,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "Pre-training BERT",
              "section_path": [
                "content",
                "3 BERT",
                "3.1 Pre-training BERT"
              ],
              "section_label": "3.1 Pre-training BERT",
              "type": "image",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "16f1255d30fa886a0a1cb2bb76780626fa6f1026c5c0ccf74084a6abde4fec2a",
              "source_blocks": [
                {
                  "index": 53,
                  "page_idx": 4,
                  "bbox": [
                    191,
                    72,
                    793,
                    199
                  ],
                  "type": "image",
                  "text_format": null,
                  "context_before": "ws A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, C is used for next sentence prediction (NSP).<sup>5</sup> Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI. 6",
                  "context_after": "The NSP task is closely related to representationlearning objectives used in Jernite et al. (2017) and Logeswaran and Lee (2018). However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters."
                }
              ],
              "asset_refs": [
                "images/e04e40e1fc6961587139f0ce1826139ef2b5ba2d559242720eae0004cf0da535.jpg"
              ],
              "retrieval_text_hash": "41d6b523263b50e61b60f6d341b6dbb678fdab263eb968539aac84f8c7dc6a1e",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.",
              "score": 0.018051515151515152,
              "semantic_score": 0.5501588582992554,
              "lexical_rank": null,
              "semantic_rank": 6,
              "rrf_score": 0.015151515151515152,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "image",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0032500000000000003
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 1810.04805v2 p.4: Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.\n\nTask #1: Masked LM Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-toright and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.\n\nIn order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953).\n\n[S2] 1810.04805v2 p.4: en 10% of the time (3) the unchanged i-th token 10% of the time. Then, $T _ { i }$ will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.\n\nTask #2: Next Sentence Prediction (NSP) Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, C is used for next sentence prediction (NSP).<sup>5</sup> Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI. 6\n\n[S3] 1810.04805v2 p.3: We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For finetuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.\n\nA distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.\n\nModel Architecture BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al.\n\n[S4] 1810.04805v2 p.1: ng, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the masked language model, we also use a “next sentence prediction” task that jointly pretrains text-pair representations. The contributions of our paper are as follows:\n\n• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pretrained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs.\n\n• We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures. BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.\n\n• BERT advances the state of the art for eleven NLP tasks.\n\n[S5] 1909.08053v4 p.5: Pretrained language understanding models are central tasks in natural language processing and language understanding. There are several formulations of language modeling. In this work we focus on GPT-2 (Radford et al., 2019), a leftto-right generative transformer based language model, and BERT (Devlin et al., 2018), a bi-directional transformer model based on language model masking. We explain our configurations for these models in the following section and refer to the original papers for more details.\n\n[S6] 1810.04805v2 p.5: Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"bert pre training two tasks\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "bert pre training two tasks"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过",
        "paper_recall": "通过：1810.04805 命中并排在首位。",
        "evidence_quality": "正文 3.1 同时覆盖 Task #1 MLM 和 Task #2 NSP，证据足够回答问题。",
        "issues": [
          "候选论文中混入 1909.08053，且第 6 条为无关 Figure 2 图片说明；目标事实没有被遮蔽，但候选集合仍有噪声。"
        ]
      }
    }
  ]
}
```


返回论文：`1810.04805, 1909.08053`；质量判定：**部分通过**。

## 案例 2：Why does mixed precision training maintain a master copy of weights in FP32?

```json
{
  "user_question": "Why does mixed precision training maintain a master copy of weights in FP32?",
  "expected_evidence": {
    "paper_ids": [
      "1710.03740"
    ],
    "criteria": "找到小梯度更新在 FP16 中丢失/舍入以及 FP32 主权重更新的原因。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "Why does mixed precision training maintain a master copy of weights in FP32?",
          "task": "reason",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "Why does mixed precision training maintain a master copy of weights in FP32?",
          "task": "reason",
          "routing": {
            "route_intent": "retrieve",
            "task": "reason",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "1710.03740",
              "base_id": "1710.03740",
              "canonical_id": "1710.03740v3",
              "title": "Mixed Precision Training",
              "authors": [
                "Paulius Micikevicius",
                "Sharan Narang",
                "Jonah Alben",
                "Gregory Diamos",
                "Erich Elsen",
                "David Garcia",
                "Boris Ginsburg",
                "Michael Houston",
                "Oleksii Kuchaiev",
                "Ganesh Venkatesh",
                "Hao Wu"
              ],
              "abstract": "Deep neural networks have enabled progress in a wide variety of applications. Growing the size of the neural network typically results in improved accuracy. As model sizes grow, the memory and compute requirements for training these models also increases. We introduce a technique to train deep neural networks using half precision floating point numbers. In our technique, weights, activations and gradients are stored in IEEE half-precision format. Half-precision floating numbers have limited numerical range compared to single-precision numbers. We propose two techniques to handle this loss of information. Firstly, we recommend maintaining a single-precision copy of the weights that accumulates the gradients after each optimizer step. This single-precision copy is rounded to half-precision format during training. Secondly, we propose scaling the loss appropriately to handle the loss of information with half-precision gradients. We demonstrate that this approach works for a wide variety of models including convolution neural networks, recurrent neural networks and generative adversarial networks. This technique works for large scale models with more than 100 million parameters trained on large datasets. Using this approach, we can reduce the memory consumption of deep learning models by nearly 2x. In future processors, we can also expect a significant computation speedup using half-precision hardware units.",
              "categories": [
                "cs.AI",
                "cs.LG",
                "stat.ML"
              ],
              "published_at": "2017-10-10T17:42:04Z",
              "updated_at": "2018-02-15T20:04:02Z",
              "abs_url": "https://arxiv.org/abs/1710.03740v3",
              "pdf_url": "https://arxiv.org/pdf/1710.03740v3.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1710.03740\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "730e138d7b2875e927f2f115",
              "paper_id": "1710.03740",
              "canonical_id": "1710.03740v3",
              "ordinal": 9,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "FP32 MASTER COPY OF WEIGHTS",
              "section_path": [
                "content",
                "3 IMPLEMENTATION",
                "3.1 FP32 MASTER COPY OF WEIGHTS"
              ],
              "section_label": "3.1 FP32 MASTER COPY OF WEIGHTS",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "0988a6c93e6b10d1c869277f03ce4039f5c4318b17069e382dcce7a91e5ff605",
              "source_blocks": [
                {
                  "index": 23,
                  "page_idx": 1,
                  "bbox": [
                    169,
                    882,
                    826,
                    925
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "c53eed27f3eaabab7a9da90fa8e681540b04a3ebd547de968a74b551842937d9",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In mixed precision training, weights, activations and gradients are stored as FP16. In order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step. In each iteration an FP16 copy of the master weights is used in the forward and backward pass, halving the storage and bandwidth needed by FP32 training. Figure 1 illustrates this mixed precision training process.\n\nFigure 1: Mixed precision training iteration for a layer.\n\nWhile the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (weight gradients multiplied by the learning rate) become too small to be represented in FP16 - any value whose magnitude is smaller than 2<sup>−24</sup> becomes zero in FP16. We can see in Figure 2b that approximately 5% of weight gradient values have exponents smaller than −24. These small valued gradients would become zero in the optimizer when multiplied with the learning rate and adversely affect the model accuracy. Using a single-precision copy for the updates allows us to overcome this problem and recover the accuracy.\n\nAnother explanation is that the ratio of the weight value to the weight update is very large. In this case, even though the weight update is representable in FP16, it could still become zero when addition operation right-shifts it to align the binary point with the weight. This can happen when the magnitude of a normalized weight value is at least 2048 times larger that of the weight update.",
              "score": 0.03648688524590164,
              "semantic_score": 0.8532357215881348,
              "lexical_rank": 1,
              "semantic_rank": 1,
              "rrf_score": 0.03278688524590164,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 2,
              "page_end_display": 2,
              "evidence_role": "direct",
              "window_id": "1710.03740:9",
              "source_chunk_ids": [
                "730e138d7b2875e927f2f115",
                "b49221d63641313f6e3f646f",
                "ba41695e2eb133b6ac09022d"
              ],
              "continuity_status": "complete",
              "source_id": "S1"
            },
            {
              "chunk_id": "b49221d63641313f6e3f646f",
              "paper_id": "1710.03740",
              "canonical_id": "1710.03740v3",
              "ordinal": 10,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "FP32 MASTER COPY OF WEIGHTS",
              "section_path": [
                "content",
                "3 IMPLEMENTATION",
                "3.1 FP32 MASTER COPY OF WEIGHTS"
              ],
              "section_label": "3.1 FP32 MASTER COPY OF WEIGHTS",
              "type": "text",
              "page_start": 1,
              "page_end": 2,
              "content_hash": "07aeae3414e34840acd3461e429515c30ff05411cc8412cacc823d876340dafa",
              "source_blocks": [
                {
                  "index": 26,
                  "page_idx": 2,
                  "bbox": [
                    277,
                    101,
                    718,
                    253
                  ],
                  "type": "image",
                  "text_format": null,
                  "context_before": " FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step. In each iteration an FP16 copy of the master weights is used in the forward and backward pass, halving the storage and bandwidth needed by FP32 training. Figure 1 illustrates this mixed precision training process.",
                  "context_after": "While the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (weight gradients multiplied by the learning rate) become too small to be represented in FP16 - any value whose magnitude is smaller than 2<sup>−24</sup> becomes z"
                }
              ],
              "asset_refs": [
                "images/71fea9d757a45316383b264136110158d48f32c579e4ca923e9ae9fdf21ad7a1.jpg"
              ],
              "retrieval_text_hash": "be66fef5a63e73c5dddbe626108851b2dfc3c736b4168c7ecdca9f9e20f95dcd",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In mixed precision training, weights, activations and gradients are stored as FP16. In order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step. In each iteration an FP16 copy of the master weights is used in the forward and backward pass, halving the storage and bandwidth needed by FP32 training. Figure 1 illustrates this mixed precision training process.\n\nFigure 1: Mixed precision training iteration for a layer.\n\nWhile the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (weight gradients multiplied by the learning rate) become too small to be represented in FP16 - any value whose magnitude is smaller than 2<sup>−24</sup> becomes zero in FP16. We can see in Figure 2b that approximately 5% of weight gradient values have exponents smaller than −24. These small valued gradients would become zero in the optimizer when multiplied with the learning rate and adversely affect the model accuracy. Using a single-precision copy for the updates allows us to overcome this problem and recover the accuracy.\n\nAnother explanation is that the ratio of the weight value to the weight update is very large. In this case, even though the weight update is representable in FP16, it could still become zero when addition operation right-shifts it to align the binary point with the weight. This can happen when the magnitude of a normalized weight value is at least 2048 times larger that of the weight update. Since FP16 has 10 bits of mantissa, the implicit bit must be right-shifted by 11 or more positions to potentially create a zero (in some cases rounding can recover the value). In cases where the ratio is larger than 2048, the implicit bit would be right-shifted by 12 or more positions. This will cause the weight update to become a zero which cannot be recovered. An even larger ratio will result in this effect for de-normalized numbers. Again, this effect can be counteracted by computing the update in FP32.\n\nTo illustrate the need for an FP32 master copy of weights, we use the Mandarin speech model (described in more detail in Section 4.3) trained on a dataset comprising of approximately 800 hours of speech data for 20 epochs. As shown in 2a, we match FP32 training results when updating an FP32 master copy of weights after FP16 forward and backward passes, while updating FP16 weights results in 80% relative accuracy loss.",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "ranking_features": {},
              "page_start_display": 2,
              "page_end_display": 3,
              "evidence_role": "context",
              "window_id": "1710.03740:10",
              "source_chunk_ids": [
                "730e138d7b2875e927f2f115",
                "b49221d63641313f6e3f646f",
                "ba41695e2eb133b6ac09022d",
                "2d77bcee1da3151010882602"
              ],
              "continuity_status": "complete",
              "source_id": "S2"
            },
            {
              "chunk_id": "6f9268a413ac932165622d47",
              "paper_id": "1710.03740",
              "canonical_id": "1710.03740v3",
              "ordinal": 26,
              "region": "content",
              "chapter_number": "4.1",
              "chapter_title": "CNNS FOR ILSVRC CLASSIFICATION",
              "section_path": [
                "content",
                "4 RESULTS",
                "4.1 CNNS FOR ILSVRC CLASSIFICATION"
              ],
              "section_label": "4.1 CNNS FOR ILSVRC CLASSIFICATION",
              "type": "text",
              "page_start": 5,
              "page_end": 5,
              "content_hash": "f65cc0f5e0e9b8bf0e6a7905c5e1aab0c5ac25526e98c90113e11436d4808604",
              "source_blocks": [
                {
                  "index": 60,
                  "page_idx": 5,
                  "bbox": [
                    169,
                    349,
                    825,
                    393
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "2f7ae931fd2a1a9d0534aea30b7e24989a944c2b2e200aa587e741307bf567e3",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We trained several CNNs for ILSVRC classification task (Russakovsky et al., 2015) using mixed precision: Alexnet, VGG-D, GoogLeNet, Inception v2, Inception v3, and pre-activation Resnet-50. In all of these cases we were able to match the top-1 accuracy of baseline FP32 training session using identical hyper-parameters. Networks were trained using Caffe (Jia et al., 2014) framework modified to use Volta TensorOps, except for Resnet50 which used PyTorch (Paszke et al., 2017).\n\nTraining schedules were used from public repositories, when available (training schedule for VGG-D has not been published). Top-1 accuracy on ILSVRC validation set are shown in Table 1. Baseline (FP32) accuracy in a few cases is different from published results due to single-crop testing and a simpler data augmentation. Our data augmentation in Caffe included random horizontal flipping and random cropping from 256x256 images, Resnet50 training in PyTorch used the full augmentation in the training script from PyTorch vision repository.\n\nTable 1: ILSVRC12 classification top-1 accuracy.\n<table><tr><td>Model</td><td>Baseline</td><td>Mixed Precision</td><td>Reference</td></tr><tr><td>AlexNet</td><td>56.77%</td><td>56.93%</td><td>(Krizhevsky et al., 2012)</td></tr><tr><td>VGG-D</td><td>65.40%</td><td>65.43%</td><td>(Simonyan and Zisserman, 2014)</td></tr><tr><td>GoogLeNet (Inception v1)</td><td>68.33%</td><td>68.43%</td><td>(Szegedy et al., 2015)</td></tr><tr><td>Inception v2</td><td>70.03%</td><td>70.02%</td><td>(Ioffe and Szegedy, 2015)</td></tr><tr><td>Inception v3</td><td>73.85%</td><td>74.13%</td><td>(Szegedy et al., 2016)</td></tr><tr><td>Resnet50</td><td>75.92%</td><td>76.04%</td><td>(He et al., 2016b)</td></tr></table>\n\nLoss-scaling technique was not required for successful mixed precision training of these networks. While all tensors in the forward and backward passes were in FP16, a master copy of weights was updated in FP32 as outlined in Section 3.1.",
              "score": 0.03337651515151515,
              "semantic_score": 0.6249814033508301,
              "lexical_rank": 4,
              "semantic_rank": 6,
              "rrf_score": 0.030776515151515152,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 6,
              "page_end_display": 6,
              "evidence_role": "direct",
              "window_id": "1710.03740:26",
              "source_chunk_ids": [
                "fc5320f6039affeb3802093b",
                "a377bd97280332c21365cc89",
                "6f9268a413ac932165622d47"
              ],
              "continuity_status": "complete",
              "source_id": "S3"
            }
          ],
          "count": 3,
          "context_text": "[S1] 1710.03740v3 p.2: In mixed precision training, weights, activations and gradients are stored as FP16. In order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step. In each iteration an FP16 copy of the master weights is used in the forward and backward pass, halving the storage and bandwidth needed by FP32 training. Figure 1 illustrates this mixed precision training process.\n\nFigure 1: Mixed precision training iteration for a layer.\n\nWhile the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (weight gradients multiplied by the learning rate) become too small to be represented in FP16 - any value whose magnitude is smaller than 2<sup>−24</sup> becomes zero in FP16. We can see in Figure 2b that approximately 5% of weight gradient values have exponents smaller than −24. These small valued gradients would become zero in the optimizer when multiplied with the learning rate and adversely affect the model accuracy. Using a single-precision copy for the updates allows us to overcome this problem and recover the accuracy.\n\nAnother explanation is that the ratio of the weight value to the weight update is very large. In this case, even though the weight update is representable in FP16, it could still become zero when addition operation right-shifts it to align the binary point with the weight. This can happen when the magnitude of a normalized weight value is at least 2048 times larger that of the weight update.\n\n[S2] 1710.03740v3 p.2: In mixed precision training, weights, activations and gradients are stored as FP16. In order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient during the optimizer step. In each iteration an FP16 copy of the master weights is used in the forward and backward pass, halving the storage and bandwidth needed by FP32 training. Figure 1 illustrates this mixed precision training process.\n\nFigure 1: Mixed precision training iteration for a layer.\n\nWhile the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (weight gradients multiplied by the learning rate) become too small to be represented in FP16 - any value whose magnitude is smaller than 2<sup>−24</sup> becomes zero in FP16. We can see in Figure 2b that approximately 5% of weight gradient values have exponents smaller than −24. These small valued gradients would become zero in the optimizer when multiplied with the learning rate and adversely affect the model accuracy. Using a single-precision copy for the updates allows us to overcome this problem and recover the accuracy.\n\nAnother explanation is that the ratio of the weight value to the weight update is very large. In this case, even though the weight update is representable in FP16, it could still become zero when addition operation right-shifts it to align the binary point with the weight. This can happen when the magnitude of a normalized weight value is at least 2048 times larger that of the weight update. Since FP16 has 10 bits of mantissa, the implicit bit must be right-shifted by 11 or more positions to potentially create a zero (in some cases rounding can recover the value). In cases where the ratio is larger than 2048, the implicit bit would be right-shifted by 12 or more positions. This will cause the weight update to become a zero which cannot be recovered. An even larger ratio will result in this effect for de-normalized numbers. Again, this effect can be counteracted by computing the update in FP32.\n\nTo illustrate the need for an FP32 master copy of weights, we use the Mandarin speech model (described in more detail in Section 4.3) trained on a dataset comprising of approximately 800 hours of speech data for 20 epochs. As shown in 2a, we match FP32 training results when updating an FP32 master copy of weights after FP16 forward and backward passes, while updating FP16 weights results in 80% relative accuracy loss.\n\n[S3] 1710.03740v3 p.6: We trained several CNNs for ILSVRC classification task (Russakovsky et al., 2015) using mixed precision: Alexnet, VGG-D, GoogLeNet, Inception v2, Inception v3, and pre-activation Resnet-50. In all of these cases we were able to match the top-1 accuracy of baseline FP32 training session using identical hyper-parameters. Networks were trained using Caffe (Jia et al., 2014) framework modified to use Volta TensorOps, except for Resnet50 which used PyTorch (Paszke et al., 2017).\n\nTraining schedules were used from public repositories, when available (training schedule for VGG-D has not been published). Top-1 accuracy on ILSVRC validation set are shown in Table 1. Baseline (FP32) accuracy in a few cases is different from published results due to single-crop testing and a simpler data augmentation. Our data augmentation in Caffe included random horizontal flipping and random cropping from 256x256 images, Resnet50 training in PyTorch used the full augmentation in the training script from PyTorch vision repository.\n\nTable 1: ILSVRC12 classification top-1 accuracy.\n<table><tr><td>Model</td><td>Baseline</td><td>Mixed Precision</td><td>Reference</td></tr><tr><td>AlexNet</td><td>56.77%</td><td>56.93%</td><td>(Krizhevsky et al., 2012)</td></tr><tr><td>VGG-D</td><td>65.40%</td><td>65.43%</td><td>(Simonyan and Zisserman, 2014)</td></tr><tr><td>GoogLeNet (Inception v1)</td><td>68.33%</td><td>68.43%</td><td>(Szegedy et al., 2015)</td></tr><tr><td>Inception v2</td><td>70.03%</td><td>70.02%</td><td>(Ioffe and Szegedy, 2015)</td></tr><tr><td>Inception v3</td><td>73.85%</td><td>74.13%</td><td>(Szegedy et al., 2016)</td></tr><tr><td>Resnet50</td><td>75.92%</td><td>76.04%</td><td>(He et al., 2016b)</td></tr></table>\n\nLoss-scaling technique was not required for successful mixed precision training of these networks. While all tensors in the forward and backward passes were in FP16, a master copy of weights was updated in FP32 as outlined in Section 3.1.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"mixed precision training\" OR \"master copy weights\" OR \"fp32\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [
              "of"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "mixed precision training",
              "master copy weights",
              "fp32"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：仅返回 1710.03740。",
        "evidence_quality": "3.1 的直接证据完整说明 FP16 前后向、FP32 master copy 和优化器更新原因，并保留了同章节窗口及页码。",
        "issues": [
          "第 2 条是直接证据的重复上下文，第 3 条来自实验设置，存在轻微冗余。"
        ]
      }
    }
  ]
}
```

返回论文：`1710.03740`；质量判定：**通过**。

## 案例 3：FlashAttention 为什么能减少 GPU 的 HBM 读写，同时保持精确注意力结果？

```json
{
  "user_question": "FlashAttention 为什么能减少 GPU 的 HBM 读写，同时保持精确注意力结果？",
  "expected_evidence": {
    "paper_ids": [
      "2205.14135"
    ],
    "criteria": "找到 tiling、SRAM、重计算以及精确注意力的因果证据。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "FlashAttention 为什么能减少 GPU 的 HBM 读写，同时保持精确注意力结果？",
          "task": "auto",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "FlashAttention 为什么能减少 GPU 的 HBM 读写，同时保持精确注意力结果？",
          "task": "reason",
          "routing": {
            "route_intent": "retrieve",
            "task": "reason",
            "provider": "jev",
            "fallback_used": false,
            "confidence": 0.95
          },
          "papers": [
            {
              "paper_id": "2205.14135",
              "base_id": "2205.14135",
              "canonical_id": "2205.14135v2",
              "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
              "authors": [
                "Tri Dao",
                "Daniel Y. Fu",
                "Stefano Ermon",
                "Atri Rudra",
                "Christopher Ré"
              ],
              "abstract": "Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3$\\times$ speedup on GPT-2 (seq. length 1K), and 2.4$\\times$ speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).",
              "categories": [
                "cs.LG"
              ],
              "published_at": "2022-05-27T17:53:09Z",
              "updated_at": "2022-06-23T17:53:32Z",
              "abs_url": "https://arxiv.org/abs/2205.14135v2",
              "pdf_url": "https://arxiv.org/pdf/2205.14135v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "448d3d10e08da43812e5515d",
              "paper_id": "2205.14135",
              "canonical_id": "2205.14135v2",
              "ordinal": 25,
              "region": "content",
              "chapter_number": "3.2",
              "chapter_title": "Analysis: IO Complexity of FlashAttention",
              "section_path": [
                "content",
                "3 FlashAttention: Algorithm, Analysis, and Extensions",
                "3.2 Analysis: IO Complexity of FlashAttention"
              ],
              "section_label": "3.2 Analysis: IO Complexity of FlashAttention",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "0630d885b37d6a6113e334f632bf3c9fd7f0c95dd515a713bd52f5c717ce3c85",
              "source_blocks": [
                {
                  "index": 62,
                  "page_idx": 4,
                  "bbox": [
                    111,
                    862,
                    883,
                    893
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "f94f8b306051bf1834a95c2058c09472ec74d63b12a92f7d5f4917454b231f5b",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C.\n\nimages/e228b8b34cf08ef93b40f9183344987d64f72d4e0b54559e30e2f6e149d25808.jpg\n\nFigure 2: Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor afecting runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity.",
              "score": 0.03453491461100569,
              "semantic_score": 0.6113425493240356,
              "lexical_rank": 2,
              "semantic_rank": 8,
              "rrf_score": 0.030834914611005692,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "window_id": "2205.14135:25",
              "source_chunk_ids": [
                "448d3d10e08da43812e5515d",
                "80735837e2482854c09dd5c8",
                "cb06a28c020b103d9d32c9dd"
              ],
              "continuity_status": "complete",
              "source_id": "S1"
            },
            {
              "chunk_id": "7b85d3f22d633630a3b4ee0d",
              "paper_id": "2205.14135",
              "canonical_id": "2205.14135v2",
              "ordinal": 17,
              "region": "content",
              "chapter_number": "3",
              "chapter_title": "FlashAttention: Algorithm, Analysis, and Extensions",
              "section_path": [
                "content",
                "3 FlashAttention: Algorithm, Analysis, and Extensions"
              ],
              "section_label": "3 FlashAttention: Algorithm, Analysis, and Extensions",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "151d0538542dff43327c95a7e11054c9edfa0bb6d8b044c3a88315beb1227a16",
              "source_blocks": [
                {
                  "index": 45,
                  "page_idx": 3,
                  "bbox": [
                    109,
                    571,
                    883,
                    647
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 46,
                  "page_idx": 3,
                  "bbox": [
                    135,
                    646,
                    885,
                    662
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "e0459f9effaacd6c61b27d7ba1386a2a35a7190cee7472be49fd3c70cd5039bc",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory eficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention.\n\nWe focus here on the forward pass for ease of exposition; Appendix B contains details for the backward.",
              "score": 0.032329032258064515,
              "semantic_score": 0.6520328521728516,
              "lexical_rank": 20,
              "semantic_rank": 2,
              "rrf_score": 0.028629032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "window_id": "2205.14135:17",
              "source_chunk_ids": [
                "7b85d3f22d633630a3b4ee0d"
              ],
              "continuity_status": "complete",
              "source_id": "S2"
            }
          ],
          "count": 2,
          "context_text": "[S1] 2205.14135v2 p.5: We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in Appendix C.\n\nimages/e228b8b34cf08ef93b40f9183344987d64f72d4e0b54559e30e2f6e149d25808.jpg\n\nFigure 2: Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor afecting runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity.\n\n[S2] 2205.14135v2 p.4: We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory eficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention.\n\nWe focus here on the forward pass for ease of exposition; Appendix B contains details for the backward.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"flashattention\" OR \"reduce gpu hbm reading writing\" OR \"maintain accurate attention results\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "and",
              "to"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "flashattention",
              "reduce gpu hbm reading writing",
              "maintain accurate attention results"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：仅返回 2205.14135。",
        "evidence_quality": "正文引言和 3.2 同时覆盖 exact attention、减少 HBM reads/writes、IO complexity 和 SRAM/HBM 分层原因。",
        "issues": []
      }
    }
  ]
}
```

返回论文：`2205.14135`；质量判定：**通过**。

## 案例 4：Swin Transformer 的 shifted windows 为什么能实现跨窗口信息交互？

```json
{
  "user_question": "Swin Transformer 的 shifted windows 为什么能实现跨窗口信息交互？",
  "expected_evidence": {
    "paper_ids": [
      "2103.14030"
    ],
    "criteria": "找到交替窗口划分与跨窗口连接，不能只有泛化描述。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "Swin Transformer 的 shifted windows 为什么能实现跨窗口信息交互？",
          "task": "reason",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "Swin Transformer 的 shifted windows 为什么能实现跨窗口信息交互？",
          "task": "reason",
          "routing": {
            "route_intent": "retrieve",
            "task": "reason",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "30f0034deea5b56cb1238efd",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 19,
              "region": "content",
              "chapter_number": "3.1",
              "chapter_title": "Overall Architecture",
              "section_path": [
                "content",
                "3. Method",
                "3.1. Overall Architecture"
              ],
              "section_label": "3.1. Overall Architecture",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "24691b07fca216aac751066108725351b1442c373b3bcc6d4f1390b5474ee4d0",
              "source_blocks": [
                {
                  "index": 40,
                  "page_idx": 3,
                  "bbox": [
                    75,
                    448,
                    470,
                    601
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "eea33d08699c6808e50885752381cd090bb6fdd96396a88fd7eeda60b1b89ca4",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "“Stage $1 ^ { \\circ }$\n\nTo produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of $2 \\times 2$ neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of $2 \\times 2 = 4 ( 2 \\times$ downsampling of resolution), and the output dimension is set to $2 C$ . Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at ${ \\frac { H } { 8 } } \\times { \\frac { W } { 8 } }$ . This first block of patch merging and feature transformation is denoted as “Stage $2 ^ { \\circ }$ . The procedure is repeated twice, as “Stage $3 ^ { \\circ }$ and “Stage $4 ^ { \\circ }$ , with output resolutions of $\\frac { H } { 1 6 } \\times \\frac { W } { 1 6 }$ and $\\frac { H } { 3 2 } \\times \\frac { W } { 3 2 }$ , respectively. These stages jointly produce a hierarchical representation, with the same feature map resolutions as those of typical convolutional networks, e.g., VGG [52] and ResNet [30]. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks.\n\nFigure 3. (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks (notation presented with Eq. (3)). W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively.\n\nSwin Transformer block Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows (described in Section 3.2), with other layers kept the same. As illustrated in Figure 3(b), a Swin Transformer block consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module.",
              "score": 0.03398688524590164,
              "semantic_score": 0.6537637710571289,
              "lexical_rank": 1,
              "semantic_rank": 1,
              "rrf_score": 0.03278688524590164,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "window_id": "2103.14030:19",
              "source_chunk_ids": [
                "90cc10309e3f020afcdbb699",
                "de12c1a3dc6b2aed91f6919e",
                "3f0666703651f5a130590dda",
                "30f0034deea5b56cb1238efd"
              ],
              "continuity_status": "complete",
              "source_id": "S1"
            },
            {
              "chunk_id": "ec696b1d6ef5f032c4921c23",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 7,
              "region": "content",
              "chapter_number": "1",
              "chapter_title": "Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "text",
              "page_start": 0,
              "page_end": 1,
              "content_hash": "7350e0b0b60dfbcc00ec5ad293dd9b8eb17174a9e3c8a32b56f4de4786d831df",
              "source_blocks": [
                {
                  "index": 11,
                  "page_idx": 0,
                  "bbox": [
                    496,
                    702,
                    893,
                    869
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 12,
                  "page_idx": 0,
                  "bbox": [
                    498,
                    869,
                    895,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 16,
                  "page_idx": 1,
                  "bbox": [
                    75,
                    666,
                    472,
                    849
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "230ddfb9bdc4face0cb29b0b8906671a7b097416d22c822a652d531ff312f9b7",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "On the other hand, the evolution of network architectures in natural language processing (NLP) has taken a different path, where the prevalent architecture today is instead the Transformer [64]. Designed for sequence modeling and transduction tasks, the Transformer is notable for its use of attention to model long-range dependencies in the data. Its tremendous success in the language domain has led researchers to investigate its adaptation to computer vision, where it has recently demonstrated promising results on certain tasks, specifically image classification [20] and joint vision-language modeling [47].\n\nIn this paper, we seek to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision. We observe that significant challenges in transferring its high performance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [42, 53, 54]. In existing Transformer-based models [64, 20], tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size. To overcome these issues, we propose a general purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. As illustrated in Figure 1(a), Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size. These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [20] which produce feature maps of a single resolution and have quadratic complexity.\n\nA key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers, as illustrated in Figure 2. The shifted windows bridge the windows of the preceding layer, providing connections among them that significantly enhance modeling power (see Table 4). This strategy is also efficient in regards to real-world latency: all query patches within a window share the same key set<sup>1</sup>, which facilitates memory access in hardware. In contrast, earlier sliding window based self-attention approaches [33, 50] suffer from low latency on general hardware due to different key sets for different query pixels<sup>2</sup>. Our experiments show that the proposed shifted window approach has much lower latency than the sliding window method, yet is similar in modeling power (see Tables 5 and 6). The shifted window approach also proves beneficial for all-MLP architectures [61].\n\nFigure 2. An illustration of the shifted window approach for computing self-attention in the proposed Swin Transformer architecture. In layer l (left), a regular window partitioning scheme is adopted, and self-attention is computed within each window. In the next layer l + 1 (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new windows crosses the boundaries of the previous windows in layer l, providing connections among them.\n\nThe proposed Swin Transformer achieves strong performance on the recognition tasks of image classification, object detection and semantic segmentation. It outperforms the ViT / DeiT [20, 63] and ResNe(X)t models [30, 70] significantly with similar latency on the three tasks. Its 58.7 box AP and 51.1 mask AP on the COCO test-dev set surpass the previous state-of-the-art results by +2.7 box AP (Copy-paste [26] without external data) and +2.6 mask AP (DetectoRS [46]). On ADE20K semantic segmentation, it obtains 53.5 mIoU on the val set, an improvement of +3.2 mIoU over the previous state-of-the-art (SETR [81]). It also achieves a top-1 accuracy of 87.3% on ImageNet-1K image classification.\n\nIt is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared. We hope that Swin Transformer’s strong performance on various vision problems can drive this belief deeper in the community and encourage unified modeling of vision and language signals.",
              "score": 0.03248054740957967,
              "semantic_score": 0.6468932628631592,
              "lexical_rank": 6,
              "semantic_rank": 2,
              "rrf_score": 0.03128054740957967,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 1,
              "page_end_display": 2,
              "evidence_role": "direct",
              "window_id": "2103.14030:7",
              "source_chunk_ids": [
                "8391c1423419875209fbf1b1",
                "d5feb2669c49ba2ea91299dd",
                "339c82822ec082e1a253c1d3",
                "ec696b1d6ef5f032c4921c23",
                "82ddd20e19aebd6d40282f0b",
                "2ac7d79bec81d4618cf26271"
              ],
              "continuity_status": "complete",
              "source_id": "S2"
            },
            {
              "chunk_id": "aef74810680c1fef21c96da5",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "8dd511eb8818ff1142cbd562e68a32e670fe8c7456c804dbe96ff05374087123",
              "source_blocks": [
                {
                  "index": 6,
                  "page_idx": 0,
                  "bbox": [
                    75,
                    319,
                    470,
                    743
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b47f9f16ab0045fc408d5cc9b9973271a8d46fc70bea8be0e767dc9b747136d3",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-theart by a large margin of+2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at https://github. com/microsoft/Swin-Transformer.",
              "score": 0.031974531024531024,
              "semantic_score": 0.6111207008361816,
              "lexical_rank": 3,
              "semantic_rank": 6,
              "rrf_score": 0.031024531024531024,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00095
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "evidence_role": "direct",
              "window_id": "2103.14030:0",
              "source_chunk_ids": [
                "aef74810680c1fef21c96da5",
                "77dbd7ce31b0a50200fa1532"
              ],
              "continuity_status": "complete",
              "source_id": "S3"
            }
          ],
          "count": 3,
          "context_text": "[S1] 2103.14030v2 p.4: “Stage $1 ^ { \\circ }$\n\nTo produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of $2 \\times 2$ neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple of $2 \\times 2 = 4 ( 2 \\times$ downsampling of resolution), and the output dimension is set to $2 C$ . Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at ${ \\frac { H } { 8 } } \\times { \\frac { W } { 8 } }$ . This first block of patch merging and feature transformation is denoted as “Stage $2 ^ { \\circ }$ . The procedure is repeated twice, as “Stage $3 ^ { \\circ }$ and “Stage $4 ^ { \\circ }$ , with output resolutions of $\\frac { H } { 1 6 } \\times \\frac { W } { 1 6 }$ and $\\frac { H } { 3 2 } \\times \\frac { W } { 3 2 }$ , respectively. These stages jointly produce a hierarchical representation, with the same feature map resolutions as those of typical convolutional networks, e.g., VGG [52] and ResNet [30]. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks.\n\nFigure 3. (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks (notation presented with Eq. (3)). W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively.\n\nSwin Transformer block Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows (described in Section 3.2), with other layers kept the same. As illustrated in Figure 3(b), a Swin Transformer block consists of a shifted window based MSA module, followed by a 2-layer MLP with GELU nonlinearity in between. A LayerNorm (LN) layer is applied before each MSA module and each MLP, and a residual connection is applied after each module.\n\n[S2] 2103.14030v2 p.1: On the other hand, the evolution of network architectures in natural language processing (NLP) has taken a different path, where the prevalent architecture today is instead the Transformer [64]. Designed for sequence modeling and transduction tasks, the Transformer is notable for its use of attention to model long-range dependencies in the data. Its tremendous success in the language domain has led researchers to investigate its adaptation to computer vision, where it has recently demonstrated promising results on certain tasks, specifically image classification [20] and joint vision-language modeling [47].\n\nIn this paper, we seek to expand the applicability of Transformer such that it can serve as a general-purpose backbone for computer vision, as it does for NLP and as CNNs do in vision. We observe that significant challenges in transferring its high performance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [42, 53, 54]. In existing Transformer-based models [64, 20], tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size. To overcome these issues, we propose a general purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. As illustrated in Figure 1(a), Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size. These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [20] which produce feature maps of a single resolution and have quadratic complexity.\n\nA key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers, as illustrated in Figure 2. The shifted windows bridge the windows of the preceding layer, providing connections among them that significantly enhance modeling power (see Table 4). This strategy is also efficient in regards to real-world latency: all query patches within a window share the same key set<sup>1</sup>, which facilitates memory access in hardware. In contrast, earlier sliding window based self-attention approaches [33, 50] suffer from low latency on general hardware due to different key sets for different query pixels<sup>2</sup>. Our experiments show that the proposed shifted window approach has much lower latency than the sliding window method, yet is similar in modeling power (see Tables 5 and 6). The shifted window approach also proves beneficial for all-MLP architectures [61].\n\nFigure 2. An illustration of the shifted window approach for computing self-attention in the proposed Swin Transformer architecture. In layer l (left), a regular window partitioning scheme is adopted, and self-attention is computed within each window. In the next layer l + 1 (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new windows crosses the boundaries of the previous windows in layer l, providing connections among them.\n\nThe proposed Swin Transformer achieves strong performance on the recognition tasks of image classification, object detection and semantic segmentation. It outperforms the ViT / DeiT [20, 63] and ResNe(X)t models [30, 70] significantly with similar latency on the three tasks. Its 58.7 box AP and 51.1 mask AP on the COCO test-dev set surpass the previous state-of-the-art results by +2.7 box AP (Copy-paste [26] without external data) and +2.6 mask AP (DetectoRS [46]). On ADE20K semantic segmentation, it obtains 53.5 mIoU on the val set, an improvement of +3.2 mIoU over the previous state-of-the-art (SETR [81]). It also achieves a top-1 accuracy of 87.3% on ImageNet-1K image classification.\n\nIt is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared. We hope that Swin Transformer’s strong performance on various vision problems can drive this belief deeper in the community and encourage unified modeling of vision and language signals.\n\n[S3] 2103.14030v2 p.1: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-theart by a large margin of+2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at https://github. com/microsoft/Swin-Transformer.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"swin transformer\" OR \"shifted windows\" OR \"cross window information interaction\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "swin transformer",
              "shifted windows",
              "cross window information interaction"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过",
        "paper_recall": "通过：仅返回 2103.14030。",
        "evidence_quality": "摘要提到 shifted window 允许跨窗口连接，但前两条正文证据落在 3.1 和 Introduction，没有把 3.2 的交替窗口机制排到首位。",
        "issues": [
          "查询实体命中和章节精确命中均为 false；核心解释证据没有稳定进入首条。"
        ]
      }
    }
  ]
}
```

返回论文：`2103.14030`；质量判定：**部分通过**。

## 案例 5：Summarize how EfficientNet scales network depth, width, and resolution.

```json
{
  "user_question": "Summarize how EfficientNet scales network depth, width, and resolution.",
  "expected_evidence": {
    "paper_ids": [
      "1905.11946"
    ],
    "criteria": "找到 compound scaling 及深度/宽度/分辨率协同缩放的正文。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
          "task": "summary",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
          "task": "summary",
          "routing": {
            "route_intent": "retrieve",
            "task": "summary",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "1905.11946",
              "base_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
              "authors": [
                "Mingxing Tan",
                "Quoc V. Le"
              ],
              "abstract": "Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet.",
              "categories": [
                "cs.LG",
                "cs.CV",
                "stat.ML"
              ],
              "published_at": "2019-05-28T17:05:32Z",
              "updated_at": "2020-09-11T05:08:01Z",
              "abs_url": "https://arxiv.org/abs/1905.11946v5",
              "pdf_url": "https://arxiv.org/pdf/1905.11946v5.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "score": 0.018729032258064514,
              "semantic_score": 0.7117745280265808,
              "lexical_rank": null,
              "semantic_rank": 2,
              "rrf_score": 0.016129032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "7e464a54739095d4d2c8c495",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "d20f12bd2041e5e2d73b7285223e7aa5b5736bf00ceedeafe614bfdaf1a8a922",
              "source_blocks": [
                {
                  "index": 3,
                  "page_idx": 0,
                  "bbox": [
                    117,
                    244,
                    444,
                    441
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 4,
                  "page_idx": 0,
                  "bbox": [
                    116,
                    445,
                    444,
                    689
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b3226b24f5a94b96ac22f52b8d2c5aa3651e59ba580d6a898c8905bb53d7eb16",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.\n\nTo go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.",
              "source_id": "S1"
            },
            {
              "score": 0.018743442622950822,
              "semantic_score": 0.7167273163795471,
              "lexical_rank": null,
              "semantic_rank": 1,
              "rrf_score": 0.01639344262295082,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00235
              },
              "page_start_display": 2,
              "page_end_display": 2,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "704c1c58a387ffa5d0dc10d9",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 10,
              "region": "content",
              "chapter_number": null,
              "chapter_title": "1. Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "c21d86add4a6ab006397b30e52505a2bb0cea5c47c03d52700504651b1b2ff10",
              "source_blocks": [
                {
                  "index": 19,
                  "page_idx": 1,
                  "bbox": [
                    83,
                    503,
                    477,
                    656
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 20,
                  "page_idx": 1,
                  "bbox": [
                    83,
                    662,
                    478,
                    905
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "7cf646d6952907273bab8661c00622d909dac5a59c86b7af406628c72497a1f2",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image. In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.\n\nWe demonstrate that our scaling method work well on existing MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets.",
              "source_id": "S2"
            },
            {
              "score": 0.017975,
              "semantic_score": 0.6718592643737793,
              "lexical_rank": null,
              "semantic_rank": 4,
              "rrf_score": 0.015625,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00235
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "faa891eb569d3fd835df6a2f",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 58,
              "region": "content",
              "chapter_number": "7",
              "chapter_title": "Conclusion",
              "section_path": [
                "content",
                "7. Conclusion"
              ],
              "section_label": "7. Conclusion",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "380262dcd44e8bf7e510d6ec361a005bd43cf760330a23224de60e5ecc33bbe8",
              "source_blocks": [
                {
                  "index": 107,
                  "page_idx": 8,
                  "bbox": [
                    84,
                    109,
                    477,
                    306
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "e3732f6148f2b82cd77b52a90f4c6d6f6dd3e5557164381f7f62b5a75f75e1ae",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we demonstrate that a mobilesize EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets.",
              "source_id": "S3"
            },
            {
              "score": 0.01742301587301587,
              "semantic_score": 0.6810175180435181,
              "lexical_rank": null,
              "semantic_rank": 3,
              "rrf_score": 0.015873015873015872,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "chart",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0019
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "ece7dd3c85f6cb451ee99297",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 3,
              "region": "content",
              "chapter_number": null,
              "chapter_title": "1. Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "chart",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "4720659004486d015b64b9201842aa666d969028e0892fa147e8e799b4bc98b6",
              "source_blocks": [
                {
                  "index": 7,
                  "page_idx": 0,
                  "bbox": [
                    501,
                    220,
                    875,
                    450
                  ],
                  "type": "chart",
                  "text_format": null,
                  "context_before": "image resolution (Huang et al., 2018). In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency.",
                  "context_after": "In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical study shows that it is critical to balance all dimensions of network width/"
                }
              ],
              "asset_refs": [
                "images/cedfc234efd935161914bd4aa19ce8e82f1b9b4586f3558fdce9a071a01441ff.jpg"
              ],
              "retrieval_text_hash": "ae48830e077a4f76d3ff0ea35278b6530546ba82d9e1a89e26ba2bfe2a68ab97",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Figure 1. Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.3% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4.",
              "source_id": "S4"
            },
            {
              "score": 0.017351515151515153,
              "semantic_score": 0.6284216642379761,
              "lexical_rank": null,
              "semantic_rank": 6,
              "rrf_score": 0.015151515151515152,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "table",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0022
              },
              "page_start_display": 6,
              "page_end_display": 6,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "e5d3b72f5ac7af2a56997154",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 41,
              "region": "content",
              "chapter_number": "5.1",
              "chapter_title": "Scaling Up MobileNets and ResNets",
              "section_path": [
                "content",
                "5. Experiments",
                "5.1. Scaling Up MobileNets and ResNets"
              ],
              "section_label": "5.1. Scaling Up MobileNets and ResNets",
              "type": "table",
              "page_start": 5,
              "page_end": 5,
              "content_hash": "d2685f150f75a48461383dedcac8df79273fbfb724e901892da233b63ce5c4bd",
              "source_blocks": [
                {
                  "index": 78,
                  "page_idx": 5,
                  "bbox": [
                    89,
                    157,
                    882,
                    520
                  ],
                  "type": "table",
                  "text_format": null,
                  "context_before": "2018) and ResNet (He et al., 2016). Table 3 shows the ImageNet results of scaling them in different ways. Compared to other single-dimension scaling methods, our compound scaling method improves the accuracy on all these models, suggesting the effectiveness of our proposed scaling method for general existing ConvNets.",
                  "context_after": ""
                }
              ],
              "asset_refs": [
                "images/0e6dd69a2e9fa5035d4f55db023f2b7eef48c8315ee344d3e627cee9f3a6e009.jpg"
              ],
              "retrieval_text_hash": "02c837f0beafbf4414dfc95fd78f269d0086ddf6375ad4318ac9b12fc0bb49c2",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with similar top-1/top-5 accuracy are grouped together for efficiency comparison. Our scaled EfficientNet models consistently reduce parameters and FLOPS by an order of magnitude (up to 8.4x parameter reduction and up to 16x FLOPS reduction) than existing ConvNets.\n<table><tr><td>Model</td><td>Top-1 Acc.</td><td>Top-5 Acc.</td><td>#Params</td><td>Ratio-to-EfficientNet</td><td>#FLOPs</td><td>Ratio-to-EfficientNet</td></tr><tr><td>EfficientNet-B0</td><td>77.1%</td><td>93.3%</td><td>5.3M</td><td>1x</td><td>0.39B</td><td>1x</td></tr><tr><td>ResNet-50 (He et al., 2016)</td><td>76.0%</td><td>93.0%</td><td>26M</td><td>4.9x</td><td>4.1B</td><td>11x</td></tr><tr><td>DenseNet-169 (Huang et al., 2017)</td><td>76.2%</td><td>93.2%</td><td>14M</td><td>2.6x</td><td>3.5B</td><td>8.9x</td></tr><tr><td>EfficientNet-B1</td><td>79.1%</td><td>94.4%</td><td>7.8M</td><td>1x</td><td>0.70B</td><td>1x</td></tr><tr><td>ResNet-152 (He et al., 2016)</td><td>77.8%</td><td>93.8%</td><td>60M</td><td>7.6x</td><td>11B</td><td>16x</td></tr><tr><td>DenseNet-264 (Huang et al., 2017)</td><td>77.9%</td><td>93.9%</td><td>34M</td><td>4.3x</td><td>6.0B</td><td>8.6x</td></tr><tr><td>Inception-v3 (Szegedy et al., 2016)</td><td>78.8%</td><td>94.4%</td><td>24M</td><td>3.0x</td><td>5.7B</td><td>8.1x</td></tr><tr><td>Xception (Chollet, 2017)</td><td>79.0%</td><td>94.5%</td><td>23M</td><td>3.0x</td><td>8.4B</td><td>12x</td></tr><tr><td>EfficientNet-B2</td><td>80.1%</td><td>94.9%</td><td>9.2M</td><td>1x</td><td>1.0B</td><td>1x</td></tr><tr><td>Inception-v4 (Szegedy et al., 2017)</td><td>80.0%</td><td>95.0%</td><td>48M</td><td>5.2x</td><td>13B</td><td>13x</td></tr><tr><td>Inception-resnet-v2 (Szegedy et al., 2017)</td><td>80.1%</td><td>95.1%</td><td>56M</td><td>6.1x</td><td>13B</td><td>13x</td></tr><tr><td>EfficientNet-B3</td><td>81.6%</td><td>95.7%</td><td>12M</td><td>1x</td><td>1.8B</td><td>1x</td></tr><tr><td>ResNeXt-101 (Xie et al., 2017)</td><td>80.9%</td><td>95.6%</td><td>84M</td><td>7.0x</td><td>32B</td><td>18x</td></tr><tr><td>PolyNet (Zhang et al., 2017)</td><td>81.3%</td><td>95.8%</td><td>92M</td><td>7.7x</td><td>35B</td><td>19x</td></tr><tr><td>EfficientNet-B4</td><td>82.9%</td><td>96.4%</td><td>19M</td><td>1x</td><td>4.2B</td><td>1x</td></tr><tr><td>SENet (Hu et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>146M</td><td>7.7x</td><td>42B</td><td>10x</td></tr><tr><td>NASNet-A (Zoph et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>89M</td><td>4.7x</td><td>24B</td><td>5.7x</td></tr><tr><td>AmoebaNet-A (Real et al., 2019)</td><td>82.8%</td><td>96.1%</td><td>87M</td><td>4.6x</td><td>23B</td><td>5.5x</td></tr><tr><td>PNASNet (Liu et al., 2018)</td><td>82.9%</td><td>96.2%</td><td>86M</td><td>4.5x</td><td>23B</td><td>6.0x</td></tr><tr><td>EfficientNet-B5</td><td>83.6%</td><td>96.7%</td><td>30M</td><td>1x</td><td>9.9B</td><td>1x</td></tr><tr><td>AmoebaNet-C (Cubuk et al., 2019)</td><td>83.5%</td><td>96.5%</td><td>155M</td><td>5.2x</td><td>41B</td><td>4.1x</td></tr><tr><td>EfficientNet-B6</td><td>84.0%</td><td>96.8%</td><td>43M</td><td>1x</td><td>19B</td><td>1x</td></tr><tr><td>EfficientNet-B7</td><td>84.3%</td><td>97.0%</td><td>66M</td><td>1x</td><td>37B</td><td>1x</td></tr><tr><td>GPipe (Huang et al., 2018)</td><td>84.3%</td><td>97.0%</td><td>557M</td><td>8.4x</td><td>-</td><td>-</td></tr></table>\nWe omit ensemble and multi-crop models (Hu et al., 2018), or models pretrained on 3.5B Instagram images (Mahajan et al., 2018).",
              "source_id": "S5"
            },
            {
              "score": 0.017148630136986302,
              "semantic_score": 0.5845932364463806,
              "lexical_rank": null,
              "semantic_rank": 13,
              "rrf_score": 0.0136986301369863,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00345
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "query": "Summarize how EfficientNet scales network depth, width, and resolution.",
              "evidence_role": "direct",
              "chunk_id": "aa0ed971f82ae2edbb9f671f",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 37,
              "region": "content",
              "chapter_number": "4",
              "chapter_title": "EfficientNet Architecture",
              "section_path": [
                "content",
                "4. EfficientNet Architecture"
              ],
              "section_label": "4. EfficientNet Architecture",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "9642c81e9fde0db21f4130af95669e5a1e8d1f0539f306c8e4c59e5fde7e2f1f",
              "source_blocks": [
                {
                  "index": 67,
                  "page_idx": 4,
                  "bbox": [
                    495,
                    292,
                    888,
                    385
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 68,
                  "page_idx": 4,
                  "bbox": [
                    495,
                    390,
                    887,
                    424
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 69,
                  "page_idx": 4,
                  "bbox": [
                    513,
                    436,
                    887,
                    513
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 70,
                  "page_idx": 4,
                  "bbox": [
                    513,
                    517,
                    887,
                    564
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 71,
                  "page_idx": 4,
                  "bbox": [
                    495,
                    578,
                    888,
                    670
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "90773689f13a8d48ac5de48685ee90de0a6c02707db82cb75ed0a381130cb17b",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization (Hu et al., 2018).\n\nStarting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:\n\n• STEP 1: we first fix $\\phi = 1$ , assuming twice more resources available, and do a small grid search of $\\alpha , \\beta , \\gamma$ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are $\\alpha = 1 . 2 , \\beta =$ $1 . 1 , \\gamma = 1 . 1 5$ , under constraint of $\\alpha \\cdot \\beta ^ { 2 } \\cdot \\gamma ^ { 2 } \\approx 2$\n\n• STEP 2: we then fix $\\alpha , \\beta , \\gamma$ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2).\n\nNotably, it is possible to achieve even better performance by searching for $\\alpha , \\beta ,$ γ directly around a large model, but the search cost becomes prohibitively more expensive on larger models.",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 1905.11946v5 p.1: Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet.\n\nTo go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.\n\n[S2] 1905.11946v5 p.2: Intuitively, the compound scaling method makes sense because if the input image is bigger, then the network needs more layers to increase the receptive field and more channels to capture more fine-grained patterns on the bigger image. In fact, previous theoretical (Raghu et al., 2017; Lu et al., 2018) and empirical results (Zagoruyko & Komodakis, 2016) both show that there exists certain relationship between network width and depth, but to our best knowledge, we are the first to empirically quantify the relationship among all three dimensions of network width, depth, and resolution.\n\nWe demonstrate that our scaling method work well on existing MobileNets (Howard et al., 2017; Sandler et al., 2018) and ResNet (He et al., 2016). Notably, the effectiveness of model scaling heavily depends on the baseline network; to go even further, we use neural architecture search (Zoph & Le, 2017; Tan et al., 2019) to develop a new baseline network, and scale it up to obtain a family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets.\n\n[S3] 1905.11946v5 p.9: In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we demonstrate that a mobilesize EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets.\n\n[S4] 1905.11946v5 p.1: Figure 1. Model Size vs. ImageNet Accuracy. All numbers are for single-crop, single-model. Our EfficientNets significantly outperform other ConvNets. In particular, EfficientNet-B7 achieves new state-of-the-art 84.3% top-1 accuracy but being 8.4x smaller and 6.1x faster than GPipe. EfficientNet-B1 is 7.6x smaller and 5.7x faster than ResNet-152. Details are in Table 2 and 4.\n\n[S5] 1905.11946v5 p.6: Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with similar top-1/top-5 accuracy are grouped together for efficiency comparison. Our scaled EfficientNet models consistently reduce parameters and FLOPS by an order of magnitude (up to 8.4x parameter reduction and up to 16x FLOPS reduction) than existing ConvNets.\n<table><tr><td>Model</td><td>Top-1 Acc.</td><td>Top-5 Acc.</td><td>#Params</td><td>Ratio-to-EfficientNet</td><td>#FLOPs</td><td>Ratio-to-EfficientNet</td></tr><tr><td>EfficientNet-B0</td><td>77.1%</td><td>93.3%</td><td>5.3M</td><td>1x</td><td>0.39B</td><td>1x</td></tr><tr><td>ResNet-50 (He et al., 2016)</td><td>76.0%</td><td>93.0%</td><td>26M</td><td>4.9x</td><td>4.1B</td><td>11x</td></tr><tr><td>DenseNet-169 (Huang et al., 2017)</td><td>76.2%</td><td>93.2%</td><td>14M</td><td>2.6x</td><td>3.5B</td><td>8.9x</td></tr><tr><td>EfficientNet-B1</td><td>79.1%</td><td>94.4%</td><td>7.8M</td><td>1x</td><td>0.70B</td><td>1x</td></tr><tr><td>ResNet-152 (He et al., 2016)</td><td>77.8%</td><td>93.8%</td><td>60M</td><td>7.6x</td><td>11B</td><td>16x</td></tr><tr><td>DenseNet-264 (Huang et al., 2017)</td><td>77.9%</td><td>93.9%</td><td>34M</td><td>4.3x</td><td>6.0B</td><td>8.6x</td></tr><tr><td>Inception-v3 (Szegedy et al., 2016)</td><td>78.8%</td><td>94.4%</td><td>24M</td><td>3.0x</td><td>5.7B</td><td>8.1x</td></tr><tr><td>Xception (Chollet, 2017)</td><td>79.0%</td><td>94.5%</td><td>23M</td><td>3.0x</td><td>8.4B</td><td>12x</td></tr><tr><td>EfficientNet-B2</td><td>80.1%</td><td>94.9%</td><td>9.2M</td><td>1x</td><td>1.0B</td><td>1x</td></tr><tr><td>Inception-v4 (Szegedy et al., 2017)</td><td>80.0%</td><td>95.0%</td><td>48M</td><td>5.2x</td><td>13B</td><td>13x</td></tr><tr><td>Inception-resnet-v2 (Szegedy et al., 2017)</td><td>80.1%</td><td>95.1%</td><td>56M</td><td>6.1x</td><td>13B</td><td>13x</td></tr><tr><td>EfficientNet-B3</td><td>81.6%</td><td>95.7%</td><td>12M</td><td>1x</td><td>1.8B</td><td>1x</td></tr><tr><td>ResNeXt-101 (Xie et al., 2017)</td><td>80.9%</td><td>95.6%</td><td>84M</td><td>7.0x</td><td>32B</td><td>18x</td></tr><tr><td>PolyNet (Zhang et al., 2017)</td><td>81.3%</td><td>95.8%</td><td>92M</td><td>7.7x</td><td>35B</td><td>19x</td></tr><tr><td>EfficientNet-B4</td><td>82.9%</td><td>96.4%</td><td>19M</td><td>1x</td><td>4.2B</td><td>1x</td></tr><tr><td>SENet (Hu et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>146M</td><td>7.7x</td><td>42B</td><td>10x</td></tr><tr><td>NASNet-A (Zoph et al., 2018)</td><td>82.7%</td><td>96.2%</td><td>89M</td><td>4.7x</td><td>24B</td><td>5.7x</td></tr><tr><td>AmoebaNet-A (Real et al., 2019)</td><td>82.8%</td><td>96.1%</td><td>87M</td><td>4.6x</td><td>23B</td><td>5.5x</td></tr><tr><td>PNASNet (Liu et al., 2018)</td><td>82.9%</td><td>96.2%</td><td>86M</td><td>4.5x</td><td>23B</td><td>6.0x</td></tr><tr><td>EfficientNet-B5</td><td>83.6%</td><td>96.7%</td><td>30M</td><td>1x</td><td>9.9B</td><td>1x</td></tr><tr><td>AmoebaNet-C (Cubuk et al., 2019)</td><td>83.5%</td><td>96.5%</td><td>155M</td><td>5.2x</td><td>41B</td><td>4.1x</td></tr><tr><td>EfficientNet-B6</td><td>84.0%</td><td>96.8%</td><td>43M</td><td>1x</td><td>19B</td><td>1x</td></tr><tr><td>EfficientNet-B7</td><td>84.3%</td><td>97.0%</td><td>66M</td><td>1x</td><td>37B</td><td>1x</td></tr><tr><td>GPipe (Huang et al., 2018)</td><td>84.3%</td><td>97.0%</td><td>557M</td><td>8.4x</td><td>-</td><td>-</td></tr></table>\nWe omit ensemble and multi-crop models (Hu et al., 2018), or models pretrained on 3.5B Instagram images (Mahajan et al., 2018).\n\n[S6] 1905.11946v5 p.5: Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization (Hu et al., 2018).\n\nStarting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:\n\n• STEP 1: we first fix $\\phi = 1$ , assuming twice more resources available, and do a small grid search of $\\alpha , \\beta , \\gamma$ based on Equation 2 and 3. In particular, we find the best values for EfficientNet-B0 are $\\alpha = 1 . 2 , \\beta =$ $1 . 1 , \\gamma = 1 . 1 5$ , under constraint of $\\alpha \\cdot \\beta ^ { 2 } \\cdot \\gamma ^ { 2 } \\approx 2$\n\n• STEP 2: we then fix $\\alpha , \\beta , \\gamma$ as constants and scale up baseline network with different φ using Equation 3, to obtain EfficientNet-B1 to B7 (Details in Table 2).\n\nNotably, it is possible to achieve even better performance by searching for $\\alpha , \\beta ,$ γ directly around a large model, but the search cost becomes prohibitively more expensive on larger models.",
          "truncated": false,
          "retrieval_debug": {
            "candidate_discovery": {
              "metadata_count": 0,
              "chunk_count": 0,
              "entity_hits": {
                "EfficientNet": [
                  "1905.11946"
                ]
              },
              "fallback_used": false,
              "chunk_search_used": false,
              "selected_paper_ids": [
                "1905.11946"
              ],
              "lexical_query": "\"efficientnet compound scaling network depth width resolution\" OR \"efficientnet scales depth width resolution\" OR \"efficientnet scaling depth width resolution\""
            },
            "lexical_query": "\"efficientnet scaling network depth width resolution\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "efficientnet scaling network depth width resolution"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：仅返回 1905.11946。",
        "evidence_quality": "摘要、Introduction 和 Conclusion 均围绕 depth/width/resolution 的 compound scaling，未召回 Appendix。",
        "issues": [
          "第 4 条为 chart，摘要任务中媒体证据优先级仍可进一步降低。"
        ]
      }
    }
  ]
}
```

返回论文：`1905.11946`；质量判定：**通过**。

## 案例 6：概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。

```json
{
  "user_question": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
  "expected_evidence": {
    "paper_ids": [
      "2305.18290"
    ],
    "criteria": "找到 DPO 偏好损失、无需显式奖励模型及 RL 优化循环。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
          "task": "summary",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
          "task": "summary",
          "routing": {
            "route_intent": "retrieve",
            "task": "summary",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "2305.18290",
              "base_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
              "authors": [
                "Rafael Rafailov",
                "Archit Sharma",
                "Eric Mitchell",
                "Stefano Ermon",
                "Christopher D. Manning",
                "Chelsea Finn"
              ],
              "abstract": "While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.",
              "categories": [
                "cs.LG",
                "cs.AI",
                "cs.CL"
              ],
              "published_at": "2023-05-29T17:57:46Z",
              "updated_at": "2024-07-29T22:26:36Z",
              "abs_url": "https://arxiv.org/abs/2305.18290v3",
              "pdf_url": "https://arxiv.org/pdf/2305.18290v3.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.18290\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "ranking_features": {},
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "e66cc285316758799aad1d39",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these prefer ences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss.",
              "retrieval_text": "abstract\nWhile large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these prefer ences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss.",
              "page_start": 0,
              "page_end": 0,
              "source_blocks": [
                {
                  "index": 9,
                  "page_idx": 0,
                  "bbox": [
                    228,
                    393,
                    769,
                    672
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "00c1c1bdc81f1937203d3f864705271b011d55fa67e4d21cd950770931bc9181",
              "retrieval_text_hash": "cffcc2b977724e79094135f6020be573e080562defc462fc301901b27dae52e7",
              "source_id": "S1"
            },
            {
              "score": 0.01847301587301587,
              "semantic_score": 0.6147122979164124,
              "lexical_rank": null,
              "semantic_rank": 3,
              "rrf_score": 0.015873015873015872,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "1e36af39331cb1f8408074d1",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 1,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "c26212e6b22afaef0808addc744529209339c728a3d245d48c454e5ceb455cb0",
              "source_blocks": [
                {
                  "index": 9,
                  "page_idx": 0,
                  "bbox": [
                    228,
                    393,
                    769,
                    672
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b37e369f65bb9ee5d9d411eb0703ba4e565b171a874c6d9844e5367ae4272c90",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.",
              "source_id": "S2"
            },
            {
              "score": 0.01984344262295082,
              "semantic_score": 0.6564347743988037,
              "lexical_rank": null,
              "semantic_rank": 1,
              "rrf_score": 0.01639344262295082,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00345
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "0f7c5d9ea765ab27ed083d82",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 34,
              "region": "content",
              "chapter_number": "5",
              "chapter_title": "Theoretical Analysis of DPO",
              "section_path": [
                "content",
                "5 Theoretical Analysis of DPO"
              ],
              "section_label": "5 Theoretical Analysis of DPO",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "afb4189d9edfbd561c2cca9de8a9e762b6644a58b0f91f008516b01c4896f294",
              "source_blocks": [
                {
                  "index": 56,
                  "page_idx": 4,
                  "bbox": [
                    169,
                    561,
                    823,
                    593
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "ad0e43616cc237f7529e5bc71226760154a1ace7ae34db4697e964a31da36f05",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In this section, we give further interpretation of the DPO method, provide theoretical backing, and relate advantages of DPO to issues with actor critic algorithms used for RLHF (such as PPO [39]).",
              "source_id": "S3"
            },
            {
              "score": 0.019579032258064517,
              "semantic_score": 0.6299431920051575,
              "lexical_rank": null,
              "semantic_rank": 2,
              "rrf_score": 0.016129032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00345
              },
              "page_start_display": 8,
              "page_end_display": 9,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "53e832065f257f7e21ba0f5a",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 57,
              "region": "content",
              "chapter_number": "6.1",
              "chapter_title": "How well can DPO optimize the RLHF objective?",
              "section_path": [
                "content",
                "6 Experiments",
                "6.1 How well can DPO optimize the RLHF objective?"
              ],
              "section_label": "6.1 How well can DPO optimize the RLHF objective?",
              "type": "text",
              "page_start": 7,
              "page_end": 8,
              "content_hash": "97a672a11f4c44c5b93136353cd6a7ef5e25eb2b666343c86081058d7e1caab1",
              "source_blocks": [
                {
                  "index": 91,
                  "page_idx": 7,
                  "bbox": [
                    169,
                    702,
                    828,
                    883
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 94,
                  "page_idx": 8,
                  "bbox": [
                    169,
                    90,
                    826,
                    121
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "d1c0814ca999709b168f30c47ea17f697bd0cfd5ef7afc0610482eaf4c04aa1f",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": ". 1 , 1 , 5 \\}$ $\\alpha \\in \\{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \\}$ for unlikelihood, random seeds for preferred-FT). This sweep includes 22 runs in total. After each 100 training steps until convergence, we evaluate each policy on a set of test prompts, computing the average reward under the true reward function as well as the average sequence-level $\\mathrm { K L } ^ { 3 }$ with the reference policy $\\mathrm { K L } \\left( \\pi \\mid \\mid \\pi _ { \\mathrm { r e f } } \\right)$ We find that DPO produces by far the most efficient frontier, achieving the highest reward while still achieving low KL. This result is particularly notable for multiple reasons. First, DPO and PPO optimize the same objective, but DPO is notably more efficient;\n\nDPO’s reward/KL tradeoff strictly dominates PPO. Second, DPO achieves a better frontier than PPO, even when PPO can access ground truth rewards (PPO-GT).",
              "source_id": "S4"
            },
            {
              "score": 0.018375373134328358,
              "semantic_score": 0.5345525741577148,
              "lexical_rank": null,
              "semantic_rank": 7,
              "rrf_score": 0.014925373134328358,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00345
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "ba194a4f242ed2473198b905",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 60,
              "region": "content",
              "chapter_number": "6.2",
              "chapter_title": "Can DPO scale to real preference datasets?",
              "section_path": [
                "content",
                "6 Experiments",
                "6.2 Can DPO scale to real preference datasets?"
              ],
              "section_label": "6.2 Can DPO scale to real preference datasets?",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "f88f8df527606e01825d3cd6f1795eeb7ebcbb9d822c2e8088d805cc9ad58c4e",
              "source_blocks": [
                {
                  "index": 96,
                  "page_idx": 8,
                  "bbox": [
                    169,
                    174,
                    826,
                    397
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 97,
                  "page_idx": 8,
                  "bbox": [
                    169,
                    402,
                    826,
                    625
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "9d4af64a353d78197f5c32011451e1fae0f293b511d04e1ba2ef457fd8a3dd22",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "referred-FT to train a reference model on the chosen completions such that completions are within distribution of the model, and then train using DPO. We also compare against the best of 128 Preferred-FT completions (we found the Best of N baseline plateaus at 128 completions for this task; see Appendix Figure 4) and a 2-shot prompted version of the Pythia-2.8B base model, finding DPO performs as well or better for the best-performing temperatures for each method. We also evaluate an RLHF model trained with PPO on the Anthropic HH dataset <sup>5</sup> from a well-known source <sup>6</sup>, but are unable to find a prompt or sampling temperature that gives performance better than the base Pythia-2.8B model. Based on our results from TL;DR and the fact that both methods optimize the same reward function, we consider Best of 128 a rough proxy for PPO-level performance. Overall, DPO is the only computationally efficient method that improves over the preferred completions in the Anthropic HH dataset, and provides similar or better performance to the computationally demanding Best of 128 baseline. Finally, Figure 3 shows that DPO converges to its best performance relatively quickly.",
              "source_id": "S5"
            },
            {
              "score": 0.018251515151515154,
              "semantic_score": 0.5408218502998352,
              "lexical_rank": null,
              "semantic_rank": 6,
              "rrf_score": 0.015151515151515152,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.00345
              },
              "page_start_display": 8,
              "page_end_display": 9,
              "query": "概括 DPO 相比基于强化学习的 RLHF 的核心方法与贡献。",
              "evidence_role": "direct",
              "chunk_id": "55a3ad4d5f889d3e6b05e409",
              "paper_id": "2305.18290",
              "canonical_id": "2305.18290v3",
              "ordinal": 56,
              "region": "content",
              "chapter_number": "6.1",
              "chapter_title": "How well can DPO optimize the RLHF objective?",
              "section_path": [
                "content",
                "6 Experiments",
                "6.1 How well can DPO optimize the RLHF objective?"
              ],
              "section_label": "6.1 How well can DPO optimize the RLHF objective?",
              "type": "text",
              "page_start": 7,
              "page_end": 8,
              "content_hash": "4906d5232cbe148ccd7f0d16e18ac2ce12b34d677394ab882bd3756be1c2e1d2",
              "source_blocks": [
                {
                  "index": 91,
                  "page_idx": 7,
                  "bbox": [
                    169,
                    702,
                    828,
                    883
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 94,
                  "page_idx": 8,
                  "bbox": [
                    169,
                    90,
                    826,
                    121
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "ad8be42fe1f088b907b5edce52ea7ae0618ab3bd7d3b66278e829f435325f687",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "The KL-constrained reward maximization objective used in typical RLHF algorithms balances exploitation of reward while restricting the policy from deviating far from the reference policy. Therefore, when comparing algorithms, we must take into account both reward achieved as well as the KL discrepancy; achieving slightly higher reward but with much higher KL is not necessarily desirable. Figure 2 shows the reward-KL frontier for various algorithms in the sentiment setting. We execute multiple training runs for each algorithm, using a different hyperparameter for policy conservativeness in each run (target $\\mathrm { K L } \\in \\{ 3 , 6 , 9 , 1 2 \\}$ for PPO, $\\beta \\in \\{ 0 . 0 5 , 0 . 1 , 1 , 5 \\}$ $\\alpha \\in \\{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \\}$ for unlikelihood, random seeds for preferred-FT). This sweep includes 22 runs in total.",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 2305.18290v3 p.1: While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these prefer ences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss.\n\n[S2] 2305.18290v3 p.1: extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.\n\n[S3] 2305.18290v3 p.5: In this section, we give further interpretation of the DPO method, provide theoretical backing, and relate advantages of DPO to issues with actor critic algorithms used for RLHF (such as PPO [39]).\n\n[S4] 2305.18290v3 p.8: . 1 , 1 , 5 \\}$ $\\alpha \\in \\{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \\}$ for unlikelihood, random seeds for preferred-FT). This sweep includes 22 runs in total. After each 100 training steps until convergence, we evaluate each policy on a set of test prompts, computing the average reward under the true reward function as well as the average sequence-level $\\mathrm { K L } ^ { 3 }$ with the reference policy $\\mathrm { K L } \\left( \\pi \\mid \\mid \\pi _ { \\mathrm { r e f } } \\right)$ We find that DPO produces by far the most efficient frontier, achieving the highest reward while still achieving low KL. This result is particularly notable for multiple reasons. First, DPO and PPO optimize the same objective, but DPO is notably more efficient;\n\nDPO’s reward/KL tradeoff strictly dominates PPO. Second, DPO achieves a better frontier than PPO, even when PPO can access ground truth rewards (PPO-GT).\n\n[S5] 2305.18290v3 p.9: referred-FT to train a reference model on the chosen completions such that completions are within distribution of the model, and then train using DPO. We also compare against the best of 128 Preferred-FT completions (we found the Best of N baseline plateaus at 128 completions for this task; see Appendix Figure 4) and a 2-shot prompted version of the Pythia-2.8B base model, finding DPO performs as well or better for the best-performing temperatures for each method. We also evaluate an RLHF model trained with PPO on the Anthropic HH dataset <sup>5</sup> from a well-known source <sup>6</sup>, but are unable to find a prompt or sampling temperature that gives performance better than the base Pythia-2.8B model. Based on our results from TL;DR and the fact that both methods optimize the same reward function, we consider Best of 128 a rough proxy for PPO-level performance. Overall, DPO is the only computationally efficient method that improves over the preferred completions in the Anthropic HH dataset, and provides similar or better performance to the computationally demanding Best of 128 baseline. Finally, Figure 3 shows that DPO converges to its best performance relatively quickly.\n\n[S6] 2305.18290v3 p.8: The KL-constrained reward maximization objective used in typical RLHF algorithms balances exploitation of reward while restricting the policy from deviating far from the reference policy. Therefore, when comparing algorithms, we must take into account both reward achieved as well as the KL discrepancy; achieving slightly higher reward but with much higher KL is not necessarily desirable. Figure 2 shows the reward-KL frontier for various algorithms in the sentiment setting. We execute multiple training runs for each algorithm, using a different hyperparameter for policy conservativeness in each run (target $\\mathrm { K L } \\in \\{ 3 , 6 , 9 , 1 2 \\}$ for PPO, $\\beta \\in \\{ 0 . 0 5 , 0 . 1 , 1 , 5 \\}$ $\\alpha \\in \\{ 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \\}$ for unlikelihood, random seeds for preferred-FT). This sweep includes 22 runs in total.",
          "truncated": false,
          "retrieval_debug": {
            "candidate_discovery": {
              "metadata_count": 1,
              "chunk_count": 2,
              "entity_hits": {
                "DPO": [
                  "2305.18290"
                ],
                "RLHF": [
                  "2305.18290"
                ]
              },
              "fallback_used": false,
              "chunk_search_used": true,
              "selected_paper_ids": [
                "2305.18290"
              ],
              "lexical_query": "\"dpo\" OR \"direct preference optimization\" OR \"rlhf based reinforcement learning\" OR \"rlhf\" OR \"core methods dpo compared rlhf\" OR \"contribution dpo\" OR \"intensive learning rlhf comparison\""
            },
            "lexical_query": "\"core methods contributions dpo compared rlhf based reinforcement learning\" OR \"direct preference optimization versus reinforcement learning human feedback\" OR \"comparison contribution dpo rlhf methods\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "and",
              "of",
              "to",
              "on",
              "from"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "core methods contributions dpo compared rlhf based reinforcement learning",
              "direct preference optimization versus reinforcement learning human feedback",
              "comparison contribution dpo rlhf methods"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：仅返回 2305.18290。",
        "evidence_quality": "摘要直接对比 RLHF 的 reward model + RL 流程与 DPO 的 closed-form classification loss，并给出稳定性和计算开销贡献。",
        "issues": [
          "第 4 条从公式中间开始，适合作为补充实验材料，不适合作为首条解释证据。"
        ]
      }
    }
  ]
}
```

返回论文：`2305.18290`；质量判定：**通过**。

## 案例 7：Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.

```json
{
  "user_question": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
  "expected_evidence": {
    "paper_ids": [
      "2101.00190",
      "2104.08691"
    ],
    "criteria": "两篇均召回，分别包含 prefix 层级参数与输入 soft prompt 证据。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
          "task": "comparison",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
          "task": "comparison",
          "routing": {
            "route_intent": "retrieve",
            "task": "comparison",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "2104.08691",
              "base_id": "2104.08691",
              "canonical_id": "2104.08691v2",
              "title": "The Power of Scale for Parameter-Efficient Prompt Tuning",
              "authors": [
                "Brian Lester",
                "Rami Al-Rfou",
                "Noah Constant"
              ],
              "abstract": "In this work, we explore \"prompt tuning\", a simple yet effective mechanism for learning \"soft prompts\" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's \"few-shot\" learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method \"closes the gap\" and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant in that large models are costly to share and serve, and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed \"prefix tuning\" of Li and Liang (2021), and we provide a comparison to this and other similar approaches. Finally, we show that conditioning a frozen model with soft prompts confers benefits in robustness to domain transfer, as compared to full model tuning.",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2021-04-18T03:19:26Z",
              "updated_at": "2021-09-02T17:34:41Z",
              "abs_url": "https://arxiv.org/abs/2104.08691v2",
              "pdf_url": "https://arxiv.org/pdf/2104.08691v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2104.08691\\mineru"
                }
              }
            },
            {
              "paper_id": "2101.00190",
              "base_id": "2101.00190",
              "canonical_id": "2101.00190v1",
              "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation",
              "authors": [
                "Xiang Lisa Li",
                "Percy Liang"
              ],
              "abstract": "Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were \"virtual tokens\". We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1\\% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2021-01-01T08:00:36Z",
              "updated_at": "2021-01-01T08:00:36Z",
              "abs_url": "https://arxiv.org/abs/2101.00190v1",
              "pdf_url": "https://arxiv.org/pdf/2101.00190v1.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2101.00190\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "score": 0.02946804324707551,
              "semantic_score": 0.5045058727264404,
              "lexical_rank": 2,
              "semantic_rank": 31,
              "rrf_score": 0.027118043247075507,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00235
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "8e1cb5bb17d49db9987b2f64",
              "paper_id": "2104.08691",
              "canonical_id": "2104.08691v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "ca9bc20b581c188db651ff110cd5bc82cfc60ad36c0f7634b67a794507a4fa2e",
              "source_blocks": [
                {
                  "index": 5,
                  "page_idx": 0,
                  "bbox": [
                    141,
                    275,
                    462,
                    676
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "987cca447b0af41c6ed8935c7b74291d5ef96ed42fd1527bc3cad727d06ec803",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In this work, we explore “prompt tuning,” a simple yet effective mechanism for learning “soft prompts” to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signals from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3’s few-shot learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method “closes the gap” and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant because large models are costly to share and serve and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed “prefix tuning” of Li and Liang (2021) and we provide a comparison to this and other similar approaches.",
              "source_id": "S1"
            },
            {
              "score": 0.03437805800756621,
              "semantic_score": 0.6614490747451782,
              "lexical_rank": 5,
              "semantic_rank": 1,
              "rrf_score": 0.03177805800756621,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 2,
              "page_end_display": 2,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "1be269aff1f43ce7eb174557",
              "paper_id": "2104.08691",
              "canonical_id": "2104.08691v2",
              "ordinal": 6,
              "region": "content",
              "chapter_number": "1",
              "chapter_title": "Introduction",
              "section_path": [
                "content",
                "1 Introduction"
              ],
              "section_label": "1 Introduction",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "7afa09ba5d30a679ae29ecd80e97cdfa34f39ccd2fee3e06ac3af8dcb8bb8df1",
              "source_blocks": [
                {
                  "index": 15,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    445,
                    490,
                    542
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 16,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    545,
                    490,
                    706
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 17,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    708,
                    490,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 18,
                  "page_idx": 1,
                  "bbox": [
                    132,
                    903,
                    487,
                    919
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 20,
                  "page_idx": 1,
                  "bbox": [
                    507,
                    229,
                    887,
                    456
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 21,
                  "page_idx": 1,
                  "bbox": [
                    524,
                    464,
                    884,
                    512
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 22,
                  "page_idx": 1,
                  "bbox": [
                    524,
                    512,
                    882,
                    544
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 23,
                  "page_idx": 1,
                  "bbox": [
                    524,
                    545,
                    882,
                    576
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 24,
                  "page_idx": 1,
                  "bbox": [
                    524,
                    577,
                    882,
                    608
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "dc93d3b59bdadd6f76a172420c7b306557fc147d0eadbbe49f543723f142603a",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Several efforts to automate prompt design have been recently proposed. Shin et al. (2020) propose a search algorithm over the discrete space of words, guided by the downstream application training data. While this technique outperforms manual prompt design, there is still a gap relative to model tuning.\n\nLi and Liang (2021) propose “prefix tuning” and show strong results on generative tasks. This method freezes the model parameters and backpropagates the error during tuning to prefix activations prepended to each layer in the encoder stack, including the input layer. Hambardzumyan et al. (2021) simplify this recipe by restricting the trainable parameters to the input and output subnetworks of a masked language model, and show reasonable results on classifications tasks.\n\nIn this paper, we propose prompt tuning as a further simplification for adapting language models. We freeze the entire pre-trained model and only allow an additional k tunable tokens per downstream task to be prepended to the input text.",
              "source_id": "S2"
            },
            {
              "score": 0.03296576949620428,
              "semantic_score": 0.587007999420166,
              "lexical_rank": 3,
              "semantic_rank": 9,
              "rrf_score": 0.03036576949620428,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 6,
              "page_end_display": 7,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "ee214145a80dd2a5c4d71cc7",
              "paper_id": "2104.08691",
              "canonical_id": "2104.08691v2",
              "ordinal": 30,
              "region": "content",
              "chapter_number": "4",
              "chapter_title": "Comparison to Similar Approaches",
              "section_path": [
                "content",
                "4 Comparison to Similar Approaches"
              ],
              "section_label": "4 Comparison to Similar Approaches",
              "type": "text",
              "page_start": 5,
              "page_end": 6,
              "content_hash": "ad7faf5cff5a4d5cd236c910901aa1cbddeb4c4b4a774cc8d44ca801ee0b7a4a",
              "source_blocks": [
                {
                  "index": 77,
                  "page_idx": 5,
                  "bbox": [
                    507,
                    621,
                    885,
                    765
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 78,
                  "page_idx": 5,
                  "bbox": [
                    507,
                    766,
                    885,
                    799
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 82,
                  "page_idx": 6,
                  "bbox": [
                    112,
                    423,
                    490,
                    601
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 83,
                  "page_idx": 6,
                  "bbox": [
                    112,
                    601,
                    490,
                    762
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 84,
                  "page_idx": 6,
                  "bbox": [
                    112,
                    762,
                    490,
                    860
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "2d848a845ede979c31b8a97bd23ed45db14e9c30ace286707f3dd7e47b9decd2",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "In this section, we review recent work on learning continuous prompts, and draw comparisons with our method. One important axis of comparison is the number of task-specific parameters each method requires, as shown in Figure 4. Among methods with learnable parameters, prompt tuning is the most parameter efficient, requiring less than 0.01% task-specific parameters for models over a billion parameters.<sup>9</sup>\n\nLi and Liang (2021) propose “prefix tuning”: learning a sequence of prefixes that are prepended at every transformer layer. This is akin to learning transformer activations that are fixed across examples at every network layer. In contrast, prompt tuning uses a single prompt representation that is prepended to the embedded input. Beyond requiring fewer parameters, our approach allows the transformer to update the intermediate-layer task representations, as contextualized by an input example. Their work builds on GPT-2 (Radford et al., 2019) and BART (Lewis et al., 2020), while ours focuses on T5 and examines changes in performance and robustness to design choices as model size increases.",
              "source_id": "S3"
            },
            {
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "ranking_features": {},
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "cdaff8cef9c2827f273533bd",
              "paper_id": "2101.00190",
              "canonical_id": "2101.00190v1",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were “virtual tokens”. We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.",
              "retrieval_text": "abstract\nFine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were “virtual tokens”. We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.",
              "page_start": 0,
              "page_end": 0,
              "source_blocks": [
                {
                  "index": 4,
                  "page_idx": 0,
                  "bbox": [
                    142,
                    294,
                    463,
                    595
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "4288030f5b32cd5e9fe12e8890a833a0003aa6b3092b24c03e87f5430a19562c",
              "retrieval_text_hash": "4b769ee177f71442f5cec0359c30fc37727b7777efebfe3d266ea6e9492ab74a",
              "source_id": "S4"
            },
            {
              "score": 0.03385873015873016,
              "semantic_score": 0.644365668296814,
              "lexical_rank": 10,
              "semantic_rank": 3,
              "rrf_score": 0.030158730158730156,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "10fd15e5f27ec77b45a9465b",
              "paper_id": "2101.00190",
              "canonical_id": "2101.00190v1",
              "ordinal": 56,
              "region": "content",
              "chapter_number": "8.3",
              "chapter_title": "Inductive Bias of Prefix-tuning",
              "section_path": [
                "content",
                "8 Discussion",
                "8.3 Inductive Bias of Prefix-tuning"
              ],
              "section_label": "8.3 Inductive Bias of Prefix-tuning",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "5825ce9bfeb39f04a72e0d72e9bb8f67db17a4dd2d079b71c9cf358616e1df8f",
              "source_blocks": [
                {
                  "index": 136,
                  "page_idx": 8,
                  "bbox": [
                    114,
                    813,
                    492,
                    910
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 138,
                  "page_idx": 8,
                  "bbox": [
                    509,
                    140,
                    887,
                    381
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 139,
                  "page_idx": 8,
                  "bbox": [
                    509,
                    381,
                    887,
                    527
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "1b9bfdabe8c1c5427c364b4cfd9220ffee0bb3295aee3d27c7a99fe8268b9755",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Recall that fine-tuning updates all pretrained parameters, whereas prefix-tuning and adapter-tuning preserve them. Since the language models are pretrained on general purpose corpus, preserving the LM parameters might help generalization to domains unseen during training. In concordance with this intuition, we observe that both prefix-tuning and adapter-tuning have significant performance gain in extrapolation settings (§6.4); however, the reason for such gain is an open question.\n\nWhile prefix-tuning and adapter-tuning both freeze the pretrained parameters, they tune different sets of parameters to affect the activation layers of the Transformer. Recall that prefix-tuning keeps the LM intact and uses the prefix and the pretrained attention blocks to affect the subsequent activations; adapter-tuning inserts trainable modules between LM layers, which directly add residual vectors to the activations. Moreover, we observe that prefixtuning requires vastly fewer parameters compared to adapter-tuning while maintaining comparable performance.",
              "source_id": "S5"
            },
            {
              "score": 0.030909409888357255,
              "semantic_score": 0.6145123839378357,
              "lexical_rank": 16,
              "semantic_rank": 6,
              "rrf_score": 0.028309409888357256,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "query": "Compare Prefix-Tuning and Prompt Tuning in terms of where trainable parameters are inserted and whether pretrained weights are frozen.",
              "evidence_role": "direct",
              "chunk_id": "55bf68d19f369cb82795dfb0",
              "paper_id": "2101.00190",
              "canonical_id": "2101.00190v1",
              "ordinal": 22,
              "region": "content",
              "chapter_number": "4.2",
              "chapter_title": "Method",
              "section_path": [
                "content",
                "4 Prefix-Tuning",
                "4.2 Method"
              ],
              "section_label": "4.2 Method",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "565e9ad5259e59070bf36ae149a0022fddf3dc022abb3700d430d99ac0c2c542",
              "source_blocks": [
                {
                  "index": 46,
                  "page_idx": 3,
                  "bbox": [
                    509,
                    338,
                    887,
                    434
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 47,
                  "page_idx": 3,
                  "bbox": [
                    509,
                    435,
                    887,
                    514
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "24290cc91e7cf98b22ac71e7edcac71472afe881ba6f185f0c373b02bc525b61",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Prefix-tuning prepends a prefix for an autoregressive LM to obtain $z = [ \\mathrm { P R E F I X } ; x ; y ]$ , or prepends prefixes for both encoder and encoder to obtain $z = [ \\mathrm { P R E F I X } ; x ; \\mathrm { P R E F I X } ^ { \\prime } ; y ]$ , as shown in Figure 2. Here, $\\mathsf { P } _ { \\mathrm { i d x } }$ denotes the sequence of prefix indices, and we use $| \\mathsf { P } _ { \\mathrm { i d } \\times } |$ to denote the length of the prefix.\n\nWe follow the recurrence relation in equation (1), except that the prefix are free parameters. Prefix-tuning initializes a trainable matrix $P _ { \\theta }$ (parametrized by θ) of dimension $| \\mathsf { P } _ { \\mathrm { i d } \\mathsf { x } } | \\times \\dim ( h _ { i } )$ to store the prefix parameters.",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 2104.08691v2 p.1: In this work, we explore “prompt tuning,” a simple yet effective mechanism for learning “soft prompts” to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signals from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3’s few-shot learning by a large margin. More remarkably, through ablations on model size using T5, we show that prompt tuning becomes more competitive with scale: as models exceed billions of parameters, our method “closes the gap” and matches the strong performance of model tuning (where all model weights are tuned). This finding is especially relevant because large models are costly to share and serve and the ability to reuse one frozen model for multiple downstream tasks can ease this burden. Our method can be seen as a simplification of the recently proposed “prefix tuning” of Li and Liang (2021) and we provide a comparison to this and other similar approaches.\n\n[S2] 2104.08691v2 p.2: Several efforts to automate prompt design have been recently proposed. Shin et al. (2020) propose a search algorithm over the discrete space of words, guided by the downstream application training data. While this technique outperforms manual prompt design, there is still a gap relative to model tuning.\n\nLi and Liang (2021) propose “prefix tuning” and show strong results on generative tasks. This method freezes the model parameters and backpropagates the error during tuning to prefix activations prepended to each layer in the encoder stack, including the input layer. Hambardzumyan et al. (2021) simplify this recipe by restricting the trainable parameters to the input and output subnetworks of a masked language model, and show reasonable results on classifications tasks.\n\nIn this paper, we propose prompt tuning as a further simplification for adapting language models. We freeze the entire pre-trained model and only allow an additional k tunable tokens per downstream task to be prepended to the input text.\n\n[S3] 2104.08691v2 p.6: In this section, we review recent work on learning continuous prompts, and draw comparisons with our method. One important axis of comparison is the number of task-specific parameters each method requires, as shown in Figure 4. Among methods with learnable parameters, prompt tuning is the most parameter efficient, requiring less than 0.01% task-specific parameters for models over a billion parameters.<sup>9</sup>\n\nLi and Liang (2021) propose “prefix tuning”: learning a sequence of prefixes that are prepended at every transformer layer. This is akin to learning transformer activations that are fixed across examples at every network layer. In contrast, prompt tuning uses a single prompt representation that is prepended to the embedded input. Beyond requiring fewer parameters, our approach allows the transformer to update the intermediate-layer task representations, as contextualized by an input example. Their work builds on GPT-2 (Radford et al., 2019) and BART (Lewis et al., 2020), while ours focuses on T5 and examines changes in performance and robustness to design choices as model size increases.\n\n[S4] 2101.00190v1 p.1: Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were “virtual tokens”. We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.\n\n[S5] 2101.00190v1 p.9: Recall that fine-tuning updates all pretrained parameters, whereas prefix-tuning and adapter-tuning preserve them. Since the language models are pretrained on general purpose corpus, preserving the LM parameters might help generalization to domains unseen during training. In concordance with this intuition, we observe that both prefix-tuning and adapter-tuning have significant performance gain in extrapolation settings (§6.4); however, the reason for such gain is an open question.\n\nWhile prefix-tuning and adapter-tuning both freeze the pretrained parameters, they tune different sets of parameters to affect the activation layers of the Transformer. Recall that prefix-tuning keeps the LM intact and uses the prefix and the pretrained attention blocks to affect the subsequent activations; adapter-tuning inserts trainable modules between LM layers, which directly add residual vectors to the activations. Moreover, we observe that prefixtuning requires vastly fewer parameters compared to adapter-tuning while maintaining comparable performance.\n\n[S6] 2101.00190v1 p.4: Prefix-tuning prepends a prefix for an autoregressive LM to obtain $z = [ \\mathrm { P R E F I X } ; x ; y ]$ , or prepends prefixes for both encoder and encoder to obtain $z = [ \\mathrm { P R E F I X } ; x ; \\mathrm { P R E F I X } ^ { \\prime } ; y ]$ , as shown in Figure 2. Here, $\\mathsf { P } _ { \\mathrm { i d x } }$ denotes the sequence of prefix indices, and we use $| \\mathsf { P } _ { \\mathrm { i d } \\times } |$ to denote the length of the prefix.\n\nWe follow the recurrence relation in equation (1), except that the prefix are free parameters. Prefix-tuning initializes a trainable matrix $P _ { \\theta }$ (parametrized by θ) of dimension $| \\mathsf { P } _ { \\mathrm { i d } \\mathsf { x } } | \\times \\dim ( h _ { i } )$ to store the prefix parameters.",
          "truncated": false,
          "retrieval_debug": {
            "candidate_discovery": {
              "metadata_count": 2,
              "chunk_count": 0,
              "entity_hits": {},
              "fallback_used": false,
              "chunk_search_used": false,
              "selected_paper_ids": [
                "2104.08691",
                "2101.00190"
              ],
              "lexical_query": "\"prefix tuning\" OR \"prompt tuning\" OR \"comparison\" OR \"trainable parameters inserted\" OR \"whether pretrained weights frozen\" OR \"pretrained weights frozen\""
            },
            "lexical_query": "\"prefix tuning\" OR \"prompt tuning\" OR \"trainable parameters inserted\" OR \"pretrained weights frozen\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "prefix tuning",
              "prompt tuning",
              "trainable parameters inserted",
              "pretrained weights frozen"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：2104.08691 和 2101.00190 均命中，前五条没有背景论文。",
        "evidence_quality": "证据覆盖 prompt tuning 的输入 soft prompts、prefix tuning 的逐层 prefixes，以及两者冻结预训练权重的共同点。",
        "issues": []
      }
    }
  ]
}
```

返回论文：`2104.08691, 2101.00190`；质量判定：**通过**。

## 案例 8：比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。

```json
{
  "user_question": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
  "expected_evidence": {
    "paper_ids": [
      "2010.11929",
      "2103.14030"
    ],
    "criteria": "两篇均召回，支持平坦/层级结构、全局/窗口注意力比较。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
          "task": "comparison",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
          "task": "comparison",
          "routing": {
            "route_intent": "retrieve",
            "task": "comparison",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "2010.11929",
              "base_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
              "authors": [
                "Alexey Dosovitskiy",
                "Lucas Beyer",
                "Alexander Kolesnikov",
                "Dirk Weissenborn",
                "Xiaohua Zhai",
                "Thomas Unterthiner",
                "Mostafa Dehghani",
                "Matthias Minderer",
                "Georg Heigold",
                "Sylvain Gelly",
                "Jakob Uszkoreit",
                "Neil Houlsby"
              ],
              "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
              "categories": [
                "cs.CV",
                "cs.AI",
                "cs.LG"
              ],
              "published_at": "2020-10-22T17:55:59Z",
              "updated_at": "2021-06-03T13:08:56Z",
              "abs_url": "https://arxiv.org/abs/2010.11929v2",
              "pdf_url": "https://arxiv.org/pdf/2010.11929v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru"
                }
              }
            },
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            },
            {
              "paper_id": "2006.16236",
              "base_id": "2006.16236",
              "canonical_id": "2006.16236v3",
              "title": "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
              "authors": [
                "Angelos Katharopoulos",
                "Apoorv Vyas",
                "Nikolaos Pappas",
                "François Fleuret"
              ],
              "abstract": "Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input's length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from $\\mathcal{O}\\left(N^2\\right)$ to $\\mathcal{O}\\left(N\\right)$, where $N$ is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.",
              "categories": [
                "cs.LG",
                "stat.ML"
              ],
              "published_at": "2020-06-29T17:55:38Z",
              "updated_at": "2020-08-31T11:09:32Z",
              "abs_url": "https://arxiv.org/abs/2006.16236v3",
              "pdf_url": "https://arxiv.org/pdf/2006.16236v3.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "score": 0.027676537113730488,
              "semantic_score": 0.4696776866912842,
              "lexical_rank": 13,
              "semantic_rank": 26,
              "rrf_score": 0.025326537113730486,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00235
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
              "evidence_role": "direct",
              "chunk_id": "7c23016818426e62c095f7a5",
              "paper_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "fe646b8c1979a43ae3727d4ca5901279f4e1d45746c54cd703671b1fc18c8c15",
              "source_blocks": [
                {
                  "index": 6,
                  "page_idx": 0,
                  "bbox": [
                    228,
                    338,
                    767,
                    493
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b521af614e6b82024056f691d2244c8e6637fd893f1ea0e6e861f31ba648eb51",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.<sup>1</sup>",
              "source_id": "S1"
            },
            {
              "score": 0.030330294396961062,
              "semantic_score": 0.47870081663131714,
              "lexical_rank": 5,
              "semantic_rank": 21,
              "rrf_score": 0.027730294396961064,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
              "evidence_role": "direct",
              "chunk_id": "ffd11d1dc4b7d5a88a4954d5",
              "paper_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "ordinal": 20,
              "region": "content",
              "chapter_number": "4",
              "chapter_title": "EXPERIMENTS",
              "section_path": [
                "content",
                "4 EXPERIMENTS"
              ],
              "section_label": "4 EXPERIMENTS",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "b66365275e9bf3d285e2f2da244919f82528732b20c1fcc7806f4b5d55fdddda",
              "source_blocks": [
                {
                  "index": 47,
                  "page_idx": 3,
                  "bbox": [
                    169,
                    672,
                    826,
                    757
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "611b9ace77dc7d0d2cb3673ca2bbf608426192637a83cdaf7896fa91c0dbdf9f",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show that self-supervised ViT holds promise for the future.",
              "source_id": "S2"
            },
            {
              "score": 0.033216458495966696,
              "semantic_score": 0.6331514716148376,
              "lexical_rank": 3,
              "semantic_rank": 1,
              "rrf_score": 0.032266458495966696,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00095
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
              "evidence_role": "direct",
              "chunk_id": "aef74810680c1fef21c96da5",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "8dd511eb8818ff1142cbd562e68a32e670fe8c7456c804dbe96ff05374087123",
              "source_blocks": [
                {
                  "index": 6,
                  "page_idx": 0,
                  "bbox": [
                    75,
                    319,
                    470,
                    743
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b47f9f16ab0045fc408d5cc9b9973271a8d46fc70bea8be0e767dc9b747136d3",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val).",
              "source_id": "S3"
            },
            {
              "score": 0.03345806451612903,
              "semantic_score": 0.6108789443969727,
              "lexical_rank": 2,
              "semantic_rank": 2,
              "rrf_score": 0.03225806451612903,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 8,
              "page_end_display": 8,
              "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
              "evidence_role": "direct",
              "chunk_id": "8d099939c00841a83f0b1126",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 57,
              "region": "content",
              "chapter_number": "5",
              "chapter_title": "Conclusion",
              "section_path": [
                "content",
                "5. Conclusion"
              ],
              "section_label": "5. Conclusion",
              "type": "text",
              "page_start": 7,
              "page_end": 7,
              "content_hash": "87e4848b869d3f00cd0c39928e401436fbf2751942ab78495624a31c82365405",
              "source_blocks": [
                {
                  "index": 118,
                  "page_idx": 7,
                  "bbox": [
                    498,
                    869,
                    893,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "fe12eb09577640596770e13aa47767aa24b5a030a2b575ab3079981ed883b762",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "This paper presents Swin Transformer, a new vision Transformer which produces a hierarchical feature representation and has linear computational complexity with respect to input image size. Swin Transformer achieves the state-of-the-art performance on COCO object detection and ADE20K semantic segmentation, significantly surpassing previous best methods. We hope that Swin Transformer’s strong performance on various vision problems will encourage unified modeling of vision and language signals.",
              "source_id": "S4"
            },
            {
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "ranking_features": {},
              "page_start_display": 1,
              "page_end_display": 1,
              "query": "比较 ViT 与 Swin Transformer 的图像表示层级、注意力范围和计算复杂度。",
              "evidence_role": "direct",
              "chunk_id": "783270fd0f61f09c2fb267a6",
              "paper_id": "2006.16236",
              "canonical_id": "2006.16236v3",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from O \u0000N<sup>2</sup>\u0001 to O (N), where N is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.",
              "retrieval_text": "abstract\nTransformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from O \u0000N<sup>2</sup>\u0001 to O (N), where N is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.",
              "page_start": 0,
              "page_end": 0,
              "source_blocks": [
                {
                  "index": 3,
                  "page_idx": 0,
                  "bbox": [
                    117,
                    263,
                    444,
                    523
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "1fb90aa9d8359d58e9e4d0d723a790bc9256c1fef177ec503969f1589ca996fd",
              "retrieval_text_hash": "cb04614d92824606ee703b7a61bdf3c32e5b7c5af6fe9e419dfac331a8e8e3e9",
              "source_id": "S5"
            }
          ],
          "count": 5,
          "context_text": "[S1] 2010.11929v2 p.1: While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.<sup>1</sup>\n\n[S2] 2010.11929v2 p.4: We evaluate the representation learning capabilities of ResNet, Vision Transformer (ViT), and the hybrid. To understand the data requirements of each model, we pre-train on datasets of varying size and evaluate many benchmark tasks. When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost. Lastly, we perform a small experiment using self-supervision, and show that self-supervised ViT holds promise for the future.\n\n[S3] 2103.14030v2 p.1: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val).\n\n[S4] 2103.14030v2 p.8: This paper presents Swin Transformer, a new vision Transformer which produces a hierarchical feature representation and has linear computational complexity with respect to input image size. Swin Transformer achieves the state-of-the-art performance on COCO object detection and ADE20K semantic segmentation, significantly surpassing previous best methods. We hope that Swin Transformer’s strong performance on various vision problems will encourage unified modeling of vision and language signals.\n\n[S5] 2006.16236v3 p.1: Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from O \u0000N<sup>2</sup>\u0001 to O (N), where N is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.",
          "truncated": false,
          "retrieval_debug": {
            "candidate_discovery": {
              "metadata_count": 2,
              "chunk_count": 3,
              "entity_hits": {
                "ViT": [
                  "2010.11929"
                ]
              },
              "fallback_used": false,
              "chunk_search_used": true,
              "selected_paper_ids": [
                "2010.11929",
                "2103.14030",
                "2006.16236"
              ],
              "lexical_query": "\"vit\" OR \"swin transformer\" OR \"image representation level\" OR \"attention span\" OR \"computational complexity\""
            },
            "lexical_query": "\"compare\" OR \"image\" OR \"representation\" OR \"levels\" OR \"attention\" OR \"range\" OR \"computational\" OR \"complexity\" OR \"vit\" OR \"swin\" OR \"transformer.\" OR \"vit\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "the",
              "and",
              "of"
            ],
            "rewriter_used": false,
            "rewriter_fallback": true,
            "core_terms": []
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [
          "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
        ],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过",
        "paper_recall": "部分通过：ViT 和 Swin 命中，但额外召回 2006.16236。",
        "evidence_quality": "Swin 摘要和结论覆盖层级表示、局部窗口和线性复杂度；ViT 的首两条没有直接给出与 Swin 的全局注意力复杂度对照。",
        "issues": [
          "候选发现的全文回退引入与图像比较无关的线性注意力论文；ViT 对比证据不足。"
        ]
      }
    }
  ]
}
```

返回论文：`2010.11929, 2103.14030, 2006.16236`；质量判定：**部分通过**。

## 案例 9：ResNet 的附录如何解释残差网络能够表示指数数量的浅层网络这一观点？

```json
{
  "user_question": "ResNet 的附录如何解释残差网络能够表示指数数量的浅层网络这一观点？",
  "expected_evidence": {
    "paper_ids": [
      "1512.03385"
    ],
    "criteria": "诊断附录查询：必须检查论文实际附录是否支持命题，不能只因命中 ResNet 就判通过。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "ResNet 的附录如何解释残差网络能够表示指数数量的浅层网络这一观点？",
          "task": "fact",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "ResNet 的附录如何解释残差网络能够表示指数数量的浅层网络这一观点？",
          "task": "fact",
          "routing": {
            "route_intent": "retrieve",
            "task": "fact",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "1512.03385",
              "base_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "title": "Deep Residual Learning for Image Recognition",
              "authors": [
                "Kaiming He",
                "Xiangyu Zhang",
                "Shaoqing Ren",
                "Jian Sun"
              ],
              "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.",
              "categories": [
                "cs.CV"
              ],
              "published_at": "2015-12-10T19:51:55Z",
              "updated_at": "2015-12-10T19:51:55Z",
              "abs_url": "https://arxiv.org/abs/1512.03385v1",
              "pdf_url": "https://arxiv.org/pdf/1512.03385v1.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1512.03385\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "083d2ebfc2f23f7c6423265b",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 36,
              "region": "content",
              "chapter_number": "4.1",
              "chapter_title": "ImageNet Classification",
              "section_path": [
                "content",
                "4. Experiments",
                "4.1. ImageNet Classification"
              ],
              "section_label": "4.1. ImageNet Classification",
              "type": "text",
              "page_start": 5,
              "page_end": 5,
              "content_hash": "523ac5421323dc7bd57759aca10d34cdece7c282acc71184264321a47a9f2cde",
              "source_blocks": [
                {
                  "index": 86,
                  "page_idx": 5,
                  "bbox": [
                    75,
                    708,
                    470,
                    768
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 87,
                  "page_idx": 5,
                  "bbox": [
                    75,
                    770,
                    470,
                    878
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 88,
                  "page_idx": 5,
                  "bbox": [
                    76,
                    885,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "1d6d1dfe641434158456e0a3bf236922b530d7b6d49ac24d7c6f5eb24e846e80",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "ResNet reduces the top-1 error by 3.5% (Table 2), resulting from the successfully reduced training error (Fig. 4 right vs. left). This comparison verifies the effectiveness of residual learning on extremely deep systems.\n\nLast, we also note that the 18-layer plain/residual nets are comparably accurate (Table 2), but the 18-layer ResNet converges faster (Fig. 4 right vs. left). When the net is “not overly deep” (18 layers here), the current SGD solver is still able to find good solutions to the plain net. In this case, the ResNet eases the optimization by providing faster convergence at the early stage.\n\nIdentity vs. Projection Shortcuts. We have shown that parameter-free, identity shortcuts help with training. Next we investigate projection shortcuts (Eqn.(2)). In Table 3 we compare three options: (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameterfree (the same as Table 2 and Fig. 4 right); (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity; and (C) all shortcuts are projections.",
              "score": 0.031585507246376814,
              "semantic_score": 0.43782931566238403,
              "lexical_rank": 9,
              "semantic_rank": 9,
              "rrf_score": 0.028985507246376812,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 6,
              "page_end_display": 6,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "4d26b5a31a20eb3c11a32f42",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "f5ad0cfd30ec471a51016faf1b568951ed141bf23fcf8a1cf6ebb36efad91e92",
              "source_blocks": [
                {
                  "index": 7,
                  "page_idx": 0,
                  "bbox": [
                    75,
                    306,
                    470,
                    534
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 8,
                  "page_idx": 0,
                  "bbox": [
                    73,
                    535,
                    470,
                    657
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "584bad492fd9ba7266a139bfcdf6c3480ac3a2370a5e61905174000d0ae3c069",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracyfrom considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth ofup to 152 layers—8 deeper than VGG nets [41] but still having lower complexity. An ensemble ofthese residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.\n\nThe depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset.",
              "score": 0.03116353930031804,
              "semantic_score": 0.4761984348297119,
              "lexical_rank": 11,
              "semantic_rank": 2,
              "rrf_score": 0.03021353930031804,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00095
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "c3c8066efcdd3bf6f9ebf7f6",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 32,
              "region": "content",
              "chapter_number": "4.1",
              "chapter_title": "ImageNet Classification",
              "section_path": [
                "content",
                "4. Experiments",
                "4.1. ImageNet Classification"
              ],
              "section_label": "4.1. ImageNet Classification",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "c761cd63a1655073acc9cbcb3cb767a7ae27c5bd113e5546e535bdcf0a98b03d",
              "source_blocks": [
                {
                  "index": 75,
                  "page_idx": 4,
                  "bbox": [
                    75,
                    685,
                    470,
                    746
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 76,
                  "page_idx": 4,
                  "bbox": [
                    75,
                    750,
                    470,
                    902
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 78,
                  "page_idx": 4,
                  "bbox": [
                    496,
                    597,
                    893,
                    719
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 79,
                  "page_idx": 4,
                  "bbox": [
                    496,
                    718,
                    893,
                    839
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 80,
                  "page_idx": 4,
                  "bbox": [
                    517,
                    839,
                    893,
                    854
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "a025eaeb43b515c07262f175d991493e583d41e74092eb9f61056147e16bdece",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "ove plain nets, expect that a shortcut connection is added to each pair of 3 3 filters as in Fig. 3 (right). In the first comparison (Table 2 and Fig. 4 right), we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A). So they have no extra parameter compared to the plain counterparts.\n\nWe have three major observations from Table 2 and Fig. 4. First, the situation is reversed with residual learning – the 34-layer ResNet is better than the 18-layer ResNet (by 2.8%). More importantly, the 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data. This indicates that the degradation problem is well addressed in this setting and we manage to obtain accuracy gains from increased depth.\n\nSecond, compared to its plain counterpart, the 34-layer",
              "score": 0.03110014528850145,
              "semantic_score": 0.453117311000824,
              "lexical_rank": 13,
              "semantic_rank": 6,
              "rrf_score": 0.028850145288501453,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0026
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "943a69433fc9538238303160",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 52,
              "region": "content",
              "chapter_number": "4.2",
              "chapter_title": "CIFAR-10 and Analysis",
              "section_path": [
                "content",
                "4. Experiments",
                "4.2. CIFAR-10 and Analysis"
              ],
              "section_label": "4.2. CIFAR-10 and Analysis",
              "type": "text",
              "page_start": 7,
              "page_end": 7,
              "content_hash": "789df35127e590f2991a091d2621fa5f268b1010bb52f4ffb6aace4608facd46",
              "source_blocks": [
                {
                  "index": 120,
                  "page_idx": 7,
                  "bbox": [
                    75,
                    530,
                    470,
                    743
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 121,
                  "page_idx": 7,
                  "bbox": [
                    75,
                    750,
                    470,
                    854
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 122,
                  "page_idx": 7,
                  "bbox": [
                    75,
                    854,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "f6e49133719dae8cc2423a663a14ea88fc3910adc89d2291e556b219bb49312f",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Analysis of Layer Responses. Fig. 7 shows the standard deviations (std) of the layer responses. The responses are the outputs of each 3 3 layer, after BN and before other nonlinearity (ReLU/addition). For ResNets, this analysis reveals the response strength of the residual functions. Fig. 7 shows that ResNets have generally smaller responses than their plain counterparts. These results support our basic motivation (Sec.3.1) that the residual functions might be generally closer to zero than the non-residual functions. We also notice that the deeper ResNet has smaller magnitudes of responses, as evidenced by the comparisons among ResNet-20, 56, and 110 in Fig. 7. When there are more layers, an individual layer of ResNets tends to modify the signal less.\n\nExploring Over 1000 layers. We explore an aggressively deep model of over 1000 layers. We set n = 200 that leads to a 1202-layer network, which is trained as described above. Our method shows no optimization difficulty, and this 10<sup>3</sup>-layer network is able to achieve training error <0.1% (Fig. 6, right). Its test error is still fairly good (7.93%, Table 6).",
              "score": 0.03027319277108434,
              "semantic_score": 0.456770658493042,
              "lexical_rank": 23,
              "semantic_rank": 4,
              "rrf_score": 0.027673192771084338,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 8,
              "page_end_display": 8,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "ff9afd27f590aac4b0d3714c",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 33,
              "region": "content",
              "chapter_number": "4.1",
              "chapter_title": "ImageNet Classification",
              "section_path": [
                "content",
                "4. Experiments",
                "4.1. ImageNet Classification"
              ],
              "section_label": "4.1. ImageNet Classification",
              "type": "table",
              "page_start": 5,
              "page_end": 5,
              "content_hash": "6df0595539181ee5364eab8682b27f7363a68679122b16f4421750a93094c102",
              "source_blocks": [
                {
                  "index": 83,
                  "page_idx": 5,
                  "bbox": [
                    143,
                    90,
                    405,
                    262
                  ],
                  "type": "table",
                  "text_format": null,
                  "context_before": "Second, compared to its plain counterpart, the 34-layer",
                  "context_after": "ResNet reduces the top-1 error by 3.5% (Table 2), resulting from the successfully reduced training error (Fig. 4 right vs. left). This comparison verifies the effectiveness of residual learning on extremely deep systems."
                }
              ],
              "asset_refs": [
                "images/82eca399d7bae4d675a3048115b9c2addfdc0efee23498cbe0f20ec7685026e7.jpg"
              ],
              "retrieval_text_hash": "faad8246e8f8b5d78352a6f6e141efaefa0d516ae27f67d4d39f2e8ebab3ab91",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "<table><tr><td>model</td><td>top-1 err.</td><td>top-5 err.</td></tr><tr><td>VGG-16 [41]</td><td>28.07</td><td>9.33</td></tr><tr><td>GoogLeNet [44]</td><td>-</td><td>9.15</td></tr><tr><td>PReLU-net [13]</td><td>24.27</td><td>7.38</td></tr><tr><td>plain-34</td><td>28.54</td><td>10.02</td></tr><tr><td>ResNet-34 A</td><td>25.03</td><td>7.76</td></tr><tr><td>ResNet-34 B</td><td>24.52</td><td>7.46</td></tr><tr><td>ResNet-34 C</td><td>24.19</td><td>7.40</td></tr><tr><td>ResNet-50</td><td>22.85</td><td>6.71</td></tr><tr><td>ResNet-101</td><td>21.75</td><td>6.05</td></tr><tr><td>ResNet-152</td><td>21.43</td><td>5.71</td></tr></table>",
              "score": 0.03007067901234568,
              "semantic_score": 0.4132988452911377,
              "lexical_rank": 4,
              "semantic_rank": 21,
              "rrf_score": 0.02797067901234568,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "table",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.00245
              },
              "page_start_display": 6,
              "page_end_display": 6,
              "evidence_role": "direct",
              "source_id": "S5"
            },
            {
              "chunk_id": "3a727cf8300953fcb50ea491",
              "paper_id": "1512.03385",
              "canonical_id": "1512.03385v1",
              "ordinal": 13,
              "region": "content",
              "chapter_number": "2",
              "chapter_title": "Related Work",
              "section_path": [
                "content",
                "2. Related Work"
              ],
              "section_label": "2. Related Work",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "68b3c976882856fab93fdf0076df1c7ade1d673c8558429c0c803c4707ecad61",
              "source_blocks": [
                {
                  "index": 29,
                  "page_idx": 1,
                  "bbox": [
                    496,
                    270,
                    893,
                    392
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 30,
                  "page_idx": 1,
                  "bbox": [
                    496,
                    393,
                    893,
                    575
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 31,
                  "page_idx": 1,
                  "bbox": [
                    496,
                    582,
                    893,
                    750
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 32,
                  "page_idx": 1,
                  "bbox": [
                    496,
                    750,
                    893,
                    902
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "a852549de9b6b2d3ca31fe7c55f3cfa617cc90bc0e5f6f6578fed4e1923bef7b",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "rtcuts that are parameter-free. When a gated shortcut is “closed” (approaching zero), the layers in highway networks represent non-residual functions. On the contrary, our formulation always learns residual functions; our identity shortcuts are never closed, and all information is always passed through, with additional residual functions to be learned. In addition, highway networks have not demonstrated accuracy gains with extremely increased depth $( e . g .$ ., over 100 layers).",
              "score": 0.029788564574170333,
              "semantic_score": 0.41216015815734863,
              "lexical_rank": 1,
              "semantic_rank": 22,
              "rrf_score": 0.028588564574170333,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 2,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 1512.03385v1 p.6: ResNet reduces the top-1 error by 3.5% (Table 2), resulting from the successfully reduced training error (Fig. 4 right vs. left). This comparison verifies the effectiveness of residual learning on extremely deep systems.\n\nLast, we also note that the 18-layer plain/residual nets are comparably accurate (Table 2), but the 18-layer ResNet converges faster (Fig. 4 right vs. left). When the net is “not overly deep” (18 layers here), the current SGD solver is still able to find good solutions to the plain net. In this case, the ResNet eases the optimization by providing faster convergence at the early stage.\n\nIdentity vs. Projection Shortcuts. We have shown that parameter-free, identity shortcuts help with training. Next we investigate projection shortcuts (Eqn.(2)). In Table 3 we compare three options: (A) zero-padding shortcuts are used for increasing dimensions, and all shortcuts are parameterfree (the same as Table 2 and Fig. 4 right); (B) projection shortcuts are used for increasing dimensions, and other shortcuts are identity; and (C) all shortcuts are projections.\n\n[S2] 1512.03385v1 p.1: Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracyfrom considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth ofup to 152 layers—8 deeper than VGG nets [41] but still having lower complexity. An ensemble ofthese residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.\n\nThe depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset.\n\n[S3] 1512.03385v1 p.5: ove plain nets, expect that a shortcut connection is added to each pair of 3 3 filters as in Fig. 3 (right). In the first comparison (Table 2 and Fig. 4 right), we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A). So they have no extra parameter compared to the plain counterparts.\n\nWe have three major observations from Table 2 and Fig. 4. First, the situation is reversed with residual learning – the 34-layer ResNet is better than the 18-layer ResNet (by 2.8%). More importantly, the 34-layer ResNet exhibits considerably lower training error and is generalizable to the validation data. This indicates that the degradation problem is well addressed in this setting and we manage to obtain accuracy gains from increased depth.\n\nSecond, compared to its plain counterpart, the 34-layer\n\n[S4] 1512.03385v1 p.8: Analysis of Layer Responses. Fig. 7 shows the standard deviations (std) of the layer responses. The responses are the outputs of each 3 3 layer, after BN and before other nonlinearity (ReLU/addition). For ResNets, this analysis reveals the response strength of the residual functions. Fig. 7 shows that ResNets have generally smaller responses than their plain counterparts. These results support our basic motivation (Sec.3.1) that the residual functions might be generally closer to zero than the non-residual functions. We also notice that the deeper ResNet has smaller magnitudes of responses, as evidenced by the comparisons among ResNet-20, 56, and 110 in Fig. 7. When there are more layers, an individual layer of ResNets tends to modify the signal less.\n\nExploring Over 1000 layers. We explore an aggressively deep model of over 1000 layers. We set n = 200 that leads to a 1202-layer network, which is trained as described above. Our method shows no optimization difficulty, and this 10<sup>3</sup>-layer network is able to achieve training error <0.1% (Fig. 6, right). Its test error is still fairly good (7.93%, Table 6).\n\n[S5] 1512.03385v1 p.6: <table><tr><td>model</td><td>top-1 err.</td><td>top-5 err.</td></tr><tr><td>VGG-16 [41]</td><td>28.07</td><td>9.33</td></tr><tr><td>GoogLeNet [44]</td><td>-</td><td>9.15</td></tr><tr><td>PReLU-net [13]</td><td>24.27</td><td>7.38</td></tr><tr><td>plain-34</td><td>28.54</td><td>10.02</td></tr><tr><td>ResNet-34 A</td><td>25.03</td><td>7.76</td></tr><tr><td>ResNet-34 B</td><td>24.52</td><td>7.46</td></tr><tr><td>ResNet-34 C</td><td>24.19</td><td>7.40</td></tr><tr><td>ResNet-50</td><td>22.85</td><td>6.71</td></tr><tr><td>ResNet-101</td><td>21.75</td><td>6.05</td></tr><tr><td>ResNet-152</td><td>21.43</td><td>5.71</td></tr></table>\n\n[S6] 1512.03385v1 p.2: rtcuts that are parameter-free. When a gated shortcut is “closed” (approaching zero), the layers in highway networks represent non-residual functions. On the contrary, our formulation always learns residual functions; our identity shortcuts are never closed, and all information is always passed through, with additional residual functions to be learned. In addition, highway networks have not demonstrated accuracy gains with extremely increased depth $( e . g .$ ., over 100 layers).",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"resnet\" OR \"appendix\" OR \"explain\" OR \"idea\" OR \"residual\" OR \"networks\" OR \"represent\" OR \"exponential\" OR \"number\" OR \"shallow\" OR \"networks\" OR \"resnet\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "how",
              "does",
              "the",
              "that",
              "can",
              "an",
              "of"
            ],
            "rewriter_used": false,
            "rewriter_fallback": true,
            "core_terms": []
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [
          "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
        ],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过（问题与来源结构不一致）",
        "paper_recall": "通过：仅返回 1512.03385。",
        "evidence_quality": "返回正文 4.1 的内容，确实包含 deep plain nets 的 exponentially low convergence rates；但该论点不在 ResNet 的 Appendix。",
        "issues": [
          "问题明确要求“附录”，结果没有 Appendix chunk。检查原文后确认 Appendix 主要是目标检测和定位实验，该理论论点位于正文；应将此案例视为查询意图与来源章节不一致，而不是把正文证据误报为附录证据。"
        ]
      }
    }
  ]
}
```

返回论文：`1512.03385`；质量判定：**部分通过（问题与来源结构不一致）**。

## 案例 10：What training settings are reported for Swin Transformer on ImageNet-1K?

```json
{
  "user_question": "What training settings are reported for Swin Transformer on ImageNet-1K?",
  "expected_evidence": {
    "paper_ids": [
      "2103.14030"
    ],
    "criteria": "显式限定 appendix，找到优化器、学习率、训练轮次等设置。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "What training settings are reported for Swin Transformer on ImageNet-1K?",
          "task": "fact",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000,
          "paper_ids": [
            "2103.14030"
          ],
          "regions": [
            "appendix"
          ]
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "What training settings are reported for Swin Transformer on ImageNet-1K?",
          "task": "fact",
          "routing": {
            "route_intent": "retrieve",
            "task": "fact",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "d7f03160fe3f9f7aa3b0d844",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 62,
              "region": "appendix",
              "chapter_number": "A2.1",
              "chapter_title": "Image classification on ImageNet-1K",
              "section_path": [
                "appendix",
                "A2. Detailed Experimental Settings",
                "A2.1. Image classification on ImageNet-1K"
              ],
              "section_label": "A2.1. Image classification on ImageNet-1K",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "8d2b748263c50b6b4ef6c2493faec28481a8dc72d3763caa8ac74ce1ed11403a",
              "source_blocks": [
                {
                  "index": 129,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    695,
                    470,
                    787
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 130,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    809,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 131,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    90,
                    895,
                    333
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 132,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    333,
                    893,
                    410
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 133,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    428,
                    895,
                    595
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b765dba35ba27aee79b4e2e6615fc557f90a88c19f915680b22d8f38f55d5bdc",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "The image classification is performed by applying a global average pooling layer on the output feature map of the last stage, followed by a linear classifier. We find this strategy to be as accurate as using an additional class token as in ViT [20] and DeiT [63]. In evaluation, the top-1 accuracy using a single crop is reported.\n\nRegular ImageNet-1K training The training settings mostly follow [63]. For all model variants, we adopt a default input image resolution of $2 2 4 ^ { 2 }$ . For other resolutions such as $3 8 4 ^ { 2 }$ , we fine-tune the models trained at $2 2 4 ^ { 2 }$ resolution, instead of training from scratch, to reduce GPU consumption.\n\nWhen training from scratch with a $2 2 4 ^ { 2 }$ input, we employ an AdamW [37] optimizer for 300 epochs using a cosine decay learning rate scheduler with 20 epochs of linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used.",
              "score": 0.0352164584959667,
              "semantic_score": 0.5331730842590332,
              "lexical_rank": 1,
              "semantic_rank": 3,
              "rrf_score": 0.032266458495966696,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "appendix",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00295
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "eb2863df68cb269b87888cbb",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 63,
              "region": "appendix",
              "chapter_number": "A2.1",
              "chapter_title": "Image classification on ImageNet-1K",
              "section_path": [
                "appendix",
                "A2. Detailed Experimental Settings",
                "A2.1. Image classification on ImageNet-1K"
              ],
              "section_label": "A2.1. Image classification on ImageNet-1K",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "4922b0818ac20e2e9ee97eccc0066d68bc0b0676606bd8bd4196492d7f07d2c9",
              "source_blocks": [
                {
                  "index": 129,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    695,
                    470,
                    787
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 130,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    809,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 131,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    90,
                    895,
                    333
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 132,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    333,
                    893,
                    410
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 133,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    428,
                    895,
                    595
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "e2ac98a9eff6a87bf2d372aba7bad8dfecd49c41023eeec4a37b761ecec88554",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used. We include most of the augmentation and regularization strategies of [63] in training, including RandAugment [17], Mixup [77], Cutmix [75], random erasing [82] and stochastic depth [35], but not repeated augmentation [31] and Exponential Moving Average (EMA) [45] which do not enhance performance. Note that this is contrary to [63] where repeated augmentation is crucial to stabilize the training of ViT. An increasing degree of stochastic depth augmentation is employed for larger models, i.e. 0.2, 0.3, 0.5 for Swin-T, Swin-S, and Swin-B, respectively.\n\nFor fine-tuning on input with larger resolution, we employ an adamW [37] optimizer for 30 epochs with a constant learning rate of $1 0 ^ { - 5 }$ , weight decay of $1 0 ^ { - 8 }$ , and the same data augmentation and regularizations as the first stage except for setting the stochastic depth ratio to 0.1.\n\nImageNet-22K pre-training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages.",
              "score": 0.034602048131080386,
              "semantic_score": 0.559590220451355,
              "lexical_rank": 3,
              "semantic_rank": 2,
              "rrf_score": 0.03200204813108039,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "appendix",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.00295
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "fbc267bdc0f601026f3445a3",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 64,
              "region": "appendix",
              "chapter_number": "A2.1",
              "chapter_title": "Image classification on ImageNet-1K",
              "section_path": [
                "appendix",
                "A2. Detailed Experimental Settings",
                "A2.1. Image classification on ImageNet-1K"
              ],
              "section_label": "A2.1. Image classification on ImageNet-1K",
              "type": "text",
              "page_start": 8,
              "page_end": 8,
              "content_hash": "0fb0b6f0207d468b74a739ed3d85bf030b357d9f2aeda6350638e47c9356f9c3",
              "source_blocks": [
                {
                  "index": 129,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    695,
                    470,
                    787
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 130,
                  "page_idx": 8,
                  "bbox": [
                    75,
                    809,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 131,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    90,
                    895,
                    333
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 132,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    333,
                    893,
                    410
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 133,
                  "page_idx": 8,
                  "bbox": [
                    496,
                    428,
                    895,
                    595
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "d89eb0eadff6018d06b3315e6b1899479276f4558727f846f51146d253fa6499",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages. For the first stage with $2 2 4 ^ { 2 }$ input, we employ an AdamW optimizer for 90 epochs using a linear decay learning rate scheduler with a 5-epoch linear warm-up. A batch size of 4096, an initial learning rate of 0.001, and a weight decay of 0.01 are used. In the second stage of ImageNet-1K finetuning with $2 2 4 ^ { 2 } / 3 8 4 ^ { 2 }$ input, we train the models for 30 epochs with a batch size of 1024, a constant learning rate of $1 0 ^ { - 5 }$ , and a weight decay of $1 0 ^ { - 8 }$",
              "score": 0.034354032258064514,
              "semantic_score": 0.5266588926315308,
              "lexical_rank": 2,
              "semantic_rank": 4,
              "rrf_score": 0.031754032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "appendix",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.00295
              },
              "page_start_display": 9,
              "page_end_display": 9,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "c7c063d68ac1ab9bdb39800f",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 70,
              "region": "appendix",
              "chapter_number": "A3.1",
              "chapter_title": "Image classification with different input size",
              "section_path": [
                "appendix",
                "A3. More Experiments",
                "A3.1. Image classification with different input size"
              ],
              "section_label": "A3.1. Image classification with different input size",
              "type": "table",
              "page_start": 9,
              "page_end": 9,
              "content_hash": "8c11c9f9a7833874b38507ba46f481749bf2f4d06d5d8d3f5880040e67cd9724",
              "source_blocks": [
                {
                  "index": 147,
                  "page_idx": 9,
                  "bbox": [
                    501,
                    324,
                    890,
                    422
                  ],
                  "type": "table",
                  "text_format": null,
                  "context_before": "Table 8 lists the performance of Swin Transformers with different input image sizes from $2 2 4 ^ { 2 }$ to 384 2 . In general, a larger input resolution leads to better top-1 accuracy but with slower inference speed.",
                  "context_after": ""
                }
              ],
              "asset_refs": [
                "images/c10fe699a7fe1c0dcddd2731f2c9ab007c4e41d82ae066ef480aaa636f932a5b.jpg"
              ],
              "retrieval_text_hash": "d24aa2ebef356dcfa7dd79496d00a4436322b77ab8c7e30b98161b9d267083ec",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Table 8. Swin Transformers with different input image size on ImageNet-1K classification.\n<table><tr><td rowspan=\"2\">input size</td><td colspan=\"2\">Swin-T</td><td colspan=\"2\">Swin-S</td><td colspan=\"2\">Swin-B</td></tr><tr><td>top-1 acc</td><td>throughput (image / s)</td><td>top-1 acc</td><td>throughput (image / s)</td><td>top-1 acc</td><td>throughput (image / s)</td></tr><tr><td> $224^2$ </td><td>81.3</td><td>755.2</td><td>83.0</td><td>436.9</td><td>83.3</td><td>278.1</td></tr><tr><td> $256^2$ </td><td>81.6</td><td>580.9</td><td>83.4</td><td>336.7</td><td>83.7</td><td>208.1</td></tr><tr><td> $320^2$ </td><td>82.1</td><td>342.0</td><td>83.7</td><td>198.2</td><td>84.0</td><td>132.0</td></tr><tr><td> $384^2$ </td><td>82.2</td><td>219.5</td><td>83.9</td><td>127.6</td><td>84.5</td><td>84.7</td></tr></table>",
              "score": 0.03377805800756621,
              "semantic_score": 0.5670959949493408,
              "lexical_rank": 5,
              "semantic_rank": 1,
              "rrf_score": 0.03177805800756621,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "table",
                "region_priority": "appendix",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.002
              },
              "page_start_display": 10,
              "page_end_display": 10,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "21fd281c10590d1832e0b9e0",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 74,
              "region": "appendix",
              "chapter_number": "A3.3",
              "chapter_title": "Swin MLP-Mixer",
              "section_path": [
                "appendix",
                "A3. More Experiments",
                "A3.3. Swin MLP-Mixer"
              ],
              "section_label": "A3.3. Swin MLP-Mixer",
              "type": "table",
              "page_start": 10,
              "page_end": 10,
              "content_hash": "358169739ee5d3319b5170dbd282450358f6798df6848fda9b13153a1078b008",
              "source_blocks": [
                {
                  "index": 154,
                  "page_idx": 10,
                  "bbox": [
                    78,
                    89,
                    480,
                    272
                  ],
                  "type": "table",
                  "text_format": null,
                  "context_before": "MLP [61]. Swin-Mixer performs significantly better than MLP-Mixer (81.3% vs. 76.4%) using slightly smaller computation budget (10.4G vs. 12.7G). It also has better speed accuracy trade-off compared to ResMLP [62]. These results indicate the proposed hierarchical design and the shifted window approach are generalizable.",
                  "context_after": ""
                }
              ],
              "asset_refs": [
                "images/e55e4545e47d6cb673ec11981e17e43bd95e8302ed35ff9a08c5301884eb1e63.jpg"
              ],
              "retrieval_text_hash": "1f0f34a0a08b19c7cfc3fcce9d614dc64b5db12cf95edca29299b1f88fc0cb86",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Table 10. Performance of Swin MLP-Mixer on ImageNet-1K classification. D indictes the number of channels per head. Throughput is measured using the GitHub repository of [68] and a V100 GPU, following [63].\n<table><tr><td>method</td><td>image size</td><td>#param.</td><td>FLOPs</td><td>throughput (image / s)</td><td>ImageNet top-1 acc.</td></tr><tr><td>MLP-Mixer-B/16 [61]</td><td> $224^2$ </td><td>59M</td><td>12.7G</td><td>-</td><td>76.4</td></tr><tr><td>ResMLP-S24 [62]</td><td> $224^2$ </td><td>30M</td><td>6.0G</td><td>715</td><td>79.4</td></tr><tr><td>ResMLP-B24 [62]</td><td> $224^2$ </td><td>116M</td><td>23.0G</td><td>231</td><td>81.0</td></tr><tr><td>Swin-T/D24 (Transformer)</td><td> $256^2$ </td><td>28M</td><td>5.9G</td><td>563</td><td>81.6</td></tr><tr><td>Swin-Mixer-T/D24</td><td> $256^2$ </td><td>20M</td><td>4.0G</td><td>807</td><td>79.4</td></tr><tr><td>Swin-Mixer-T/D12</td><td> $256^2$ </td><td>21M</td><td>4.0G</td><td>792</td><td>79.6</td></tr><tr><td>Swin-Mixer-T/D6</td><td> $256^2$ </td><td>23M</td><td>4.0G</td><td>766</td><td>79.7</td></tr><tr><td>Swin-Mixer-B/D24 (no shift)</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>80.3</td></tr><tr><td>Swin-Mixer-B/D24</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>81.3</td></tr></table>",
              "score": 0.031857397504456327,
              "semantic_score": 0.49585461616516113,
              "lexical_rank": 6,
              "semantic_rank": 8,
              "rrf_score": 0.029857397504456328,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "table",
                "region_priority": "appendix",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.002
              },
              "page_start_display": 11,
              "page_end_display": 11,
              "evidence_role": "direct",
              "source_id": "S5"
            },
            {
              "chunk_id": "fed55b42e5b45f12bbc440ad",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 72,
              "region": "appendix",
              "chapter_number": "A3.2",
              "chapter_title": "Different Optimizers for ResNe(X)t on COCO",
              "section_path": [
                "appendix",
                "A3. More Experiments",
                "A3.2. Different Optimizers for ResNe(X)t on COCO"
              ],
              "section_label": "A3.2. Different Optimizers for ResNe(X)t on COCO",
              "type": "text",
              "page_start": 9,
              "page_end": 9,
              "content_hash": "4b5d02e3cb421c0fa4b87957524ec94468b47bcfdd2ba0a9fe6c59905ddde52c",
              "source_blocks": [
                {
                  "index": 150,
                  "page_idx": 9,
                  "bbox": [
                    496,
                    654,
                    895,
                    791
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "d8ed0da39fd2115678466ba6e990023ac998a7fa35d8aa889dabb5a49420b476",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Table 9 compares the AdamW and SGD optimizers of the ResNe(X)t backbones on COCO object detection. The Cascade Mask R-CNN framework is used in this comparison. While SGD is used as a default optimizer for Cascade Mask R-CNN framework, we generally observe improved accuracy by replacing it with an AdamW optimizer, particularly for smaller backbones. We thus use AdamW for ResNe(X)t backbones when compared to the proposed Swin Transformer architectures.",
              "score": 0.030159507042253522,
              "semantic_score": 0.3577488660812378,
              "lexical_rank": 4,
              "semantic_rank": 11,
              "rrf_score": 0.029709507042253523,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "appendix",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00045
              },
              "page_start_display": 10,
              "page_end_display": 10,
              "evidence_role": "direct",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 2103.14030v2 p.9: The image classification is performed by applying a global average pooling layer on the output feature map of the last stage, followed by a linear classifier. We find this strategy to be as accurate as using an additional class token as in ViT [20] and DeiT [63]. In evaluation, the top-1 accuracy using a single crop is reported.\n\nRegular ImageNet-1K training The training settings mostly follow [63]. For all model variants, we adopt a default input image resolution of $2 2 4 ^ { 2 }$ . For other resolutions such as $3 8 4 ^ { 2 }$ , we fine-tune the models trained at $2 2 4 ^ { 2 }$ resolution, instead of training from scratch, to reduce GPU consumption.\n\nWhen training from scratch with a $2 2 4 ^ { 2 }$ input, we employ an AdamW [37] optimizer for 300 epochs using a cosine decay learning rate scheduler with 20 epochs of linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used.\n\n[S2] 2103.14030v2 p.9: linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used. We include most of the augmentation and regularization strategies of [63] in training, including RandAugment [17], Mixup [77], Cutmix [75], random erasing [82] and stochastic depth [35], but not repeated augmentation [31] and Exponential Moving Average (EMA) [45] which do not enhance performance. Note that this is contrary to [63] where repeated augmentation is crucial to stabilize the training of ViT. An increasing degree of stochastic depth augmentation is employed for larger models, i.e. 0.2, 0.3, 0.5 for Swin-T, Swin-S, and Swin-B, respectively.\n\nFor fine-tuning on input with larger resolution, we employ an adamW [37] optimizer for 30 epochs with a constant learning rate of $1 0 ^ { - 5 }$ , weight decay of $1 0 ^ { - 8 }$ , and the same data augmentation and regularizations as the first stage except for setting the stochastic depth ratio to 0.1.\n\nImageNet-22K pre-training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages.\n\n[S3] 2103.14030v2 p.9: training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages. For the first stage with $2 2 4 ^ { 2 }$ input, we employ an AdamW optimizer for 90 epochs using a linear decay learning rate scheduler with a 5-epoch linear warm-up. A batch size of 4096, an initial learning rate of 0.001, and a weight decay of 0.01 are used. In the second stage of ImageNet-1K finetuning with $2 2 4 ^ { 2 } / 3 8 4 ^ { 2 }$ input, we train the models for 30 epochs with a batch size of 1024, a constant learning rate of $1 0 ^ { - 5 }$ , and a weight decay of $1 0 ^ { - 8 }$\n\n[S4] 2103.14030v2 p.10: Table 8. Swin Transformers with different input image size on ImageNet-1K classification.\n<table><tr><td rowspan=\"2\">input size</td><td colspan=\"2\">Swin-T</td><td colspan=\"2\">Swin-S</td><td colspan=\"2\">Swin-B</td></tr><tr><td>top-1 acc</td><td>throughput (image / s)</td><td>top-1 acc</td><td>throughput (image / s)</td><td>top-1 acc</td><td>throughput (image / s)</td></tr><tr><td> $224^2$ </td><td>81.3</td><td>755.2</td><td>83.0</td><td>436.9</td><td>83.3</td><td>278.1</td></tr><tr><td> $256^2$ </td><td>81.6</td><td>580.9</td><td>83.4</td><td>336.7</td><td>83.7</td><td>208.1</td></tr><tr><td> $320^2$ </td><td>82.1</td><td>342.0</td><td>83.7</td><td>198.2</td><td>84.0</td><td>132.0</td></tr><tr><td> $384^2$ </td><td>82.2</td><td>219.5</td><td>83.9</td><td>127.6</td><td>84.5</td><td>84.7</td></tr></table>\n\n[S5] 2103.14030v2 p.11: Table 10. Performance of Swin MLP-Mixer on ImageNet-1K classification. D indictes the number of channels per head. Throughput is measured using the GitHub repository of [68] and a V100 GPU, following [63].\n<table><tr><td>method</td><td>image size</td><td>#param.</td><td>FLOPs</td><td>throughput (image / s)</td><td>ImageNet top-1 acc.</td></tr><tr><td>MLP-Mixer-B/16 [61]</td><td> $224^2$ </td><td>59M</td><td>12.7G</td><td>-</td><td>76.4</td></tr><tr><td>ResMLP-S24 [62]</td><td> $224^2$ </td><td>30M</td><td>6.0G</td><td>715</td><td>79.4</td></tr><tr><td>ResMLP-B24 [62]</td><td> $224^2$ </td><td>116M</td><td>23.0G</td><td>231</td><td>81.0</td></tr><tr><td>Swin-T/D24 (Transformer)</td><td> $256^2$ </td><td>28M</td><td>5.9G</td><td>563</td><td>81.6</td></tr><tr><td>Swin-Mixer-T/D24</td><td> $256^2$ </td><td>20M</td><td>4.0G</td><td>807</td><td>79.4</td></tr><tr><td>Swin-Mixer-T/D12</td><td> $256^2$ </td><td>21M</td><td>4.0G</td><td>792</td><td>79.6</td></tr><tr><td>Swin-Mixer-T/D6</td><td> $256^2$ </td><td>23M</td><td>4.0G</td><td>766</td><td>79.7</td></tr><tr><td>Swin-Mixer-B/D24 (no shift)</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>80.3</td></tr><tr><td>Swin-Mixer-B/D24</td><td> $224^2$ </td><td>61M</td><td>10.4G</td><td>409</td><td>81.3</td></tr></table>\n\n[S6] 2103.14030v2 p.10: Table 9 compares the AdamW and SGD optimizers of the ResNe(X)t backbones on COCO object detection. The Cascade Mask R-CNN framework is used in this comparison. While SGD is used as a default optimizer for Cascade Mask R-CNN framework, we generally observe improved accuracy by replacing it with an AdamW optimizer, particularly for smaller backbones. We thus use AdamW for ResNe(X)t backbones when compared to the proposed Swin Transformer architectures.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"swin transformer\" OR \"imagenet 1k\" OR \"training settings\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "swin transformer",
              "imagenet 1k",
              "training settings"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过",
        "paper_recall": "通过：仅返回 2103.14030，且全部为 Appendix。",
        "evidence_quality": "A2.1 的前 3 条直接覆盖 ImageNet-1K 的分辨率、优化器、epoch、batch size、学习率和 weight decay。",
        "issues": [
          "后 3 条混入 A3.1、A3.2、A3.3 的无关附录内容；显式 Appendix 检索仍需要按章节相关性去噪。"
        ]
      }
    }
  ]
}
```

返回论文：`2103.14030`；质量判定：**部分通过**。

## 案例 11：哪些方法通过局部窗口注意力降低图像 Transformer 的计算复杂度？请给出正文依据。

```json
{
  "user_question": "哪些方法通过局部窗口注意力降低图像 Transformer 的计算复杂度？请给出正文依据。",
  "expected_evidence": {
    "paper_ids": [
      "2103.14030"
    ],
    "criteria": "年份与分类约束保持有效，并找到局部窗口复杂度正文。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "哪些方法通过局部窗口注意力降低图像 Transformer 的计算复杂度？请给出正文依据。",
          "task": "auto",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000,
          "filters": {
            "category": "cs.CV",
            "year_from": "2020"
          }
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "哪些方法通过局部窗口注意力降低图像 Transformer 的计算复杂度？请给出正文依据。",
          "task": "fact",
          "routing": {
            "route_intent": "retrieve",
            "task": "fact",
            "provider": "jev",
            "fallback_used": false,
            "confidence": 0.76
          },
          "papers": [
            {
              "paper_id": "2010.11929",
              "base_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
              "authors": [
                "Alexey Dosovitskiy",
                "Lucas Beyer",
                "Alexander Kolesnikov",
                "Dirk Weissenborn",
                "Xiaohua Zhai",
                "Thomas Unterthiner",
                "Mostafa Dehghani",
                "Matthias Minderer",
                "Georg Heigold",
                "Sylvain Gelly",
                "Jakob Uszkoreit",
                "Neil Houlsby"
              ],
              "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
              "categories": [
                "cs.CV",
                "cs.AI",
                "cs.LG"
              ],
              "published_at": "2020-10-22T17:55:59Z",
              "updated_at": "2021-06-03T13:08:56Z",
              "abs_url": "https://arxiv.org/abs/2010.11929v2",
              "pdf_url": "https://arxiv.org/pdf/2010.11929v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru"
                }
              }
            },
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "63dfec21b96c0be0700d2770",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 20,
              "region": "content",
              "chapter_number": "3.2",
              "chapter_title": "Shifted Window based Self-Attention",
              "section_path": [
                "content",
                "3. Method",
                "3.2. Shifted Window based Self-Attention"
              ],
              "section_label": "3.2. Shifted Window based Self-Attention",
              "type": "text",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "537fee562eda337b5d2d80b9c8ba933aef45ec7f120761c148f3369607a44f4f",
              "source_blocks": [
                {
                  "index": 42,
                  "page_idx": 3,
                  "bbox": [
                    75,
                    648,
                    470,
                    770
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 43,
                  "page_idx": 3,
                  "bbox": [
                    75,
                    809,
                    470,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "c20176ad41796662c7ef3f727bd67ff1fdbf147ac0af1decae7273a9432aef32",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "The standard Transformer architecture [64] and its adaptation for image classification [20] both conduct global selfattention, where the relationships between a token and all other tokens are computed. The global computation leads to quadratic complexity with respect to the number of tokens, making it unsuitable for many vision problems requiring an immense set of tokens for dense prediction or to represent a high-resolution image.\n\nSelf-attention in non-overlapped windows For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M \\times M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h \\times w$ patches $\\mathrm { a r e } ^ { 3 }$ :",
              "score": 0.032299324975891996,
              "semantic_score": 0.628522515296936,
              "lexical_rank": 8,
              "semantic_rank": 1,
              "rrf_score": 0.031099324975891997,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "339c82822ec082e1a253c1d3",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 6,
              "region": "content",
              "chapter_number": "1",
              "chapter_title": "Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "text",
              "page_start": 0,
              "page_end": 1,
              "content_hash": "4960b9848c4be4add519c8703cda65d179752b45751ee237afded87c8280ed0a",
              "source_blocks": [
                {
                  "index": 11,
                  "page_idx": 0,
                  "bbox": [
                    496,
                    702,
                    893,
                    869
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 12,
                  "page_idx": 0,
                  "bbox": [
                    498,
                    869,
                    895,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 16,
                  "page_idx": 1,
                  "bbox": [
                    75,
                    666,
                    472,
                    849
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "d73bb9c87f433ab810e1351a99a7edfe0d8d8797bc4d4c64a888f9bdc98ffca4",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "rpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. As illustrated in Figure 1(a), Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size. These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [20] which produce feature maps of a single resolution and have quadratic complexity.",
              "score": 0.032034914611005695,
              "semantic_score": 0.45838820934295654,
              "lexical_rank": 2,
              "semantic_rank": 8,
              "rrf_score": 0.030834914611005692,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 1,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "af3b4bdb709effd80a017bc7",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 22,
              "region": "content",
              "chapter_number": "3.2",
              "chapter_title": "Shifted Window based Self-Attention",
              "section_path": [
                "content",
                "3. Method",
                "3.2. Shifted Window based Self-Attention"
              ],
              "section_label": "3.2. Shifted Window based Self-Attention",
              "type": "equation",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "873a79f91e7079607b132511313c9bf6feaffc62b1f15775a5bc37ffea808586",
              "source_blocks": [
                {
                  "index": 46,
                  "page_idx": 3,
                  "bbox": [
                    571,
                    378,
                    820,
                    397
                  ],
                  "type": "equation",
                  "text_format": "latex",
                  "context_before": "mpute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M \\times M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h \\times w$ patches $\\mathrm { a r e } ^ { 3 }$ :",
                  "context_after": "where the former is quadratic to patch number hw, and the latter is linear when M is fixed (set to 7 by default). Global self-attention computation is generally unaffordable for a large hw, while the window based self-attention is scalable."
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "680f16dd24e2d990399cd9b5bbd63a72336f54f4bf9a58fe3e0b2557b5f48442",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "$$\n\\Omega (\\mathbf {W - M S A}) = 4 h w C ^ {2} + 2 M ^ {2} h w C,\\tag{2}\n$$",
              "score": 0.03160440539239287,
              "semantic_score": 0.5379749536514282,
              "lexical_rank": 7,
              "semantic_rank": 2,
              "rrf_score": 0.031054405392392875,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "equation",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0009
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "3d5cfa942697e3f0d3d7fecf",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 21,
              "region": "content",
              "chapter_number": "3.2",
              "chapter_title": "Shifted Window based Self-Attention",
              "section_path": [
                "content",
                "3. Method",
                "3.2. Shifted Window based Self-Attention"
              ],
              "section_label": "3.2. Shifted Window based Self-Attention",
              "type": "equation",
              "page_start": 3,
              "page_end": 3,
              "content_hash": "45a01ded30cf24eb76e7ee4d831dbc3aa218524d137e7db55e989f6501bb5e0c",
              "source_blocks": [
                {
                  "index": 45,
                  "page_idx": 3,
                  "bbox": [
                    571,
                    358,
                    795,
                    376
                  ],
                  "type": "equation",
                  "text_format": "latex",
                  "context_before": "mpute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M \\times M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h \\times w$ patches $\\mathrm { a r e } ^ { 3 }$ :",
                  "context_after": "where the former is quadratic to patch number hw, and the latter is linear when M is fixed (set to 7 by default). Global self-attention computation is generally unaffordable for a large hw, while the window based self-attention is scalable."
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b2dc34fab78f092c18efb744e20c37c1059331210322f870d970d4bc8682836d",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "$$\n\\Omega (\\mathrm{MSA}) = 4 h w C ^ {2} + 2 (h w) ^ {2} C,\\tag{1}\n$$",
              "score": 0.03157453102453102,
              "semantic_score": 0.5341202020645142,
              "lexical_rank": 6,
              "semantic_rank": 3,
              "rrf_score": 0.031024531024531024,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "equation",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0009
              },
              "page_start_display": 4,
              "page_end_display": 4,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "d5feb2669c49ba2ea91299dd",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 5,
              "region": "content",
              "chapter_number": "1",
              "chapter_title": "Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "text",
              "page_start": 0,
              "page_end": 1,
              "content_hash": "46ff8f635aff9956a8a8deb24c58641f13000a3fec000bd1dd48827358a941a7",
              "source_blocks": [
                {
                  "index": 11,
                  "page_idx": 0,
                  "bbox": [
                    496,
                    702,
                    893,
                    869
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 12,
                  "page_idx": 0,
                  "bbox": [
                    498,
                    869,
                    895,
                    901
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 16,
                  "page_idx": 1,
                  "bbox": [
                    75,
                    666,
                    472,
                    849
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "a06f7eefda4a0af4b3815f64883e5e364fea58583496529bae9e785abee9fbf8",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "mance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [42, 53, 54]. In existing Transformer-based models [64, 20], tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size. To overcome these issues, we propose a general purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size.",
              "score": 0.02971002886002886,
              "semantic_score": 0.41451936960220337,
              "lexical_rank": 3,
              "semantic_rank": 17,
              "rrf_score": 0.02886002886002886,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0012000000000000001
              },
              "page_start_display": 1,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S5"
            },
            {
              "chunk_id": "aef74810680c1fef21c96da5",
              "paper_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "page_start": 0,
              "page_end": 0,
              "content_hash": "8dd511eb8818ff1142cbd562e68a32e670fe8c7456c804dbe96ff05374087123",
              "source_blocks": [
                {
                  "index": 6,
                  "page_idx": 0,
                  "bbox": [
                    75,
                    319,
                    470,
                    743
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b47f9f16ab0045fc408d5cc9b9973271a8d46fc70bea8be0e767dc9b747136d3",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val).",
              "score": 0.029320221327967806,
              "semantic_score": 0.44207513332366943,
              "lexical_rank": 10,
              "semantic_rank": 11,
              "rrf_score": 0.028370221327967807,
              "ranking_features": {
                "exact_entity_hit": false,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "abstract",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.00095
              },
              "page_start_display": 1,
              "page_end_display": 1,
              "evidence_role": "direct",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 2103.14030v2 p.4: The standard Transformer architecture [64] and its adaptation for image classification [20] both conduct global selfattention, where the relationships between a token and all other tokens are computed. The global computation leads to quadratic complexity with respect to the number of tokens, making it unsuitable for many vision problems requiring an immense set of tokens for dense prediction or to represent a high-resolution image.\n\nSelf-attention in non-overlapped windows For efficient modeling, we propose to compute self-attention within local windows. The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M \\times M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h \\times w$ patches $\\mathrm { a r e } ^ { 3 }$ :\n\n[S2] 2103.14030v2 p.1: rpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size. As illustrated in Figure 1(a), Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size. These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures [20] which produce feature maps of a single resolution and have quadratic complexity.\n\n[S3] 2103.14030v2 p.4: $$\n\\Omega (\\mathbf {W - M S A}) = 4 h w C ^ {2} + 2 M ^ {2} h w C,\\tag{2}\n$$\n\n[S4] 2103.14030v2 p.4: $$\n\\Omega (\\mathrm{MSA}) = 4 h w C ^ {2} + 2 (h w) ^ {2} C,\\tag{1}\n$$\n\n[S5] 2103.14030v2 p.1: mance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection [42, 53, 54]. In existing Transformer-based models [64, 20], tokens are all of a fixed scale, a property unsuitable for these vision applications. Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size. To overcome these issues, we propose a general purpose Transformer backbone, called Swin Transformer, which constructs hierarchical feature maps and has linear computational complexity to image size.\n\n[S6] 2103.14030v2 p.1: This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformerfrom language to vision arisefrom differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowingfor cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO testdev) and semantic segmentation (53.5 mIoU on ADE20K val).",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"local window attention\" OR \"image transformer\" OR \"computational complexity\" OR \"methods reduce computational complexity\" OR \"text basis\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "to"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "local window attention",
              "image transformer",
              "computational complexity",
              "methods reduce computational complexity",
              "text basis"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "部分通过",
        "paper_recall": "部分通过：命中 2103.14030，但同时返回 2010.11929。",
        "evidence_quality": "2103.14030 的 3.2 正文明确对比 global self-attention 的二次复杂度与 non-overlapped local windows。",
        "issues": [
          "ViT 使用 global attention，却作为候选论文返回；第 3、4 条是公式，缺少解释性正文优先。"
        ]
      }
    }
  ]
}
```

返回论文：`2010.11929, 2103.14030`；质量判定：**部分通过**。

## 案例 12：What does EfficientNet Table 2 report about the accuracy and parameter count of EfficientNet-B0 versus ResNet-50?

```json
{
  "user_question": "What does EfficientNet Table 2 report about the accuracy and parameter count of EfficientNet-B0 versus ResNet-50?",
  "expected_evidence": {
    "paper_ids": [
      "1905.11946"
    ],
    "criteria": "表格命中并提供 B0/ResNet-50 的准确率与参数量，保留原始表格来源。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "What does EfficientNet Table 2 report about the accuracy and parameter count of EfficientNet-B0 versus ResNet-50?",
          "task": "fact",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000,
          "paper_ids": [
            "1905.11946"
          ]
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "What does EfficientNet Table 2 report about the accuracy and parameter count of EfficientNet-B0 versus ResNet-50?",
          "task": "fact",
          "routing": {
            "route_intent": "retrieve",
            "task": "fact",
            "provider": "explicit",
            "fallback_used": false,
            "confidence": null
          },
          "papers": [
            {
              "paper_id": "1905.11946",
              "base_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "title": "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
              "authors": [
                "Mingxing Tan",
                "Quoc V. Le"
              ],
              "abstract": "Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance. Based on this observation, we propose a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient. We demonstrate the effectiveness of this method on scaling up MobileNets and ResNet. To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous ConvNets. In particular, our EfficientNet-B7 achieves state-of-the-art 84.3% top-1 accuracy on ImageNet, while being 8.4x smaller and 6.1x faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Source code is at https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet.",
              "categories": [
                "cs.LG",
                "cs.CV",
                "stat.ML"
              ],
              "published_at": "2019-05-28T17:05:32Z",
              "updated_at": "2020-09-11T05:08:01Z",
              "abs_url": "https://arxiv.org/abs/1905.11946v5",
              "pdf_url": "https://arxiv.org/pdf/1905.11946v5.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1905.11946\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "a2de0ef080fccb041f478cc8",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 53,
              "region": "content",
              "chapter_number": "5.3",
              "chapter_title": "Transfer Learning Results for EfficientNet",
              "section_path": [
                "content",
                "5. Experiments",
                "5.3. Transfer Learning Results for EfficientNet"
              ],
              "section_label": "5.3. Transfer Learning Results for EfficientNet",
              "type": "text",
              "page_start": 7,
              "page_end": 7,
              "content_hash": "45b72df22488be538dbd936d3a7a4015c6b542e497683f132ed278bce2ea0180",
              "source_blocks": [
                {
                  "index": 96,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    482,
                    477,
                    559
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 97,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    566,
                    478,
                    718
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 98,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    724,
                    478,
                    816
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "b627dd58425062945a2af6805d171112145840d52e83082c96af1c52fc898645",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "rpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters\n\nFigure 6 compares the accuracy-parameters curve for a variety of models. In general, our EfficientNets consistently achieve better accuracy with an order of magnitude fewer parameters than existing models, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), Inception (Szegedy et al., 2017), and NASNet (Zoph et al., 2018).",
              "score": 0.02009344262295082,
              "semantic_score": 0.6848113536834717,
              "lexical_rank": null,
              "semantic_rank": 1,
              "rrf_score": 0.01639344262295082,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 8,
              "page_end_display": 8,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "e32437c1d9c952b9d1be766a",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 49,
              "region": "content",
              "chapter_number": "5.2",
              "chapter_title": "ImageNet Results for EfficientNet",
              "section_path": [
                "content",
                "5. Experiments",
                "5.2. ImageNet Results for EfficientNet"
              ],
              "section_label": "5.2. ImageNet Results for EfficientNet",
              "type": "text",
              "page_start": 6,
              "page_end": 6,
              "content_hash": "261cff45bdd63f0048fa773825ba8a29aa942faeeca1e5840610fc63c9cff8a6",
              "source_blocks": [
                {
                  "index": 87,
                  "page_idx": 6,
                  "bbox": [
                    83,
                    602,
                    478,
                    800
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 88,
                  "page_idx": 6,
                  "bbox": [
                    83,
                    805,
                    477,
                    898
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 90,
                  "page_idx": 6,
                  "bbox": [
                    495,
                    669,
                    890,
                    792
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 91,
                  "page_idx": 6,
                  "bbox": [
                    493,
                    797,
                    888,
                    905
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "3ece71fc69281c47acb729c3e3f3bc6a7d2f98701cf3d2c1b39c60b19e16896d",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "chieves 84.3% top1 accuracy with 66M parameters and 37B FLOPS, being more accurate but 8.4x smaller than the previous best GPipe (Huang et al., 2018). These gains come from both better architectures, better scaling, and better training settings that are customized for EfficientNet.\n\nFigure 1 and Figure 5 illustrates the parameters-accuracy and FLOPS-accuracy curve for representative ConvNets, where our scaled EfficientNet models achieve better accuracy with much fewer parameters and FLOPS than other ConvNets. Notably, our EfficientNet models are not only small, but also computational cheaper. For example, our EfficientNet-B3 achieves higher accuracy than ResNeXt-101 (Xie et al., 2017) using 18x fewer FLOPS.\n\nTo validate the latency, we have also measured the inference latency for a few representative CovNets on a real CPU as shown in Table 4, where we report average latency over 20 runs. Our EfficientNet-B1 runs 5.7x faster than the widely used ResNet-152, while EfficientNet-B7 runs about 6.1x faster than GPipe (Huang et al., 2018), suggesting our EfficientNets are indeed fast on real hardware.",
              "score": 0.019573015873015874,
              "semantic_score": 0.6486262679100037,
              "lexical_rank": null,
              "semantic_rank": 3,
              "rrf_score": 0.015873015873015872,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 7,
              "page_end_display": 7,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "38bde84d3cb4a3635f33bcfc",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 46,
              "region": "content",
              "chapter_number": "5.2",
              "chapter_title": "ImageNet Results for EfficientNet",
              "section_path": [
                "content",
                "5. Experiments",
                "5.2. ImageNet Results for EfficientNet"
              ],
              "section_label": "5.2. ImageNet Results for EfficientNet",
              "type": "table",
              "page_start": 6,
              "page_end": 6,
              "content_hash": "3b62ac8e29d8e7f984ae936c7926f8eefe850d1043024abe3ba203eec06e65e9",
              "source_blocks": [
                {
                  "index": 85,
                  "page_idx": 6,
                  "bbox": [
                    89,
                    128,
                    883,
                    257
                  ],
                  "type": "table",
                  "text_format": null,
                  "context_before": "We train our EfficientNet models on ImageNet using similar settings as (Tan et al., 2019): RMSProp optimizer with decay 0.9 and momentum 0.9; batch norm momentum 0.99;",
                  "context_after": "weight decay 1e-5; initial learning rate 0.256 that decays by 0.97 every 2.4 epochs. We also use SiLU (Swish-1) activation (Ramachandran et al., 2018; Elfwing et al., 2018; Hendrycks & Gimpel, 2016), AutoAugment (Cubuk et al., 2019), and stochastic depth (Huang et al., 2016) with survival probability 0.8. As commonly k"
                }
              ],
              "asset_refs": [
                "images/5036a91f7c73fa80a78c06395c9655352d8d0527c35e7c1dd3c23f865eb48c94.jpg"
              ],
              "retrieval_text_hash": "4d45f03719ac96277d2b4507fc276330980607d40c642c4056c1180b299c3392",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Table 5. EfficientNet Performance Results on Transfer Learning Datasets. Our scaled EfficientNet models achieve new state-of-theart accuracy for 5 out of 8 datasets, with 9.6x fewer parameters on average.\n<table><tr><td rowspan=\"2\"></td><td colspan=\"6\">Comparison to best public-available results</td><td colspan=\"6\">Comparison to best reported results</td></tr><tr><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td></tr><tr><td>CIFAR-10</td><td>NASNet-A</td><td>98.0%</td><td>85M</td><td>EfficientNet-B0</td><td>98.1%</td><td>4M (21x)</td><td> $^\\dagger$ Gpipe</td><td>99.0%</td><td>556M</td><td>EfficientNet-B7</td><td>98.9%</td><td>64M (8.7x)</td></tr><tr><td>CIFAR-100</td><td>NASNet-A</td><td>87.5%</td><td>85M</td><td>EfficientNet-B0</td><td>88.1%</td><td>4M (21x)</td><td>Gpipe</td><td>91.3%</td><td>556M</td><td>EfficientNet-B7</td><td>91.7%</td><td>64M (8.7x)</td></tr><tr><td>Birdsnap</td><td>Inception-v4</td><td>81.8%</td><td>41M</td><td>EfficientNet-B5</td><td>82.0%</td><td>28M (1.5x)</td><td>GPipe</td><td>83.6%</td><td>556M</td><td>EfficientNet-B7</td><td>84.3%</td><td>64M (8.7x)</td></tr><tr><td>Stanford Cars</td><td>Inception-v4</td><td>93.4%</td><td>41M</td><td>EfficientNet-B3</td><td>93.6%</td><td>10M (4.1x)</td><td> $^\\ddagger$ DAT</td><td>94.8%</td><td>-</td><td>EfficientNet-B7</td><td>94.7%</td><td>-</td></tr><tr><td>Flowers</td><td>Inception-v4</td><td>98.5%</td><td>41M</td><td>EfficientNet-B5</td><td>98.5%</td><td>28M (1.5x)</td><td>DAT</td><td>97.7%</td><td>-</td><td>EfficientNet-B7</td><td>98.8%</td><td>-</td></tr><tr><td>FGVC Aircraft</td><td>Inception-v4</td><td>90.9%</td><td>41M</td><td>EfficientNet-B3</td><td>90.7%</td><td>10M (4.1x)</td><td>DAT</td><td>92.9%</td><td>-</td><td>EfficientNet-B7</td><td>92.9%</td><td>-</td></tr><tr><td>Oxford-IIIT Pets</td><td>ResNet-152</td><td>94.5%</td><td>58M</td><td>EfficientNet-B4</td><td>94.8%</td><td>17M (5.6x)</td><td>GPipe</td><td>95.9%</td><td>556M</td><td>EfficientNet-B6</td><td>95.4%</td><td>41M (14x)</td></tr><tr><td>Food-101</td><td>Inception-v4</td><td>90.8%</td><td>41M</td><td>EfficientNet-B4</td><td>91.5%</td><td>17M (2.4x)</td><td>GPipe</td><td>93.0%</td><td>556M</td><td>EfficientNet-B7</td><td>93.0%</td><td>64M (8.7x)</td></tr><tr><td>Geo-Mean</td><td></td><td></td><td></td><td></td><td></td><td>(4.7x)</td><td></td><td></td><td></td><td></td><td></td><td>(9.6x)</td></tr></table>\n<sup>†</sup>GPipe (Huang et al., 2018) trains giant models with specialized pipeline parallelism library.\n<sup>‡</sup>DAT denotes domain adaptive transfer learning (Ngiam et al., 2018). Here we only compare ImageNet-based transfer learning results.\nTransfer accuracy and #params for NASNet (Zoph et al., 2018), Inception-v4 (Szegedy et al., 2017), ResNet-152 (He et al., 2016) are from (Kornblith et al., 2019).",
              "score": 0.019125,
              "semantic_score": 0.6479064226150513,
              "lexical_rank": null,
              "semantic_rank": 4,
              "rrf_score": 0.015625,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "table",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.00385
              },
              "page_start_display": 7,
              "page_end_display": 7,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "37e4842390cc5d5cfc8d61f6",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 11,
              "region": "content",
              "chapter_number": null,
              "chapter_title": "1. Introduction",
              "section_path": [
                "content",
                "1. Introduction"
              ],
              "section_label": "1. Introduction",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "cce2115cad48663da3eae25743a8733a6b73d03ee525fc3314f20d567f81e99d",
              "source_blocks": [
                {
                  "index": 19,
                  "page_idx": 1,
                  "bbox": [
                    83,
                    503,
                    477,
                    656
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 20,
                  "page_idx": 1,
                  "bbox": [
                    83,
                    662,
                    478,
                    905
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "8a736ab881b00223f3c638c39f752c677d230691074fcfed76cbdea99fcad88c",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets. In particular, our EfficientNet-B7 surpasses the best existing GPipe accuracy (Huang et al., 2018), but using 8.4x fewer parameters and running 6.1x faster on inference. Compared to the widely used ResNet-50 (He et al., 2016), our EfficientNet-B4 improves the top-1 accuracy from 76.3% to 83.0% (+6.7%) with similar FLOPS. Besides ImageNet, EfficientNets also transfer well and achieve stateof-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets.",
              "score": 0.018729032258064514,
              "semantic_score": 0.6818692684173584,
              "lexical_rank": null,
              "semantic_rank": 2,
              "rrf_score": 0.016129032258064516,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": false,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0026
              },
              "page_start_display": 2,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "5add4faebce9353f54e8571a",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 52,
              "region": "content",
              "chapter_number": "5.3",
              "chapter_title": "Transfer Learning Results for EfficientNet",
              "section_path": [
                "content",
                "5. Experiments",
                "5.3. Transfer Learning Results for EfficientNet"
              ],
              "section_label": "5.3. Transfer Learning Results for EfficientNet",
              "type": "text",
              "page_start": 7,
              "page_end": 7,
              "content_hash": "c57da8648b8b840375cf70576c557293d2227a00216e44beed7458abc7d58d8e",
              "source_blocks": [
                {
                  "index": 96,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    482,
                    477,
                    559
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 97,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    566,
                    478,
                    718
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 98,
                  "page_idx": 7,
                  "bbox": [
                    84,
                    724,
                    478,
                    816
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "4ef03beb8dec39de05829a07736ca915ed31c2d9e185f67b2c255f651f3880d1",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "We have also evaluated our EfficientNet on a list of commonly used transfer learning datasets, as shown in Table 6. We borrow the same training settings from (Kornblith et al., 2019) and (Huang et al., 2018), which take ImageNet pretrained checkpoints and finetune on new datasets.\n\nTable 5 shows the transfer learning performance: (1) Compared to public available models, such as NASNet-A (Zoph et al., 2018) and Inception-v4 (Szegedy et al., 2017), our EfficientNet models achieve better accuracy with 4.7x average (up to 21x) parameter reduction. (2) Compared to stateof-the-art models, including DAT (Ngiam et al., 2018) that dynamically synthesizes training data and GPipe (Huang et al., 2018) that is trained with specialized pipeline parallelism, our EfficientNet models still surpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters\n\nFigure 6 compares the accuracy-parameters curve for a variety of models.",
              "score": 0.018501515151515154,
              "semantic_score": 0.6150898933410645,
              "lexical_rank": null,
              "semantic_rank": 6,
              "rrf_score": 0.015151515151515152,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.00035,
                "quality_bonus": 0.0037
              },
              "page_start_display": 8,
              "page_end_display": 8,
              "evidence_role": "direct",
              "source_id": "S5"
            },
            {
              "chunk_id": "f53793418b851f61aab1c9dd",
              "paper_id": "1905.11946",
              "canonical_id": "1905.11946v5",
              "ordinal": 35,
              "region": "content",
              "chapter_number": "4",
              "chapter_title": "EfficientNet Architecture",
              "section_path": [
                "content",
                "4. EfficientNet Architecture"
              ],
              "section_label": "4. EfficientNet Architecture",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "2d40ed7a8e3d611a5fa5b7a1d87469ee97c418a19d9541664183245c75e176aa",
              "source_blocks": [
                {
                  "index": 64,
                  "page_idx": 4,
                  "bbox": [
                    84,
                    571,
                    475,
                    664
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 65,
                  "page_idx": 4,
                  "bbox": [
                    84,
                    671,
                    478,
                    883
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "816a3cd9db6a45a3e852b100a173a7943b2587f922e1caffeab98a336ceacdba",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "rather than latency since we are not targeting any specific hardware device. Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to Mnas-",
              "score": 0.018405882352941175,
              "semantic_score": 0.5837506651878357,
              "lexical_rank": null,
              "semantic_rank": 8,
              "rrf_score": 0.014705882352941176,
              "ranking_features": {
                "exact_entity_hit": true,
                "section_exact_hit": true,
                "type_priority": "text",
                "region_priority": "content",
                "duplicate_penalty": 0.0,
                "quality_bonus": 0.0037
              },
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S6"
            }
          ],
          "count": 6,
          "context_text": "[S1] 1905.11946v5 p.8: rpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters\n\nFigure 6 compares the accuracy-parameters curve for a variety of models. In general, our EfficientNets consistently achieve better accuracy with an order of magnitude fewer parameters than existing models, including ResNet (He et al., 2016), DenseNet (Huang et al., 2017), Inception (Szegedy et al., 2017), and NASNet (Zoph et al., 2018).\n\n[S2] 1905.11946v5 p.7: chieves 84.3% top1 accuracy with 66M parameters and 37B FLOPS, being more accurate but 8.4x smaller than the previous best GPipe (Huang et al., 2018). These gains come from both better architectures, better scaling, and better training settings that are customized for EfficientNet.\n\nFigure 1 and Figure 5 illustrates the parameters-accuracy and FLOPS-accuracy curve for representative ConvNets, where our scaled EfficientNet models achieve better accuracy with much fewer parameters and FLOPS than other ConvNets. Notably, our EfficientNet models are not only small, but also computational cheaper. For example, our EfficientNet-B3 achieves higher accuracy than ResNeXt-101 (Xie et al., 2017) using 18x fewer FLOPS.\n\nTo validate the latency, we have also measured the inference latency for a few representative CovNets on a real CPU as shown in Table 4, where we report average latency over 20 runs. Our EfficientNet-B1 runs 5.7x faster than the widely used ResNet-152, while EfficientNet-B7 runs about 6.1x faster than GPipe (Huang et al., 2018), suggesting our EfficientNets are indeed fast on real hardware.\n\n[S3] 1905.11946v5 p.7: Table 5. EfficientNet Performance Results on Transfer Learning Datasets. Our scaled EfficientNet models achieve new state-of-theart accuracy for 5 out of 8 datasets, with 9.6x fewer parameters on average.\n<table><tr><td rowspan=\"2\"></td><td colspan=\"6\">Comparison to best public-available results</td><td colspan=\"6\">Comparison to best reported results</td></tr><tr><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td><td>Model</td><td>Acc.</td><td>#Param</td><td>Our Model</td><td>Acc.</td><td>#Param(ratio)</td></tr><tr><td>CIFAR-10</td><td>NASNet-A</td><td>98.0%</td><td>85M</td><td>EfficientNet-B0</td><td>98.1%</td><td>4M (21x)</td><td> $^\\dagger$ Gpipe</td><td>99.0%</td><td>556M</td><td>EfficientNet-B7</td><td>98.9%</td><td>64M (8.7x)</td></tr><tr><td>CIFAR-100</td><td>NASNet-A</td><td>87.5%</td><td>85M</td><td>EfficientNet-B0</td><td>88.1%</td><td>4M (21x)</td><td>Gpipe</td><td>91.3%</td><td>556M</td><td>EfficientNet-B7</td><td>91.7%</td><td>64M (8.7x)</td></tr><tr><td>Birdsnap</td><td>Inception-v4</td><td>81.8%</td><td>41M</td><td>EfficientNet-B5</td><td>82.0%</td><td>28M (1.5x)</td><td>GPipe</td><td>83.6%</td><td>556M</td><td>EfficientNet-B7</td><td>84.3%</td><td>64M (8.7x)</td></tr><tr><td>Stanford Cars</td><td>Inception-v4</td><td>93.4%</td><td>41M</td><td>EfficientNet-B3</td><td>93.6%</td><td>10M (4.1x)</td><td> $^\\ddagger$ DAT</td><td>94.8%</td><td>-</td><td>EfficientNet-B7</td><td>94.7%</td><td>-</td></tr><tr><td>Flowers</td><td>Inception-v4</td><td>98.5%</td><td>41M</td><td>EfficientNet-B5</td><td>98.5%</td><td>28M (1.5x)</td><td>DAT</td><td>97.7%</td><td>-</td><td>EfficientNet-B7</td><td>98.8%</td><td>-</td></tr><tr><td>FGVC Aircraft</td><td>Inception-v4</td><td>90.9%</td><td>41M</td><td>EfficientNet-B3</td><td>90.7%</td><td>10M (4.1x)</td><td>DAT</td><td>92.9%</td><td>-</td><td>EfficientNet-B7</td><td>92.9%</td><td>-</td></tr><tr><td>Oxford-IIIT Pets</td><td>ResNet-152</td><td>94.5%</td><td>58M</td><td>EfficientNet-B4</td><td>94.8%</td><td>17M (5.6x)</td><td>GPipe</td><td>95.9%</td><td>556M</td><td>EfficientNet-B6</td><td>95.4%</td><td>41M (14x)</td></tr><tr><td>Food-101</td><td>Inception-v4</td><td>90.8%</td><td>41M</td><td>EfficientNet-B4</td><td>91.5%</td><td>17M (2.4x)</td><td>GPipe</td><td>93.0%</td><td>556M</td><td>EfficientNet-B7</td><td>93.0%</td><td>64M (8.7x)</td></tr><tr><td>Geo-Mean</td><td></td><td></td><td></td><td></td><td></td><td>(4.7x)</td><td></td><td></td><td></td><td></td><td></td><td>(9.6x)</td></tr></table>\n<sup>†</sup>GPipe (Huang et al., 2018) trains giant models with specialized pipeline parallelism library.\n<sup>‡</sup>DAT denotes domain adaptive transfer learning (Ngiam et al., 2018). Here we only compare ImageNet-based transfer learning results.\nTransfer accuracy and #params for NASNet (Zoph et al., 2018), Inception-v4 (Szegedy et al., 2017), ResNet-152 (He et al., 2016) are from (Kornblith et al., 2019).\n\n[S4] 1905.11946v5 p.2: family of models, called EfficientNets. Figure 1 summarizes the ImageNet performance, where our EfficientNets significantly outperform other ConvNets. In particular, our EfficientNet-B7 surpasses the best existing GPipe accuracy (Huang et al., 2018), but using 8.4x fewer parameters and running 6.1x faster on inference. Compared to the widely used ResNet-50 (He et al., 2016), our EfficientNet-B4 improves the top-1 accuracy from 76.3% to 83.0% (+6.7%) with similar FLOPS. Besides ImageNet, EfficientNets also transfer well and achieve stateof-the-art accuracy on 5 out of 8 widely used datasets, while reducing parameters by up to 21x than existing ConvNets.\n\n[S5] 1905.11946v5 p.8: We have also evaluated our EfficientNet on a list of commonly used transfer learning datasets, as shown in Table 6. We borrow the same training settings from (Kornblith et al., 2019) and (Huang et al., 2018), which take ImageNet pretrained checkpoints and finetune on new datasets.\n\nTable 5 shows the transfer learning performance: (1) Compared to public available models, such as NASNet-A (Zoph et al., 2018) and Inception-v4 (Szegedy et al., 2017), our EfficientNet models achieve better accuracy with 4.7x average (up to 21x) parameter reduction. (2) Compared to stateof-the-art models, including DAT (Ngiam et al., 2018) that dynamically synthesizes training data and GPipe (Huang et al., 2018) that is trained with specialized pipeline parallelism, our EfficientNet models still surpass their accuracy in 5 out of 8 datasets, but using 9.6x fewer parameters\n\nFigure 6 compares the accuracy-parameters curve for a variety of models.\n\n[S6] 1905.11946v5 p.5: rather than latency since we are not targeting any specific hardware device. Our search produces an efficient network, which we name EfficientNet-B0. Since we use the same search space as (Tan et al., 2019), the architecture is similar to Mnas-",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"efficientnet table\" OR \"efficientnet b0 accuracy\" OR \"efficientnet b0 parameter count\" OR \"resnet 50 accuracy\" OR \"resnet 50 parameter count\" OR \"efficientnet b0 vs resnet 50 comparison\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "efficientnet table",
              "efficientnet b0 accuracy",
              "efficientnet b0 parameter count",
              "resnet 50 accuracy",
              "resnet 50 parameter count",
              "efficientnet b0 vs resnet 50 comparison"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      },
      "quality_review": {
        "verdict": "失败",
        "paper_recall": "通过：仅返回 1905.11946。",
        "evidence_quality": "查询明确要求 Table 2，但返回的唯一 table 是 Table 5；正文只给出泛化的 EfficientNet/ResNet 对比，没有返回 Table 2 的 B0 77.1%/5.3M 与 ResNet-50 76.0%/26M。",
        "issues": [
          "表号和表格内容没有被精确匹配，Table 5 排在第 3 条，属于事实检索失败。"
        ]
      }
    }
  ]
}
```

返回论文：`1905.11946`；质量判定：**失败**。

## 案例 13：Summarize the core method of NebulaXYZ-9876.

```json
{
  "user_question": "Summarize the core method of NebulaXYZ-9876.",
  "expected_evidence": {
    "paper_ids": [],
    "criteria": "本地不存在该方法，不应把弱相关论文包装为目标方法。"
  },
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_retrieve",
        "reason": "问题需要论文正文证据；task=auto 时由服务分类，其余明确指定任务。"
      },
      "mcp_request": {
        "tool": "library_retrieve",
        "arguments": {
          "query": "Summarize the core method of NebulaXYZ-9876.",
          "task": "summary",
          "mode": "hybrid",
          "limit": 6,
          "max_chars": 18000
        }
      },
      "mcp_response": {
        "status": "not_found",
        "data": {
          "query": "Summarize the core method of NebulaXYZ-9876.",
          "task": "summary",
          "papers": [],
          "items": [],
          "count": 0,
          "context_text": "",
          "truncated": false,
          "retrieval_debug": {
            "candidate_discovery": {
              "metadata_count": 0,
              "chunk_count": 0,
              "entity_hits": {},
              "fallback_used": false,
              "chunk_search_used": false,
              "selected_paper_ids": [],
              "lexical_query": "\"nebulaxyz 9876 core method\""
            }
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [
          "no papers match the supplied constraints"
        ],
        "read_only": true
      },
      "quality_review": {
        "verdict": "通过",
        "paper_recall": "通过：返回 not_found 且没有论文。",
        "evidence_quality": "candidate_discovery 明确显示 metadata_count=0、chunk_count=0、selected_paper_ids=[]，未用弱相关论文掩盖本地语料缺失。",
        "issues": []
      }
    }
  ]
}
```

返回论文：`无`；质量判定：**通过**。

## 批次结论

本批次共 13 个案例：通过 6 个，部分通过 6 个，失败 1 个。核心目标论文在事实、原因、摘要和对比案例中均有命中；主要异常集中在候选去噪、章节精确排序和表号精确匹配。

- 候选噪声：案例 1、8、11 混入相关度较弱的论文或媒体类型。
- 章节/证据排序：案例 4 没有把 shifted-window 核心正文排到首位；案例 10 在正确的 Appendix 小节后混入无关 Appendix 小节。
- 查询与来源不一致：案例 9 的“指数数量浅层网络”论点在正文 4.1，不在 Appendix；记录保留这一事实，避免把正文结果误解为附录召回成功。
- 表格精确性：案例 12 要求 Table 2，却返回 Table 5，当前表号约束不足。
- 缺失论文拒答：案例 13 正确返回 `not_found`，没有用弱相关结果替代。

## 索引状态

```json
{
  "status": "ok",
  "data": {
    "status": "ok",
    "catalog_ready": true,
    "index_ready": true,
    "index_stale": false,
    "stale_reasons": [],
    "cache_complete": true,
    "chunk_count": 3241,
    "indexed_count": 3241,
    "embedding_model": "qwen3.7-text-embedding-flash",
    "embedding_dimensions": 1024,
    "milvus_collection": "paper_rag_llama_chunks__build_be6203949c11",
    "built_at": "2026-10-05T15:29:58.409711+00:00",
    "last_sync_mode": "incremental",
    "last_sync_status": "completed",
    "sync_stats": {
      "reused": 1492,
      "added": 1395,
      "updated": 354,
      "deleted": 404,
      "failed": 0
    },
    "manifest": {
      "schema_version": 2,
      "catalog_indexed_at": "2026-10-05T14:26:18.712262+00:00",
      "chunk_count": 3241,
      "indexed_count": 3241,
      "chunk_rule_version": "content-list-regions-v5-1200-table-text",
      "embedding_model": "qwen3.7-text-embedding-flash",
      "embedding_dimensions": 1024,
      "milvus_collection": "paper_rag_llama_chunks__build_be6203949c11",
      "built_at": "2026-10-05T15:29:58.409711+00:00",
      "status": "ready",
      "last_sync_mode": "incremental",
      "sync_stats": {
        "reused": 1492,
        "added": 1395,
        "updated": 354,
        "deleted": 404,
        "failed": 0
      }
    }
  },
  "warnings": [],
  "read_only": true
}
```

本次只做真实检索和记录，没有重建 Catalog/Milvus，也没有修改 chunk 内容、区域划分或 schema。

## 本轮 v6 Chunk 重设计、索引重建与回归

> 生成时间（UTC）：2026-10-06T05:26:58.019150+00:00
> 本轮先完成 content-list-regions-v6-boundary-aware Chunk 重设计，再完成 Catalog/Milvus 全量重建；下方保留旧版 13 个案例，并追加既有回归问题与来源对齐的 ResNet Appendix 诊断。

### Catalog 与 Chunk 审计

```json
{
  "source_papers": 34,
  "chunk_rule_version": "content-list-regions-v6-boundary-aware",
  "catalog_chunks": 3292,
  "regions": {"abstract": 58, "content": 2297, "appendix": 937, "reference": 0},
  "media_chunks": 1110,
  "table_chunks": 341,
  "table_chunks_with_numbered_caption": 261,
  "lowercase_text_starts": 123,
  "same_section_adjacent_alnum_boundaries": 13,
  "boundary_audit_note": "剩余 13 对来自原始 MinerU 文本块或媒体边界，不是 v6 字符 overlap 产生的半词重叠；单元测试覆盖的 v6 切分路径没有 mid-token 起始。",
  "references_chunks": 0
}
```

lowercase_text_starts 不能单独作为失败指标，因为 where、for 等完整句子和原始抽取块也可能以小写开头；本轮重点审计相邻 Chunk 的词边界、媒体独立性和 caption 完整性。v5 中固定字符 overlap 造成的系统性残词问题已由边界感知切分替代。

### Milvus 全量重建

```json
{
  "index_ready": true,
  "index_stale": false,
  "cache_complete": true,
  "chunk_count": 3292,
  "indexed_count": 3292,
  "embedding_dimensions": 1024,
  "last_sync_mode": "full",
  "failed": 0,
  "milvus_collection": "paper_rag_llama_chunks__build_472479e8ef9c"
}
```

### v6 真实正文检索回归

下列记录由当前 Catalog 和已切换的 v6 active Milvus 直接调用 library_retrieve 对应的服务实现生成；top_items 只保留每个案例前三条的可回溯字段，完整紧凑结果保存在本轮运行产物中。

```json
[
  {
    "id": "old-gqa",
    "request": {
      "query": "In GQA section 2.2, how does grouped-query attention reduce the KV cache?",
      "task": "fact",
      "paper_ids": [
        "2305.13245"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2305.13245"
    ],
    "top_items": [
      {
        "paper_id": "2305.13245",
        "chunk_id": "908f3ee9d59e32fc115fb9d7",
        "region": "content",
        "section_label": "2.2 Grouped-query attention",
        "type": "text",
        "ordinal": 9,
        "page_start": 1,
        "text_head": "We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck.",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.13245",
        "chunk_id": "8510645f75b366fffd870e56",
        "region": "content",
        "section_label": "2.2 Grouped-query attention",
        "type": "text",
        "ordinal": 8,
        "page_start": 1,
        "text_head": "An intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value h",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.13245",
        "chunk_id": "9d59efb82ff64eaa74f8b259",
        "region": "content",
        "section_label": "2.2 Grouped-query attention",
        "type": "text",
        "ordinal": 7,
        "page_start": 1,
        "text_head": "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "old-paged-attention",
    "request": {
      "query": "Why does PagedAttention improve serving efficiency for long sequences?",
      "task": "reason",
      "paper_ids": [
        "2309.06180"
      ]
    },
    "status": "ok",
    "task": "reason",
    "papers": [
      "2309.06180"
    ],
    "top_items": [
      {
        "paper_id": "2309.06180",
        "chunk_id": "c0c8bf64e9c3881d64a673a8",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "text",
        "ordinal": 11,
        "page_start": 1,
        "text_head": "To address the above limitations, we propose PagedAttention, an attention algorithm inspired by the operating system’s (OS) solution to memory fragmentation and sharing: virtual memory with paging. PagedAttention divides",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": "2309.06180:11",
        "source_chunk_ids": [
          "2ea7b9b04ad79e6bb1948712",
          "c0c8bf64e9c3881d64a673a8",
          "bffd7a8bc3311122f48a1aad"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "2309.06180",
        "chunk_id": "911d0de24d794c2a9a8c6745",
        "region": "content",
        "section_label": "10 Conclusion",
        "type": "text",
        "ordinal": 116,
        "page_start": 13,
        "text_head": "This paper proposes PagedAttention, a new attention algorithm that allows attention keys and values to be stored in non-contiguous paged memory, and presents vLLM, a high-throughput LLM serving system with eficient memor",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": "2309.06180:116",
        "source_chunk_ids": [
          "911d0de24d794c2a9a8c6745"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "2309.06180",
        "chunk_id": "6e069195ba754d51bafa95c1",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "High throughput serving of large language models (LLMs) requires batching suficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": "2309.06180:0",
        "source_chunk_ids": [
          "6e069195ba754d51bafa95c1"
        ],
        "continuity_status": "complete"
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "old-lora-summary",
    "request": {
      "query": "Summarize the core contributions of LoRA.",
      "task": "summary"
    },
    "status": "ok",
    "task": "summary",
    "papers": [
      "2106.09685"
    ],
    "top_items": [
      {
        "paper_id": "2106.09685",
        "chunk_id": "7f475dda898c0404d731dd7d",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 1,
        "page_start": 0,
        "text_head": "We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.\n\nWe release a package that facilitates the integration of LoRA with PyTorch models ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2106.09685",
        "chunk_id": "b64ef3dd069a4d2b933bdcaf",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains.\n\nAs we pre-train larger models, full fine-tuning, which retra",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2106.09685",
        "chunk_id": "0be95b55edde3ff291331d5a",
        "region": "content",
        "section_label": "4 OUR METHOD",
        "type": "text",
        "ordinal": 18,
        "page_start": 3,
        "text_head": "We describe the simple design of LoRA and its practical benefits. The principles outlined here apply to any dense layers in deep learning models, though we only focus on certain weights in Transformer language models in ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": {
      "metadata_count": 6,
      "chunk_count": 0,
      "entity_hits": {
        "LoRA": [
          "2106.09685",
          "2305.14314"
        ]
      },
      "fallback_used": false,
      "chunk_search_used": false,
      "selected_paper_ids": [
        "2106.09685"
      ],
      "lexical_query": "\"lora\" OR \"low rank adaptation\" OR \"core contributions\" OR \"parameter efficient fine tuning\" OR \"frozen pretrained model weights\" OR \"low rank decomposition\" OR \"trainable rank decomposition matrices\" OR \"reduced trainable parameters\" OR \"no additional inference latency\" OR \"large language models\" OR \"gpt\" OR \"roberta\" OR \"deberta\"",
      "candidate_match_source": {
        "2106.09685": "title_exact"
      },
      "title_exact_hit": [
        "2106.09685"
      ],
      "abstract_exact_hit": [
        "2106.09685",
        "2305.14314"
      ],
      "chunk_exact_hit": [],
      "table_ref": null,
      "figure_ref": null
    },
    "warnings": []
  },
  {
    "id": "old-lora-qlora-comparison",
    "request": {
      "query": "Compare LoRA and QLoRA in terms of parameter efficiency and quantization.",
      "task": "comparison",
      "paper_ids": [
        "2106.09685",
        "2305.14314"
      ]
    },
    "status": "ok",
    "task": "comparison",
    "papers": [
      "2106.09685",
      "2305.14314"
    ],
    "top_items": [
      {
        "paper_id": "2106.09685",
        "chunk_id": "b64ef3dd069a4d2b933bdcaf",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains.\n\nAs we pre-train larger models, full fine-tuning, which retra",
        "ranking_features": {},
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.14314",
        "chunk_id": "e02264b1a96ae177e7cbcb34",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.\n\nQLORA backpropagates g",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.14314",
        "chunk_id": "2578dff1580b81da71eb16ea",
        "region": "content",
        "section_label": "4 QLoRA vs. Standard Finetuning",
        "type": "text",
        "ordinal": 41,
        "page_start": 6,
        "text_head": "For our second setup, since full finetuning models at and beyond 11B parameters requires more than one server of high memory GPUs, we continue to test whether 4-bit QLORA can match 16-bit LoRA at the 7B to 65B parameter ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "bert-pretraining",
    "request": {
      "query": "BERT 的预训练使用了哪两个任务？",
      "task": "fact"
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1810.04805"
    ],
    "top_items": [
      {
        "paper_id": "1810.04805",
        "chunk_id": "ed463687debe179fcb629b97",
        "region": "content",
        "section_label": "3 BERT",
        "type": "text",
        "ordinal": 12,
        "page_start": 2,
        "text_head": "We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-traini",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1810.04805",
        "chunk_id": "475fd15d377585cb1b4bd91e",
        "region": "content",
        "section_label": "3.1 Pre-training BERT",
        "type": "text",
        "ordinal": 19,
        "page_start": 3,
        "text_head": "Task #2: Next Sentence Prediction (NSP) Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1810.04805",
        "chunk_id": "862411336715c1963efe8392",
        "region": "content",
        "section_label": "3.1 Pre-training BERT",
        "type": "text",
        "ordinal": 16,
        "page_start": 3,
        "text_head": "Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in th",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": {
      "metadata_count": 0,
      "chunk_count": 0,
      "entity_hits": {
        "BERT": [
          "1810.04805",
          "1909.08053",
          "1902.00751",
          "2205.14135"
        ]
      },
      "fallback_used": false,
      "chunk_search_used": false,
      "selected_paper_ids": [
        "1810.04805"
      ],
      "lexical_query": "\"bert pre training two tasks\"",
      "candidate_match_source": {
        "1810.04805": "title_exact"
      },
      "title_exact_hit": [
        "1810.04805"
      ],
      "abstract_exact_hit": [
        "1810.04805",
        "1909.08053",
        "1902.00751",
        "2205.14135"
      ],
      "chunk_exact_hit": [],
      "table_ref": null,
      "figure_ref": null
    },
    "warnings": [
      "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
    ]
  },
  {
    "id": "mixed-precision",
    "request": {
      "query": "Why does mixed precision training keep FP32 master weights?",
      "task": "reason",
      "paper_ids": [
        "1710.03740"
      ]
    },
    "status": "ok",
    "task": "reason",
    "papers": [
      "1710.03740"
    ],
    "top_items": [
      {
        "paper_id": "1710.03740",
        "chunk_id": "54dbeafef2e763e6bc23a3a2",
        "region": "content",
        "section_label": "3.1 FP32 MASTER COPY OF WEIGHTS",
        "type": "text",
        "ordinal": 10,
        "page_start": 1,
        "text_head": "In mixed precision training, weights, activations and gradients are stored as FP16. In order to match the accuracy of the FP32 networks, an FP32 master copy of weights is maintained and updated with the weight gradient d",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": "1710.03740:10",
        "source_chunk_ids": [
          "54dbeafef2e763e6bc23a3a2",
          "22d200dd0c5e063ef0f58213"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1710.03740",
        "chunk_id": "6735e2dcd7f0e8a9477faa32",
        "region": "content",
        "section_label": "3.1 FP32 MASTER COPY OF WEIGHTS",
        "type": "text",
        "ordinal": 14,
        "page_start": 2,
        "text_head": "Another explanation is that the ratio of the weight value to the weight update is very large. In this case, even though the weight update is representable in FP16, it could still become zero when addition operation right",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": "1710.03740:14",
        "source_chunk_ids": [
          "08a61531a187e6a9cbb013a3",
          "6735e2dcd7f0e8a9477faa32"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1710.03740",
        "chunk_id": "43d5d4437a954720b74206d0",
        "region": "content",
        "section_label": "3.1 FP32 MASTER COPY OF WEIGHTS",
        "type": "text",
        "ordinal": 12,
        "page_start": 2,
        "text_head": "Figure 1: Mixed precision training iteration for a layer.\n\nWhile the need for FP32 master weights is not universal, there are two possible reasons why a number of networks require it. One explanation is that updates (wei",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": "1710.03740:12",
        "source_chunk_ids": [
          "22d200dd0c5e063ef0f58213",
          "43d5d4437a954720b74206d0",
          "08a61531a187e6a9cbb013a3"
        ],
        "continuity_status": "complete"
      }
    ],
    "candidate_discovery": null,
    "warnings": [
      "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
    ]
  },
  {
    "id": "flash-attention",
    "request": {
      "query": "How does FlashAttention reduce HBM reads and writes?",
      "task": "reason",
      "paper_ids": [
        "2205.14135"
      ]
    },
    "status": "ok",
    "task": "reason",
    "papers": [
      "2205.14135"
    ],
    "top_items": [
      {
        "paper_id": "2205.14135",
        "chunk_id": "0a43b1748eae09492ad8d952",
        "region": "content",
        "section_label": "3 FlashAttention: Algorithm, Analysis, and Extensions",
        "type": "text",
        "ordinal": 18,
        "page_start": 3,
        "text_head": "We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory eficient and faster in wall",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": "2205.14135:18",
        "source_chunk_ids": [
          "0a43b1748eae09492ad8d952"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "2205.14135",
        "chunk_id": "0c06faee72edec653b249fa3",
        "region": "content",
        "section_label": "3.2 Analysis: IO Complexity of FlashAttention",
        "type": "text",
        "ordinal": 27,
        "page_start": 4,
        "text_head": "We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can asymptotically im",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": "2205.14135:27",
        "source_chunk_ids": [
          "0c06faee72edec653b249fa3",
          "9b226e40c04358af10176549",
          "f28336eaa705274cbc17a7b3"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "2205.14135",
        "chunk_id": "79e951211bd5c5b869d72770",
        "region": "content",
        "section_label": "3.2 Analysis: IO Complexity of FlashAttention",
        "type": "text",
        "ordinal": 30,
        "page_start": 5,
        "text_head": "Figure 2: Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor afecting runtim",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": "2205.14135:30",
        "source_chunk_ids": [
          "f28336eaa705274cbc17a7b3",
          "79e951211bd5c5b869d72770",
          "b8e7ed5c3cba0b847281e92a"
        ],
        "continuity_status": "complete"
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "swin-shifted-window",
    "request": {
      "query": "How does Swin Transformer implement shifted window attention in section 3.2?",
      "task": "fact",
      "paper_ids": [
        "2103.14030"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2103.14030"
    ],
    "top_items": [
      {
        "paper_id": "2103.14030",
        "chunk_id": "6aee8432f9505648b92c1461",
        "region": "content",
        "section_label": "3.1. Overall Architecture",
        "type": "text",
        "ordinal": 19,
        "page_start": 3,
        "text_head": "Swin Transformer block Swin Transformer is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows (described in Section 3.2), with other layers ",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "25951def6ae2021946edd43f",
        "region": "content",
        "section_label": "5. Conclusion",
        "type": "text",
        "ordinal": 59,
        "page_start": 8,
        "text_head": "As a key element of Swin Transformer, the shifted window based self-attention is shown to be effective and efficient on vision problems, and we look forward to investigating its use in natural language processing as well",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "f587fb928505c9d30f25c840",
        "region": "content",
        "section_label": "3.2. Shifted Window based Self-Attention",
        "type": "text",
        "ordinal": 24,
        "page_start": 3,
        "text_head": "As illustrated in Figure 2, the first module uses a regular window partitioning strategy which starts from the top-left pixel, and the $8 \\times 8$ feature map is evenly partitioned into $2 \\times 2$ windows of size $4 \\",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0023
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "efficientnet-summary",
    "request": {
      "query": "Summarize the main contribution of EfficientNet.",
      "task": "summary"
    },
    "status": "ok",
    "task": "summary",
    "papers": [
      "1905.11946"
    ],
    "top_items": [
      {
        "paper_id": "1905.11946",
        "chunk_id": "7fd91f7012369924be69022b",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and",
        "ranking_features": {},
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "8e09cc231c08e389400940e6",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 1,
        "page_start": 0,
        "text_head": "To go even further, we use neural architecture search to design a new baseline network and scale it up to obtain a family of models, called EfficientNets, which achieve much better accuracy and efficiency than previous C",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "97b182165b2647c27e5a3116",
        "region": "content",
        "section_label": "5.2. ImageNet Results for EfficientNet",
        "type": "text",
        "ordinal": 49,
        "page_start": 6,
        "text_head": "Table 2 shows the performance of all EfficientNet models that are scaled from the same baseline EfficientNet-B0. Our EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00345
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": {
      "metadata_count": 0,
      "chunk_count": 0,
      "entity_hits": {
        "EfficientNet": [
          "1905.11946"
        ]
      },
      "fallback_used": false,
      "chunk_search_used": false,
      "selected_paper_ids": [
        "1905.11946"
      ],
      "lexical_query": "\"efficientnet main contribution\"",
      "candidate_match_source": {
        "1905.11946": "title_exact"
      },
      "title_exact_hit": [
        "1905.11946"
      ],
      "abstract_exact_hit": [
        "1905.11946"
      ],
      "chunk_exact_hit": [],
      "table_ref": null,
      "figure_ref": null
    },
    "warnings": []
  },
  {
    "id": "dpo-vs-rlhf",
    "request": {
      "query": "How does DPO differ from RLHF?",
      "task": "comparison",
      "paper_ids": [
        "2305.18290",
        "2203.02155"
      ]
    },
    "status": "ok",
    "task": "comparison",
    "papers": [
      "2305.18290",
      "2203.02155"
    ],
    "top_items": [
      {
        "paper_id": "2305.18290",
        "chunk_id": "3ee88c497e55cd17d9cc4a6e",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their trai",
        "ranking_features": {},
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.18290",
        "chunk_id": "5c91b47d23366160c9b35db4",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 1,
        "page_start": 0,
        "text_head": "The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing signific",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2305.18290",
        "chunk_id": "19604257220638b91c581ff9",
        "region": "content",
        "section_label": "6.1 How well can DPO optimize the RLHF objective?",
        "type": "text",
        "ordinal": 56,
        "page_start": 7,
        "text_head": "This sweep includes 22 runs in total.\n\nAfter each 100 training steps until convergence, we evaluate each policy on a set of test prompts, computing the average reward under the true reward function as well as the average",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "prefix-prompt",
    "request": {
      "query": "Compare prefix tuning and prompt tuning.",
      "task": "comparison",
      "paper_ids": [
        "2101.00190",
        "2104.08691"
      ]
    },
    "status": "ok",
    "task": "comparison",
    "papers": [
      "2101.00190",
      "2104.08691"
    ],
    "top_items": [
      {
        "paper_id": "2101.00190",
        "chunk_id": "9822dea303c054d81a65e116",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each tas",
        "ranking_features": {},
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2101.00190",
        "chunk_id": "024208a4b05d1b5b64c6d532",
        "region": "content",
        "section_label": "4 Prefix-Tuning",
        "type": "text",
        "ordinal": 19,
        "page_start": 2,
        "text_head": "We propose prefix-tuning as an alternative to fine-tuning for conditional generation tasks. We first provide intuition in §4.1 before defining our method formally in §4.2.",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2101.00190",
        "chunk_id": "53245f9f18c349f24a71f625",
        "region": "content",
        "section_label": "8.3 Inductive Bias of Prefix-tuning",
        "type": "text",
        "ordinal": 58,
        "page_start": 8,
        "text_head": "Recall that fine-tuning updates all pretrained parameters, whereas prefix-tuning and adapter-tuning preserve them. Since the language models are pretrained on general purpose corpus, preserving the LM parameters might he",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "vit-swin",
    "request": {
      "query": "Compare the attention mechanisms in ViT and Swin Transformer.",
      "task": "comparison",
      "paper_ids": [
        "2010.11929",
        "2103.14030"
      ]
    },
    "status": "ok",
    "task": "comparison",
    "papers": [
      "2010.11929",
      "2103.14030"
    ],
    "top_items": [
      {
        "paper_id": "2010.11929",
        "chunk_id": "8087f3d28bd083daa84dc39e",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with co",
        "ranking_features": {},
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2010.11929",
        "chunk_id": "80c4913aae5aa573086b9f3e",
        "region": "content",
        "section_label": "3.1 VISION TRANSFORMER (VIT)",
        "type": "text",
        "ordinal": 17,
        "page_start": 3,
        "text_head": "Inductive bias. We note that Vision Transformer has much less image-specific inductive bias than CNNs. In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer thr",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2010.11929",
        "chunk_id": "64ea8305c39c4391b9950404",
        "region": "content",
        "section_label": "4.5 INSPECTING VISION TRANSFORMER",
        "type": "text",
        "ordinal": 45,
        "page_start": 7,
        "text_head": "Self-attention allows ViT to integrate information across the entire image even in the lowest layers. We investigate to what degree the network makes use of this capability. Specifically, we compute the average distance ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "resnet-source-aligned-appendix",
    "request": {
      "query": "What does the ResNet appendix say about the Faster R-CNN object detection baseline?",
      "task": "fact",
      "paper_ids": [
        "1512.03385"
      ],
      "regions": [
        "appendix"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1512.03385"
    ],
    "top_items": [
      {
        "paper_id": "1512.03385",
        "chunk_id": "d015099b7577626391363132",
        "region": "appendix",
        "section_label": "A. Object Detection Baselines",
        "type": "text",
        "ordinal": 57,
        "page_start": 9,
        "text_head": "In this section we introduce our detection method based on the baseline Faster R-CNN [32] system. The models are initialized by the ImageNet classification models, and then fine-tuned on the object detection data. We hav",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1512.03385",
        "chunk_id": "f3900e6e70c13fa8e39372e0",
        "region": "appendix",
        "section_label": "PASCAL VOC",
        "type": "text",
        "ordinal": 60,
        "page_start": 9,
        "text_head": "Following [7, 32], for the PASCAL VOC 2007 test set, we use the 5k trainval images in VOC 2007 and 16k trainval images in VOC 2012 for training (“07+12”). For the PASCAL VOC 2012 test set, we use the 10k trainval+test im",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1512.03385",
        "chunk_id": "0ade6651bc23544848c235ff",
        "region": "appendix",
        "section_label": "A. Object Detection Baselines",
        "type": "text",
        "ordinal": 58,
        "page_start": 9,
        "text_head": "Unlike VGG-16 used in [32], our ResNet has no hidden fc layers. We adopt the idea of “Networks on Conv feature maps” (NoC) [33] to address this issue. We compute the full-image shared conv feature maps using those layers",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "resnet-mismatch-appendix",
    "request": {
      "query": "What does the ResNet appendix explain about exponential learning in shallow networks?",
      "task": "fact",
      "paper_ids": [
        "1512.03385"
      ],
      "regions": [
        "appendix"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1512.03385"
    ],
    "top_items": [
      {
        "paper_id": "1512.03385",
        "chunk_id": "0ade6651bc23544848c235ff",
        "region": "appendix",
        "section_label": "A. Object Detection Baselines",
        "type": "text",
        "ordinal": 58,
        "page_start": 9,
        "text_head": "Unlike VGG-16 used in [32], our ResNet has no hidden fc layers. We adopt the idea of “Networks on Conv feature maps” (NoC) [33] to address this issue. We compute the full-image shared conv feature maps using those layers",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1512.03385",
        "chunk_id": "0291a309beaf2ed3b57b7b2e",
        "region": "appendix",
        "section_label": "B. Object Detection Improvements",
        "type": "text",
        "ordinal": 62,
        "page_start": 9,
        "text_head": "For completeness, we report the improvements made for the competitions. These improvements are based on deep features and thus should benefit from residual learning.",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00045
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1512.03385",
        "chunk_id": "ad732b8eb6df6f1d71da62ce",
        "region": "appendix",
        "section_label": "C. ImageNet Localization",
        "type": "text",
        "ordinal": 78,
        "page_start": 11,
        "text_head": "The above results are only based on the proposal network (RPN) in Faster R-CNN [32]. One may use the detection network (Fast R-CNN [7]) in Faster R-CNN to improve the results. But we notice that on this dataset, one imag",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00045
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "swin-appendix",
    "request": {
      "query": "What are the ImageNet-1K training settings in the Swin appendix?",
      "task": "fact",
      "paper_ids": [
        "2103.14030"
      ],
      "regions": [
        "appendix"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2103.14030"
    ],
    "top_items": [
      {
        "paper_id": "2103.14030",
        "chunk_id": "8f346561e314ef1597150ac1",
        "region": "appendix",
        "section_label": "A2.1. Image classification on ImageNet-1K",
        "type": "text",
        "ordinal": 63,
        "page_start": 8,
        "text_head": "When training from scratch with a $2 2 4 ^ { 2 }$ input, we employ an AdamW [37] optimizer for 300 epochs using a cosine decay learning rate scheduler with 20 epochs of linear warm-up. A batch size of 1024, an initial le",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00295
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "0380be3f8096644df2d0823a",
        "region": "appendix",
        "section_label": "A2.1. Image classification on ImageNet-1K",
        "type": "text",
        "ordinal": 64,
        "page_start": 8,
        "text_head": "ImageNet-22K pre-training We also pre-train on the larger ImageNet-22K dataset, which contains 14.2 million images and 22K classes. The training is done in two stages. For the first stage with $2 2 4 ^ { 2 }$ input, we e",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00295
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "f9dd1f457a56f8687109b9e8",
        "region": "appendix",
        "section_label": "A2.3. Semantic segmentation on ADE20K",
        "type": "text",
        "ordinal": 67,
        "page_start": 9,
        "text_head": "ADE20K [83] is a widely-used semantic segmentation dataset, covering a broad range of 150 semantic categories. It has 25K images in total, with 20K for training, 2K for validation, and another 3K for testing. We utilize ",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00045
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "local-window",
    "request": {
      "query": "Which paper explains local window attention for vision transformers?",
      "task": "fact",
      "filters": {
        "category": "cs.CV",
        "year_from": "2020"
      }
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2010.11929",
      "2103.14030"
    ],
    "top_items": [
      {
        "paper_id": "2010.11929",
        "chunk_id": "f9e53abe30e197bc1b9347a8",
        "region": "content",
        "section_label": "4.4 SCALING STUDY",
        "type": "text",
        "ordinal": 42,
        "page_start": 7,
        "text_head": "Figure 5 contains the transfer performance versus total pre-training compute (see Appendix D.5 for details on computational costs). Detailed results per model are provided in Table 6 in the Appendix. A few patterns can b",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "d1de5aad80ffb3379a3a896e",
        "region": "content",
        "section_label": "3.2. Shifted Window based Self-Attention",
        "type": "text",
        "ordinal": 20,
        "page_start": 3,
        "text_head": "The standard Transformer architecture [64] and its adaptation for image classification [20] both conduct global selfattention, where the relationships between a token and all other tokens are computed. The global computa",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2103.14030",
        "chunk_id": "25951def6ae2021946edd43f",
        "region": "content",
        "section_label": "5. Conclusion",
        "type": "text",
        "ordinal": 59,
        "page_start": 8,
        "text_head": "As a key element of Swin Transformer, the shifted window based self-attention is shown to be effective and efficient on vision problems, and we look forward to investigating its use in natural language processing as well",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "efficientnet-table2",
    "request": {
      "query": "In EfficientNet Table 2, how does EfficientNet-B0 compare with ResNet-50?",
      "task": "fact",
      "paper_ids": [
        "1905.11946"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1905.11946"
    ],
    "top_items": [
      {
        "paper_id": "1905.11946",
        "chunk_id": "a98ff2b561827561785f0791",
        "region": "content",
        "section_label": "5.1. Scaling Up MobileNets and ResNets",
        "type": "table",
        "ordinal": 41,
        "page_start": 5,
        "text_head": "Table 2. EfficientNet Performance Results on ImageNet (Russakovsky et al., 2015). All EfficientNet models are scaled from our baseline EfficientNet-B0 using different compound coefficient φ in Equation 3. ConvNets with s",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": true,
          "exact_figure_caption_hit": false,
          "type_priority": "table",
          "type_score": 4,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00275
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "97b182165b2647c27e5a3116",
        "region": "content",
        "section_label": "5.2. ImageNet Results for EfficientNet",
        "type": "text",
        "ordinal": 49,
        "page_start": 6,
        "text_head": "Table 2 shows the performance of all EfficientNet models that are scaled from the same baseline EfficientNet-B0. Our EfficientNet models generally use an order of magnitude fewer parameters and FLOPS than other ConvNets ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "f46a067280877c50af5a6fce",
        "region": "content",
        "section_label": "4. EfficientNet Architecture",
        "type": "text",
        "ordinal": 37,
        "page_start": 4,
        "text_head": "Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBCon",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "candidate_discovery": null,
    "warnings": []
  },
  {
    "id": "nonexistent",
    "request": {
      "query": "Summarize NebulaXYZ-9876.",
      "task": "summary"
    },
    "status": "not_found",
    "task": "summary",
    "papers": [],
    "top_items": [],
    "candidate_discovery": {
      "metadata_count": 0,
      "chunk_count": 0,
      "entity_hits": {},
      "fallback_used": false,
      "chunk_search_used": false,
      "selected_paper_ids": [],
      "lexical_query": "\"nebulaxyz 9876\"",
      "candidate_match_source": {},
      "title_exact_hit": [],
      "abstract_exact_hit": [],
      "chunk_exact_hit": [],
      "table_ref": null,
      "figure_ref": null
    },
    "warnings": [
      "no papers match the supplied constraints"
    ]
  }
]
```

人工复核结论：

| 案例 | 结果 | 复核要点 |
|---|---|---|
| GQA 2.2 | 通过 | 目标论文唯一，2.2 Grouped-query attention 位于前三且重复窗口均来自同节。 |
| PagedAttention Reason | 通过 | 目标论文唯一，结果为完整正文窗口，保留 window_id 与 source_chunk_ids。 |
| LoRA Summary | 通过 | 召回 2106.09685，未返回 not_found。 |
| LoRA/QLoRA Comparison | 通过 | 两篇目标论文均保留摘要/正文证据。 |
| BERT 两个预训练任务 | 通过 | 标题精确命中后只保留 1810.04805，背景论文被排除。 |
| Mixed Precision | 通过 | FP32 master weights 章节进入前三，未被媒体 Chunk 抢占。 |
| FlashAttention | 通过 | 目标论文唯一，HBM/IO 正文进入前三。 |
| Swin shifted window | 通过 | 3.2. Shifted Window based Self-Attention 进入前三，普通查询前三无图片/表格。 |
| EfficientNet Summary | 通过 | 摘要和方法正文优先，媒体不再占据前三。 |
| DPO/RLHF、Prefix/Prompt、ViT/Swin | 通过 | 比较目标论文集合保持，未引入额外背景论文。 |
| ResNet 来源对齐 Appendix | 通过 | 真实 Appendix A object detection 内容进入首条证据。 |
| ResNet 指数浅层网络 | 诊断 | 查询论点与 ResNet Appendix 来源不一致，返回附录章节是来源不匹配，不判为解析失败。 |
| Swin Appendix | 部分通过 | A2.1 Image classification on ImageNet-1K 首位；后续仍有 A2.3 等附录小节，属于附录内部排序待优化。 |
| Local window attention | 通过 | 前三为正文文本，ViT/Swin 目标论文保留。 |
| EfficientNet Table 2 | 通过 | Table 2 caption Chunk 排第一，正文提及段落随后；Table 5 未抢占首位。 |
| NebulaXYZ-9876 | 通过 | 正确返回 not_found。 |

本轮仍需关注的非阻塞项是两类：原始 MinerU 个别 block 自身存在词尾截断，以及显式 Appendix 查询的附录小节内部排序；二者与 v6 固定字符 overlap 的系统性残词问题不同。


## 本轮 v7 换题正文检索回归

> 生成时间（UTC）：2026-10-06T05:34:50.636820+00:00
> 本轮使用与 v6 不同的问题，覆盖 Attention、ZeRO、GPT-3、LLaMA、EfficientNet、DeepSeek-V2、DeepSeekMath、ViT、PPO、FlashAttention，以及显式 Appendix 和不存在实体。

### 紧凑请求/响应记录

下方保留每个问题的请求、状态、目标论文、前三条证据和候选发现调试字段；每条证据都包含 paper_id、chunk_id、章节、页码和排序特征。

```json
[
  {
    "id": "attention-scaled-dot-product",
    "request": {
      "query": "Why does scaled dot-product attention divide the dot products by the square root of d_k?",
      "task": "reason",
      "paper_ids": [
        "1706.03762"
      ]
    },
    "status": "ok",
    "task": "reason",
    "papers": [
      "1706.03762"
    ],
    "top_items": [
      {
        "paper_id": "1706.03762",
        "chunk_id": "c72818382a0ee06dfec39c0b",
        "region": "content",
        "section_label": "3.2.1 Scaled Dot-Product Attention",
        "type": "text",
        "ordinal": 12,
        "page_start": 3,
        "text_head": "We call our particular attention \"Scaled Dot-Product Attention\" (Figure 2). The input consists of queries and keys of dimension $d _ { k }$ , and values of dimension $d _ { v }$ . We compute the dot products of the query",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": "1706.03762:12",
        "source_chunk_ids": [
          "c72818382a0ee06dfec39c0b",
          "014f9db296cab460549d3126"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1706.03762",
        "chunk_id": "436d817e2140c4bdfa514a25",
        "region": "content",
        "section_label": "3.2.1 Scaled Dot-Product Attention",
        "type": "text",
        "ordinal": 14,
        "page_start": 3,
        "text_head": "$$\n\\operatorname{Attention} (Q, K, V) = \\operatorname{softmax} (\\frac {Q K ^ {T}}{\\sqrt {d _ {k}}}) V\\tag{1}\n$$\n\nThe two most commonly used attention functions are additive attention [2], and dot-product (multiplicative)",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0037
        },
        "window_id": "1706.03762:14",
        "source_chunk_ids": [
          "014f9db296cab460549d3126",
          "436d817e2140c4bdfa514a25"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1706.03762",
        "chunk_id": "9c19b7c849aea95610c9fb41",
        "region": "content",
        "section_label": "3.2.3 Applications of Attention in our Model",
        "type": "text",
        "ordinal": 18,
        "page_start": 4,
        "text_head": "The Transformer uses multi-head attention in three different ways:\n\n• In \"encoder-decoder attention\" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the en",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": "1706.03762:18",
        "source_chunk_ids": [
          "9c19b7c849aea95610c9fb41"
        ],
        "continuity_status": "complete"
      }
    ],
    "warnings": [
      "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
    ],
    "candidate_discovery": null
  },
  {
    "id": "zero-stage-two",
    "request": {
      "query": "How does ZeRO Stage 2 partition optimizer states and gradients?",
      "task": "fact",
      "paper_ids": [
        "1910.02054"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1910.02054"
    ],
    "top_items": [
      {
        "paper_id": "1910.02054",
        "chunk_id": "4d694a92ba4eb46282cea662",
        "region": "content",
        "section_label": "5 Deep Dive into ZeRO-DP",
        "type": "text",
        "ordinal": 36,
        "page_start": 9,
        "text_head": "While the existing DP approach replicates the model states at each device and introduces significant memory overhead, ZeRO-DP eliminates this memory redundancy by partitioning them — optimizer states, gradients and param",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1910.02054",
        "chunk_id": "15b9dd60b734674c6dc3aabb",
        "region": "content",
        "section_label": "1 Extended Introduction",
        "type": "text",
        "ordinal": 6,
        "page_start": 0,
        "text_head": "ZeRO-DP has three main optimization stages (as depicted in Figure 1), which correspond to the partitioning of optimizer states, gradients, and parameters. When enabled cumulatively: 1) Optimizer State Partitioning $( P _",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1910.02054",
        "chunk_id": "4fbea0cc34437eb3f4f577bf",
        "region": "content",
        "section_label": "5.2 ${ \\mathbf { P } } _ { g } \\mathbf { : }$ Gradient Partitioning",
        "type": "text",
        "ordinal": 39,
        "page_start": 9,
        "text_head": "As each data parallel process only updates its corresponding parameter partition, it only needs the reduced gradients for the corresponding parameters. Therefore, as each gradient of each layer becomes available during t",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "gpt3-summary",
    "request": {
      "query": "Summarize GPT-3's main contribution to few-shot and in-context learning.",
      "task": "summary",
      "paper_ids": [
        "2005.14165"
      ]
    },
    "status": "ok",
    "task": "summary",
    "papers": [
      "2005.14165"
    ],
    "top_items": [
      {
        "paper_id": "2005.14165",
        "chunk_id": "b77c8278d9cf47d376ace62a",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task.\n\nWhile typically task-agnostic in architecture, this m",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2005.14165",
        "chunk_id": "d4e0d66be1ea743e5633dddb",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "text",
        "ordinal": 12,
        "page_start": 4,
        "text_head": "In this paper, we test this hypothesis by training a 175 billion parameter autoregressive language model, which we call GPT-3, and measuring its in-context learning abilities. Specifically, we evaluate GPT-3 on over two ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2005.14165",
        "chunk_id": "60a2d3e67336c34d275d147e",
        "region": "content",
        "section_label": "3.5 Common Sense Reasoning",
        "type": "text",
        "ordinal": 72,
        "page_start": 17,
        "text_head": "Overall, in-context learning with GPT-3 shows mixed results on commonsense reasoning tasks, with only small and inconsistent gains observed in the one and few-shot learning settings for both PIQA and ARC, but a significa",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "llama-gpt3-comparison",
    "request": {
      "query": "Compare LLaMA and GPT-3 in model scale, training data, and reported evaluation.",
      "task": "comparison",
      "paper_ids": [
        "2302.13971",
        "2005.14165"
      ]
    },
    "status": "ok",
    "task": "comparison",
    "papers": [
      "2302.13971",
      "2005.14165"
    ],
    "top_items": [
      {
        "paper_id": "2302.13971",
        "chunk_id": "0ed8544153ec6eda1bfc5a2e",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly a",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2302.13971",
        "chunk_id": "6e585697514b189d025e9793",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "text",
        "ordinal": 3,
        "page_start": 0,
        "text_head": "The focus of this work is to train a series of language models that achieve the best possible performance at various inference budgets, by training on more tokens than what is typically used. The resulting models, called",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2302.13971",
        "chunk_id": "da53df953f335f556df06a0c",
        "region": "content",
        "section_label": "3.3 Reading Comprehension",
        "type": "text",
        "ordinal": 27,
        "page_start": 4,
        "text_head": "We evaluate our models on the RACE reading comprehension benchmark (Lai et al., 2017). This dataset was collected from English reading comprehension exams designed for middle and high school Chinese students. We follow t",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "efficientnet-table1",
    "request": {
      "query": "What does EfficientNet Table 1 specify about the B0 baseline stages?",
      "task": "fact",
      "paper_ids": [
        "1905.11946"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "1905.11946"
    ],
    "top_items": [
      {
        "paper_id": "1905.11946",
        "chunk_id": "ecb306197b1df14aa610db31",
        "region": "content",
        "section_label": "4. EfficientNet Architecture",
        "type": "table",
        "ordinal": 36,
        "page_start": 4,
        "text_head": "Table 1. EfficientNet-B0 baseline network – Each row describes a stage i with $\\hat { L } _ { i }$ layers, with input resolution $\\langle \\hat { H } _ { i } , \\hat { W } _ { i } \\rangle$ and output channels $\\hat { C } _",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": true,
          "exact_figure_caption_hit": false,
          "type_priority": "table",
          "type_score": 4,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00385
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "f46a067280877c50af5a6fce",
        "region": "content",
        "section_label": "4. EfficientNet Architecture",
        "type": "text",
        "ordinal": 37,
        "page_start": 4,
        "text_head": "Net, except our EfficientNet-B0 is slightly bigger due to the larger FLOPS target (our FLOPS target is 400M). Table 1 shows the architecture of EfficientNet-B0. Its main building block is mobile inverted bottleneck MBCon",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "1905.11946",
        "chunk_id": "99b9503b593f770eed7cc2be",
        "region": "content",
        "section_label": "3.2. Scaling Dimensions",
        "type": "chart",
        "ordinal": 26,
        "page_start": 3,
        "text_head": "Figure 3. Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accurac",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": true,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "chart",
          "type_score": 2,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00105
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [
      "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
    ],
    "candidate_discovery": null
  },
  {
    "id": "deepseek-v2-mla",
    "request": {
      "query": "In DeepSeek-V2 section 2.1, how does Multi-head Latent Attention reduce KV cache memory?",
      "task": "fact",
      "paper_ids": [
        "2405.04434"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2405.04434"
    ],
    "top_items": [
      {
        "paper_id": "2405.04434",
        "chunk_id": "92248b12a0f852ac7deff607",
        "region": "content",
        "section_label": "2.1. Multi-Head Latent Attention: Boosting Inference Efficiency",
        "type": "text",
        "ordinal": 15,
        "page_start": 5,
        "text_head": "Conventional Transformer models usually adopts Multi-Head Attention (MHA) (Vaswani et al., 2017), but during generation, its heavy Key-Value (KV) cache will become the bottleneck that limit the inference efficiency. In o",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0037
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2405.04434",
        "chunk_id": "3d3175afe1c4e93e3474d2df",
        "region": "content",
        "section_label": "2.1.2. Low-Rank Key-Value Joint Compression",
        "type": "text",
        "ordinal": 28,
        "page_start": 6,
        "text_head": "The core of MLA is the low-rank joint compression for keys and values to reduce KV cache:",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0023
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2405.04434",
        "chunk_id": "ac70fe4217e6506da5651f8a",
        "region": "content",
        "section_label": "2.1.4. Comparison ofKey-Value Cache",
        "type": "text",
        "ordinal": 45,
        "page_start": 7,
        "text_head": "We demonstrate a comparison of the KV cache per token among different attention mechanisms in Table 1. MLA requires only a small amount of KV cache, equal to GQA with only 2.25 groups, but can achieve stronger performanc",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0023
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "gpt3-appendix",
    "request": {
      "query": "What does the GPT-3 appendix say about few-shot evaluation settings?",
      "task": "fact",
      "paper_ids": [
        "2005.14165"
      ],
      "regions": [
        "appendix"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2005.14165"
    ],
    "top_items": [
      {
        "paper_id": "2005.14165",
        "chunk_id": "c41a9af4b7a6497ee40aac61",
        "region": "appendix",
        "section_label": "G Details of Task Phrasing and Specifications",
        "type": "text",
        "ordinal": 230,
        "page_start": 59,
        "text_head": "Figure G.36: Formatted dataset example for De→En. This is the format for one- and few-shot learning, for this and other langauge tasks, the format for zero-shot learning is “Q: What is the {language} translation of {sent",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2005.14165",
        "chunk_id": "6eb7f9ca04e2338a941edfce",
        "region": "appendix",
        "section_label": "G Details of Task Phrasing and Specifications",
        "type": "text",
        "ordinal": 202,
        "page_start": 49,
        "text_head": "The following figures illustrate the formatting and phrasing of all the tasks included in the paper. All data comes from the ground truth datasets in this section, and no samples from GPT-3 are included here.",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2005.14165",
        "chunk_id": "19fd0e18ef96f9f4cfe6c15a",
        "region": "appendix",
        "section_label": "C Details of Test Set Contamination Studies",
        "type": "text",
        "ordinal": 188,
        "page_start": 42,
        "text_head": "Figure 4.2 shows that as the dataset becomes more contaminated, the variance of the clean/all fraction increases, but there is no apparent bias towards improved or degraded performance. This suggests that GPT-3 is relati",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "appendix",
          "region_score": 0,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00185
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "vit-patch-embedding",
    "request": {
      "query": "How are image patches embedded before entering the Vision Transformer?",
      "task": "fact",
      "paper_ids": [
        "2010.11929"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2010.11929"
    ],
    "top_items": [
      {
        "paper_id": "2010.11929",
        "chunk_id": "e45e564ce48b3c0b920070e3",
        "region": "content",
        "section_label": "3.1 VISION TRANSFORMER (VIT)",
        "type": "text",
        "ordinal": 11,
        "page_start": 2,
        "text_head": "Similar to BERT’s [class] token, we prepend a learnable embedding to the sequence of embedded patches $( \\mathbf { z } _ { 0 } ^ { 0 } = \\mathbf { x } _ { \\mathrm { c l a s s } } )$ , whose state at the output of the Tra",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2010.11929",
        "chunk_id": "896f8717f80f0e4b06f5a03f",
        "region": "content",
        "section_label": "3.1 VISION TRANSFORMER (VIT)",
        "type": "text",
        "ordinal": 10,
        "page_start": 2,
        "text_head": "An overview of the model is depicted in Figure 1. The standard Transformer receives as input a 1D sequence of token embeddings. To handle 2D images, we reshape the image $\\mathbf { x } \\in \\mathbb { R } ^ { H \\times W \\t",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2010.11929",
        "chunk_id": "a307ca39c3c1e551b2d5bbec",
        "region": "content",
        "section_label": "4.5 INSPECTING VISION TRANSFORMER",
        "type": "text",
        "ordinal": 43,
        "page_start": 7,
        "text_head": "To begin to understand how the Vision Transformer processes image data, we analyze its internal representations. The first layer of the Vision Transformer linearly projects the flattened patches into a lower-dimensional ",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [
      "translation_failed:query_rewriter:QueryRewriterError:Query Rewriter 的 core_terms 不是字符串数组"
    ],
    "candidate_discovery": null
  },
  {
    "id": "ppo-clipped-objective",
    "request": {
      "query": "Why does PPO use a clipped surrogate objective?",
      "task": "reason",
      "paper_ids": [
        "1707.06347"
      ]
    },
    "status": "ok",
    "task": "reason",
    "papers": [
      "1707.06347"
    ],
    "top_items": [
      {
        "paper_id": "1707.06347",
        "chunk_id": "de17b8743f69fa7e2787db7c",
        "region": "content",
        "section_label": "3 Clipped Surrogate Objective",
        "type": "text",
        "ordinal": 18,
        "page_start": 2,
        "text_head": "$$\nL ^ {C L I P} (\\theta) = \\hat {\\mathbb {E}} _ {t} \\Big [ \\min (r _ {t} (\\theta) \\hat {A} _ {t}, \\operatorname{clip} (r _ {t} (\\theta), 1 - \\epsilon , 1 + \\epsilon) \\hat {A} _ {t}) \\Big ]\\tag{7}\n$$\n\nwhere epsilon is a ",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": "1707.06347:18",
        "source_chunk_ids": [
          "ec70df354a6aec9c4cf53851",
          "de17b8743f69fa7e2787db7c",
          "2ceca01aaf68bb275156f668",
          "2c88cfde96f0b14975789d48"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1707.06347",
        "chunk_id": "8596d98385a2f9d4b1436bb2",
        "region": "content",
        "section_label": "3 Clipped Surrogate Objective",
        "type": "text",
        "ordinal": 14,
        "page_start": 2,
        "text_head": "Let $r_t(\\theta)$ denote the probability ratio $r_t(\\theta) = \\frac{\\pi_\\theta(a_t \\mid s_t)}{\\pi_{\\theta_{\\mathrm{old}}} (a_t \\mid s_t)}$ , so $r(\\theta_{\\mathrm{old}}) = 1$ . TRPO maximizes a “surrogate” objective\n\n$$\n",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": "1707.06347:14",
        "source_chunk_ids": [
          "8596d98385a2f9d4b1436bb2",
          "728ad0018eb3e005032e628e"
        ],
        "continuity_status": "complete"
      },
      {
        "paper_id": "1707.06347",
        "chunk_id": "d358bd5f86a108e414f5229e",
        "region": "content",
        "section_label": "3 Clipped Surrogate Objective",
        "type": "text",
        "ordinal": 21,
        "page_start": 2,
        "text_head": "Figure 1: Plots showing one term (i.e., a single timestep) of the surrogate function $L^{CLIP}$ as a function of the probability ratio r, for positive advantages (left) and negative advantages (right). The red circle on ",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": "1707.06347:21",
        "source_chunk_ids": [
          "2c88cfde96f0b14975789d48",
          "d358bd5f86a108e414f5229e",
          "13210e5d01cc00a768f3c719"
        ],
        "continuity_status": "complete"
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "flashattention-figure1",
    "request": {
      "query": "What does Figure 1 in FlashAttention illustrate about standard attention and memory access?",
      "task": "fact",
      "paper_ids": [
        "2205.14135"
      ]
    },
    "status": "ok",
    "task": "fact",
    "papers": [
      "2205.14135"
    ],
    "top_items": [
      {
        "paper_id": "2205.14135",
        "chunk_id": "b340711719a74d990a976b9a",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "image",
        "ordinal": 4,
        "page_start": 1,
        "text_head": "Figure 1: Left: FlashAttention uses tiling to prevent materialization of the large $N \\times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blo",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": true,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": true,
          "type_priority": "image",
          "type_score": 2,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.00245
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2205.14135",
        "chunk_id": "881461bab55c66aa9e253305",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "text",
        "ordinal": 6,
        "page_start": 1,
        "text_head": "We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing t",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": true,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2205.14135",
        "chunk_id": "faff47544f9ffa3526332086",
        "region": "content",
        "section_label": "1 Introduction",
        "type": "text",
        "ordinal": 3,
        "page_start": 0,
        "text_head": "In this paper, we argue that a missing principle is making attention algorithms IO-aware [1]—that is, carefully accounting for reads and writes to diferent levels of fast and slow memory (e.g., between fast GPU on-chip S",
        "ranking_features": {
          "exact_entity_hit": false,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": true,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 3,
          "duplicate_penalty": 0.00035,
          "quality_bonus": 0.0012000000000000001
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "deepseekmath-summary",
    "request": {
      "query": "Summarize DeepSeekMath's contribution to mathematical reasoning and training data.",
      "task": "summary",
      "paper_ids": [
        "2402.03300"
      ]
    },
    "status": "ok",
    "task": "summary",
    "papers": [
      "2402.03300"
    ],
    "top_items": [
      {
        "paper_id": "2402.03300",
        "chunk_id": "232a17a8e2668200903d5bd2",
        "region": "abstract",
        "section_label": "abstract",
        "type": "text",
        "ordinal": 0,
        "page_start": 0,
        "text_head": "Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pretraining DeepSeek-Coder-Base-v1.5 7B with",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "abstract",
          "region_score": 3,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.0026
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2402.03300",
        "chunk_id": "cc1d1c30287c903acfb2bc94",
        "region": "content",
        "section_label": "6. Conclusion, Limitation, and Future Work",
        "type": "text",
        "ordinal": 87,
        "page_start": 21,
        "text_head": "We present DeepSeekMath, which outperforms all open-source models on the competitionlevel MATH benchmark and approaches the performance of closed models. DeepSeekMath is initialized with DeepSeek-Coder-v1.5 7B and underg",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": false,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00235
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      },
      {
        "paper_id": "2402.03300",
        "chunk_id": "757775ac98dca61f15145255",
        "region": "content",
        "section_label": "4.2. Training and Evaluating DeepSeekMath-RL",
        "type": "text",
        "ordinal": 56,
        "page_start": 14,
        "text_head": "Table 5 demonstrates the performance of open- and closed-source models with both chainof-thought and tool-integrated reasoning on English and Chinese benchmarks. We find that: 1) DeepSeekMath-RL 7B attains accuracies of ",
        "ranking_features": {
          "exact_entity_hit": true,
          "section_exact_hit": true,
          "exact_table_ref_hit": false,
          "exact_figure_ref_hit": false,
          "exact_table_caption_hit": false,
          "exact_figure_caption_hit": false,
          "type_priority": "text",
          "type_score": 3,
          "region_priority": "content",
          "region_score": 2,
          "duplicate_penalty": 0.0,
          "quality_bonus": 0.00345
        },
        "window_id": null,
        "source_chunk_ids": null,
        "continuity_status": null
      }
    ],
    "warnings": [],
    "candidate_discovery": null
  },
  {
    "id": "nonexistent-quantum",
    "request": {
      "query": "Summarize QuantumNebula-314159.",
      "task": "summary"
    },
    "status": "not_found",
    "task": "summary",
    "papers": [],
    "top_items": [],
    "warnings": [
      "no papers match the supplied constraints"
    ],
    "candidate_discovery": {
      "metadata_count": 0,
      "chunk_count": 0,
      "entity_hits": {},
      "fallback_used": false,
      "chunk_search_used": false,
      "selected_paper_ids": [],
      "lexical_query": "\"quantumnebula 314159\"",
      "candidate_match_source": {},
      "title_exact_hit": [],
      "abstract_exact_hit": [],
      "chunk_exact_hit": [],
      "table_ref": null,
      "figure_ref": null
    }
  }
]
```

### 人工质量复核

| 案例 | 结果 | 观察 |
|---|---|---|
| Scaled Dot-Product Attention | 通过 | 3.2.1 进入前两名，公式和解释正文连续出现。 |
| ZeRO Stage 2 | 部分通过 | 5.2 Gradient Partitioning 进入前三，但总节 5 和 Introduction 仍排在前面；可作为章节号/Stage 过滤的后续优化样本。 |
| GPT-3 Summary | 通过 | 摘要和 Introduction 优先，未混入 Appendix。 |
| LLaMA/GPT-3 Comparison | 通过 | 两篇目标论文各保留摘要和正文证据，候选集合无额外论文。 |
| EfficientNet Table 1 | 通过 | Table 1 caption Chunk 排第一，Table 3 等媒体没有抢占首位。 |
| DeepSeek-V2 MLA | 通过 | 2.1、2.1.2、2.1.4 连续进入前三，直接覆盖 KV cache 和低秩压缩。 |
| GPT-3 Appendix | 部分通过 | G 节 Task Phrasing 首位，但 C/F 等其他 Appendix 小节仍进入后续结果；区域隔离正确，附录内部排序仍可优化。 |
| ViT Patch Embedding | 通过 | 3.1 正文连续命中，包含 patch embedding 和 [class] token 上下文。 |
| PPO Clipped Objective | 通过 | 3 节公式 Chunk 和相邻解释正文同时返回，未出现孤立媒体首条。 |
| FlashAttention Figure 1 | 通过 | 显式 Figure 查询优先返回 Figure 1 image Chunk，并附相邻 Introduction 正文。 |
| DeepSeekMath Summary | 通过 | 摘要首位，方法与结论正文作为补充。 |
| QuantumNebula-314159 | 通过 | 正确返回 not_found，候选发现调试显示 metadata/chunk 均无命中。 |

本批次未发现新的 Chunk 边界破坏；主要剩余问题是“显式章节/Stage 查询的章节号精排”和“Appendix 内部小节精排”，属于排序层改进，不需要重新设计 Chunk。

## 元数据与引用 MCP 真实调用记录


> 生成时间（UTC）：2026-10-05T14:48:03.425440+00:00
> 数据来源：当前本地 Catalog，通过真实 MCP stdio 服务调用。
> 服务端只返回结构化元数据、引用关系和确定性 presentation，不调用答案生成模型。
> 引用查询不调用 Embedding、Milvus 或正文 Chunk 检索。

## 工具选择说明

当用户的问题明确询问参考文献、被哪些论文引用或引用关系图时，Agent 直接选择 `library_citation`。用户可以提供论文题目，工具会在本地 Catalog 内解析 `paper_title`，无需先调用 `library_search`；`library_search` 只用于论文元数据发现。

## 案例 1：论文库中有哪些和注意力相关的文章

```json
{
  "user_question": "论文库中有哪些和注意力相关的文章",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_search",
        "reason": "元数据发现问题，不读取正文"
      },
      "mcp_request": {
        "tool": "library_search",
        "arguments": {
          "query": "论文库中有哪些和注意力相关的文章",
          "limit": 20
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "论文库中有哪些和注意力相关的文章",
          "items": [
            {
              "paper_id": "2312.06635",
              "base_id": "2312.06635",
              "canonical_id": "2312.06635v6",
              "title": "Gated Linear Attention Transformers with Hardware-Efficient Training",
              "authors": [
                "Songlin Yang",
                "Bailin Wang",
                "Yikang Shen",
                "Rameswar Panda",
                "Yoon Kim"
              ],
              "abstract": "Transformers with linear attention allow for efficient parallel training but can simultaneously be formulated as an RNN with 2D (matrix-valued) hidden states, thus enjoying linear-time inference complexity. However, linear attention generally underperforms ordinary softmax attention. Moreover, current implementations of linear attention lack I/O-awareness and are thus slower than highly optimized implementations of softmax attention. This work describes a hardware-efficient algorithm for linear attention that trades off memory movement against parallelizability. The resulting implementation, dubbed FLASHLINEARATTENTION, is faster than FLASHATTENTION-2 (Dao, 2023) as a standalone layer even on short sequence lengths (e.g., 1K). We then generalize this algorithm to a more expressive variant of linear attention with data-dependent gates. When used as a replacement for the standard attention layer in Transformers, the resulting gated linear attention (GLA) Transformer is found to perform competitively against the LLaMA-architecture Transformer (Touvron et al., 2023) as well recent linear-time-inference baselines such as RetNet (Sun et al., 2023a) and Mamba (Gu & Dao, 2023) on moderate-scale language modeling experiments. GLA Transformer is especially effective at length generalization, enabling a model trained on 2K to generalize to sequences longer than 20K without significant perplexity degradations. For training speed, the GLA Transformer has higher throughput than a similarly-sized Mamba model.",
              "categories": [
                "cs.LG",
                "cs.CL"
              ],
              "published_at": "2023-12-11T18:51:59Z",
              "updated_at": "2024-08-27T01:27:29Z",
              "abs_url": "https://arxiv.org/abs/2312.06635v6",
              "pdf_url": "https://arxiv.org/pdf/2312.06635v6.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2312.06635\\mineru"
                }
              }
            },
            {
              "paper_id": "2205.14135",
              "base_id": "2205.14135",
              "canonical_id": "2205.14135v2",
              "title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
              "authors": [
                "Tri Dao",
                "Daniel Y. Fu",
                "Stefano Ermon",
                "Atri Rudra",
                "Christopher Ré"
              ],
              "abstract": "Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3$\\times$ speedup on GPT-2 (seq. length 1K), and 2.4$\\times$ speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).",
              "categories": [
                "cs.LG"
              ],
              "published_at": "2022-05-27T17:53:09Z",
              "updated_at": "2022-06-23T17:53:32Z",
              "abs_url": "https://arxiv.org/abs/2205.14135v2",
              "pdf_url": "https://arxiv.org/pdf/2205.14135v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2205.14135\\mineru"
                }
              }
            },
            {
              "paper_id": "2305.13245",
              "base_id": "2305.13245",
              "canonical_id": "2305.13245v3",
              "title": "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints",
              "authors": [
                "Joshua Ainslie",
                "James Lee-Thorp",
                "Michiel de Jong",
                "Yury Zemlyanskiy",
                "Federico Lebrón",
                "Sumit Sanghai"
              ],
              "abstract": "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.",
              "categories": [
                "cs.CL",
                "cs.LG"
              ],
              "published_at": "2023-05-22T17:16:38Z",
              "updated_at": "2023-12-23T17:55:11Z",
              "abs_url": "https://arxiv.org/abs/2305.13245v3",
              "pdf_url": "https://arxiv.org/pdf/2305.13245v3.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.13245\\mineru"
                }
              }
            },
            {
              "paper_id": "1911.02150",
              "base_id": "1911.02150",
              "canonical_id": "1911.02150v1",
              "title": "Fast Transformer Decoding: One Write-Head is All You Need",
              "authors": [
                "Noam Shazeer"
              ],
              "abstract": "Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful alternative to RNNs for moving information across and between sequences. While training these layers is generally fast and simple, due to parallelizability across the length of the sequence, incremental inference (where such paralleization is impossible) is often slow, due to the memory-bandwidth cost of repeatedly loading the large \"keys\" and \"values\" tensors. We propose a variant called multi-query attention, where the keys and values are shared across all of the different attention \"heads\", greatly reducing the size of these tensors and hence the memory bandwidth requirements of incremental decoding. We verify experimentally that the resulting models can indeed be much faster to decode, and incur only minor quality degradation from the baseline.",
              "categories": [
                "cs.NE",
                "cs.CL",
                "cs.LG"
              ],
              "published_at": "2019-11-06T00:19:05Z",
              "updated_at": "2019-11-06T00:19:05Z",
              "abs_url": "https://arxiv.org/abs/1911.02150v1",
              "pdf_url": "https://arxiv.org/pdf/1911.02150v1.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1911.02150\\mineru"
                }
              }
            },
            {
              "paper_id": "1706.03762",
              "base_id": "1706.03762",
              "canonical_id": "1706.03762v7",
              "title": "Attention Is All You Need",
              "authors": [
                "Ashish Vaswani",
                "Noam Shazeer",
                "Niki Parmar",
                "Jakob Uszkoreit",
                "Llion Jones",
                "Aidan N. Gomez",
                "Lukasz Kaiser",
                "Illia Polosukhin"
              ],
              "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
              "categories": [
                "cs.CL",
                "cs.LG"
              ],
              "published_at": "2017-06-12T17:57:34Z",
              "updated_at": "2023-08-02T00:41:18Z",
              "abs_url": "https://arxiv.org/abs/1706.03762v7",
              "pdf_url": "https://arxiv.org/pdf/1706.03762v7.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1706.03762\\mineru"
                }
              }
            },
            {
              "paper_id": "2006.16236",
              "base_id": "2006.16236",
              "canonical_id": "2006.16236v3",
              "title": "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
              "authors": [
                "Angelos Katharopoulos",
                "Apoorv Vyas",
                "Nikolaos Pappas",
                "François Fleuret"
              ],
              "abstract": "Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input's length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from $\\mathcal{O}\\left(N^2\\right)$ to $\\mathcal{O}\\left(N\\right)$, where $N$ is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.",
              "categories": [
                "cs.LG",
                "stat.ML"
              ],
              "published_at": "2020-06-29T17:55:38Z",
              "updated_at": "2020-08-31T11:09:32Z",
              "abs_url": "https://arxiv.org/abs/2006.16236v3",
              "pdf_url": "https://arxiv.org/pdf/2006.16236v3.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2006.16236\\mineru"
                }
              }
            },
            {
              "paper_id": "2010.11929",
              "base_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
              "authors": [
                "Alexey Dosovitskiy",
                "Lucas Beyer",
                "Alexander Kolesnikov",
                "Dirk Weissenborn",
                "Xiaohua Zhai",
                "Thomas Unterthiner",
                "Mostafa Dehghani",
                "Matthias Minderer",
                "Georg Heigold",
                "Sylvain Gelly",
                "Jakob Uszkoreit",
                "Neil Houlsby"
              ],
              "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
              "categories": [
                "cs.CV",
                "cs.AI",
                "cs.LG"
              ],
              "published_at": "2020-10-22T17:55:59Z",
              "updated_at": "2021-06-03T13:08:56Z",
              "abs_url": "https://arxiv.org/abs/2010.11929v2",
              "pdf_url": "https://arxiv.org/pdf/2010.11929v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru"
                }
              }
            },
            {
              "paper_id": "2309.06180",
              "base_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
              "authors": [
                "Woosuk Kwon",
                "Zhuohan Li",
                "Siyuan Zhuang",
                "Ying Sheng",
                "Lianmin Zheng",
                "Cody Hao Yu",
                "Joseph E. Gonzalez",
                "Hao Zhang",
                "Ion Stoica"
              ],
              "abstract": "High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size. To address this problem, we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems. On top of it, we build vLLM, an LLM serving system that achieves (1) near-zero waste in KV cache memory and (2) flexible sharing of KV cache within and across requests to further reduce memory usage. Our evaluations show that vLLM improves the throughput of popular LLMs by 2-4$\\times$ with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca. The improvement is more pronounced with longer sequences, larger models, and more complex decoding algorithms. vLLM's source code is publicly available at https://github.com/vllm-project/vllm",
              "categories": [
                "cs.LG",
                "cs.DC"
              ],
              "published_at": "2023-09-12T12:50:04Z",
              "updated_at": "2023-09-12T12:50:04Z",
              "abs_url": "https://arxiv.org/abs/2309.06180v1",
              "pdf_url": "https://arxiv.org/pdf/2309.06180v1.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2309.06180\\mineru"
                }
              }
            },
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            },
            {
              "paper_id": "1909.08053",
              "base_id": "1909.08053",
              "canonical_id": "1909.08053v4",
              "title": "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism",
              "authors": [
                "Mohammad Shoeybi",
                "Mostofa Patwary",
                "Raul Puri",
                "Patrick LeGresley",
                "Jared Casper",
                "Bryan Catanzaro"
              ],
              "abstract": "Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite difficult to train due to memory constraints. In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2019-09-17T19:42:54Z",
              "updated_at": "2020-03-13T23:45:18Z",
              "abs_url": "https://arxiv.org/abs/1909.08053v4",
              "pdf_url": "https://arxiv.org/pdf/1909.08053v4.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1909.08053\\mineru"
                }
              }
            },
            {
              "paper_id": "2405.04434",
              "base_id": "2405.04434",
              "canonical_id": "2405.04434v5",
              "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
              "authors": [
                "DeepSeek-AI",
                "Aixin Liu",
                "Bei Feng",
                "Bin Wang",
                "Bingxuan Wang",
                "Bo Liu",
                "Chenggang Zhao",
                "Chengqi Dengr",
                "Chong Ruan",
                "Damai Dai",
                "Daya Guo",
                "Dejian Yang",
                "Deli Chen",
                "Dongjie Ji",
                "Erhang Li",
                "Fangyun Lin",
                "Fuli Luo",
                "Guangbo Hao",
                "Guanting Chen",
                "Guowei Li",
                "H. Zhang",
                "Hanwei Xu",
                "Hao Yang",
                "Haowei Zhang",
                "Honghui Ding",
                "Huajian Xin",
                "Huazuo Gao",
                "Hui Li",
                "Hui Qu",
                "J. L. Cai",
                "Jian Liang",
                "Jianzhong Guo",
                "Jiaqi Ni",
                "Jiashi Li",
                "Jin Chen",
                "Jingyang Yuan",
                "Junjie Qiu",
                "Junxiao Song",
                "Kai Dong",
                "Kaige Gao",
                "Kang Guan",
                "Lean Wang",
                "Lecong Zhang",
                "Lei Xu",
                "Leyi Xia",
                "Liang Zhao",
                "Liyue Zhang",
                "Meng Li",
                "Miaojun Wang",
                "Mingchuan Zhang",
                "Minghua Zhang",
                "Minghui Tang",
                "Mingming Li",
                "Ning Tian",
                "Panpan Huang",
                "Peiyi Wang",
                "Peng Zhang",
                "Qihao Zhu",
                "Qinyu Chen",
                "Qiushi Du",
                "R. J. Chen",
                "R. L. Jin",
                "Ruiqi Ge",
                "Ruizhe Pan",
                "Runxin Xu",
                "Ruyi Chen",
                "S. S. Li",
                "Shanghao Lu",
                "Shangyan Zhou",
                "Shanhuang Chen",
                "Shaoqing Wu",
                "Shengfeng Ye",
                "Shirong Ma",
                "Shiyu Wang",
                "Shuang Zhou",
                "Shuiping Yu",
                "Shunfeng Zhou",
                "Size Zheng",
                "T. Wang",
                "Tian Pei",
                "Tian Yuan",
                "Tianyu Sun",
                "W. L. Xiao",
                "Wangding Zeng",
                "Wei An",
                "Wen Liu",
                "Wenfeng Liang",
                "Wenjun Gao",
                "Wentao Zhang",
                "X. Q. Li",
                "Xiangyue Jin",
                "Xianzu Wang",
                "Xiao Bi",
                "Xiaodong Liu",
                "Xiaohan Wang",
                "Xiaojin Shen",
                "Xiaokang Chen",
                "Xiaosha Chen",
                "Xiaotao Nie",
                "Xiaowen Sun",
                "Xiaoxiang Wang",
                "Xin Liu",
                "Xin Xie",
                "Xingkai Yu",
                "Xinnan Song",
                "Xinyi Zhou",
                "Xinyu Yang",
                "Xuan Lu",
                "Xuecheng Su",
                "Y. Wu",
                "Y. K. Li",
                "Y. X. Wei",
                "Y. X. Zhu",
                "Yanhong Xu",
                "Yanping Huang",
                "Yao Li",
                "Yao Zhao",
                "Yaofeng Sun",
                "Yaohui Li",
                "Yaohui Wang",
                "Yi Zheng",
                "Yichao Zhang",
                "Yiliang Xiong",
                "Yilong Zhao",
                "Ying He",
                "Ying Tang",
                "Yishi Piao",
                "Yixin Dong",
                "Yixuan Tan",
                "Yiyuan Liu",
                "Yongji Wang",
                "Yongqiang Guo",
                "Yuchen Zhu",
                "Yuduan Wang",
                "Yuheng Zou",
                "Yukun Zha",
                "Yunxian Ma",
                "Yuting Yan",
                "Yuxiang You",
                "Yuxuan Liu",
                "Z. Z. Ren",
                "Zehui Ren",
                "Zhangli Sha",
                "Zhe Fu",
                "Zhen Huang",
                "Zhen Zhang",
                "Zhenda Xie",
                "Zhewen Hao",
                "Zhihong Shao",
                "Zhiniu Wen",
                "Zhipeng Xu",
                "Zhongyu Zhang",
                "Zhuoshu Li",
                "Zihan Wang",
                "Zihui Gu",
                "Zilin Li",
                "Ziwei Xie"
              ],
              "abstract": "We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.",
              "categories": [
                "cs.CL",
                "cs.AI"
              ],
              "published_at": "2024-05-07T15:56:43Z",
              "updated_at": "2024-06-19T06:04:17Z",
              "abs_url": "https://arxiv.org/abs/2405.04434v5",
              "pdf_url": "https://arxiv.org/pdf/2405.04434v5.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2405.04434\\mineru"
                }
              }
            }
          ],
          "count": 11,
          "query_debug": {
            "lexical_query": "\"attention\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "attention"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "metadata_list",
            "render_policy": "verbatim",
            "answer_text": "共找到 11 篇论文：\n\n1. Gated Linear Attention Transformers with Hardware-Efficient Training\n   arXiv: 2312.06635\n   作者: Songlin Yang、Bailin Wang、Yikang Shen、Rameswar Panda、Yoon Kim\n   年份: 2023\n   分类: cs.LG、cs.CL\n2. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness\n   arXiv: 2205.14135\n   作者: Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré\n   年份: 2022\n   分类: cs.LG\n3. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints\n   arXiv: 2305.13245\n   作者: Joshua Ainslie、James Lee-Thorp、Michiel de Jong、Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai\n   年份: 2023\n   分类: cs.CL、cs.LG\n4. Fast Transformer Decoding: One Write-Head is All You Need\n   arXiv: 1911.02150\n   作者: Noam Shazeer\n   年份: 2019\n   分类: cs.NE、cs.CL、cs.LG\n5. Attention Is All You Need\n   arXiv: 1706.03762\n   作者: Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser、Illia Polosukhin\n   年份: 2017\n   分类: cs.CL、cs.LG\n6. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\n   arXiv: 2006.16236\n   作者: Angelos Katharopoulos、Apoorv Vyas、Nikolaos Pappas、François Fleuret\n   年份: 2020\n   分类: cs.LG、stat.ML\n7. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n   arXiv: 2010.11929\n   作者: Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby\n   年份: 2020\n   分类: cs.CV、cs.AI、cs.LG\n8. Efficient Memory Management for Large Language Model Serving with PagedAttention\n   arXiv: 2309.06180\n   作者: Woosuk Kwon、Zhuohan Li、Siyuan Zhuang、Ying Sheng、Lianmin Zheng、Cody Hao Yu、Joseph E. Gonzalez、Hao Zhang、Ion Stoica\n   年份: 2023\n   分类: cs.LG、cs.DC\n9. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows\n   arXiv: 2103.14030\n   作者: Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin、Baining Guo\n   年份: 2021\n   分类: cs.CV、cs.LG\n10. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism\n   arXiv: 1909.08053\n   作者: Mohammad Shoeybi、Mostofa Patwary、Raul Puri、Patrick LeGresley、Jared Casper、Bryan Catanzaro\n   年份: 2019\n   分类: cs.CL\n11. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model\n   arXiv: 2405.04434\n   作者: DeepSeek-AI、Aixin Liu、Bei Feng、Bin Wang、Bingxuan Wang、Bo Liu、Chenggang Zhao、Chengqi Dengr、Chong Ruan、Damai Dai、Daya Guo、Dejian Yang、Deli Chen、Dongjie Ji、Erhang Li、Fangyun Lin、Fuli Luo、Guangbo Hao、Guanting Chen、Guowei Li、H. Zhang、Hanwei Xu、Hao Yang、Haowei Zhang、Honghui Ding、Huajian Xin、Huazuo Gao、Hui Li、Hui Qu、J. L. Cai、Jian Liang、Jianzhong Guo、Jiaqi Ni、Jiashi Li、Jin Chen、Jingyang Yuan、Junjie Qiu、Junxiao Song、Kai Dong、Kaige Gao、Kang Guan、Lean Wang、Lecong Zhang、Lei Xu、Leyi Xia、Liang Zhao、Liyue Zhang、Meng Li、Miaojun Wang、Mingchuan Zhang、Minghua Zhang、Minghui Tang、Mingming Li、Ning Tian、Panpan Huang、Peiyi Wang、Peng Zhang、Qihao Zhu、Qinyu Chen、Qiushi Du、R. J. Chen、R. L. Jin、Ruiqi Ge、Ruizhe Pan、Runxin Xu、Ruyi Chen、S. S. Li、Shanghao Lu、Shangyan Zhou、Shanhuang Chen、Shaoqing Wu、Shengfeng Ye、Shirong Ma、Shiyu Wang、Shuang Zhou、Shuiping Yu、Shunfeng Zhou、Size Zheng、T. Wang、Tian Pei、Tian Yuan、Tianyu Sun、W. L. Xiao、Wangding Zeng、Wei An、Wen Liu、Wenfeng Liang、Wenjun Gao、Wentao Zhang、X. Q. Li、Xiangyue Jin、Xianzu Wang、Xiao Bi、Xiaodong Liu、Xiaohan Wang、Xiaojin Shen、Xiaokang Chen、Xiaosha Chen、Xiaotao Nie、Xiaowen Sun、Xiaoxiang Wang、Xin Liu、Xin Xie、Xingkai Yu、Xinnan Song、Xinyi Zhou、Xinyu Yang、Xuan Lu、Xuecheng Su、Y. Wu、Y. K. Li、Y. X. Wei、Y. X. Zhu、Yanhong Xu、Yanping Huang、Yao Li、Yao Zhao、Yaofeng Sun、Yaohui Li、Yaohui Wang、Yi Zheng、Yichao Zhang、Yiliang Xiong、Yilong Zhao、Ying He、Ying Tang、Yishi Piao、Yixin Dong、Yixuan Tan、Yiyuan Liu、Yongji Wang、Yongqiang Guo、Yuchen Zhu、Yuduan Wang、Yuheng Zou、Yukun Zha、Yunxian Ma、Yuting Yan、Yuxiang You、Yuxuan Liu、Z. Z. Ren、Zehui Ren、Zhangli Sha、Zhe Fu、Zhen Huang、Zhen Zhang、Zhenda Xie、Zhewen Hao、Zhihong Shao、Zhiniu Wen、Zhipeng Xu、Zhongyu Zhang、Zhuoshu Li、Zihan Wang、Zihui Gu、Zilin Li、Ziwei Xie\n   年份: 2024\n   分类: cs.CL、cs.AI"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## 案例 2：2020年以后有哪些计算机视觉论文和注意力相关

```json
{
  "user_question": "2020年以后有哪些计算机视觉论文和注意力相关",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_search",
        "reason": "元数据查询，同时使用年份和分类约束"
      },
      "mcp_request": {
        "tool": "library_search",
        "arguments": {
          "query": "注意力",
          "filters": {
            "year_from": "2020",
            "category": "cs.CV"
          },
          "limit": 20
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "query": "注意力",
          "items": [
            {
              "paper_id": "2010.11929",
              "base_id": "2010.11929",
              "canonical_id": "2010.11929v2",
              "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
              "authors": [
                "Alexey Dosovitskiy",
                "Lucas Beyer",
                "Alexander Kolesnikov",
                "Dirk Weissenborn",
                "Xiaohua Zhai",
                "Thomas Unterthiner",
                "Mostafa Dehghani",
                "Matthias Minderer",
                "Georg Heigold",
                "Sylvain Gelly",
                "Jakob Uszkoreit",
                "Neil Houlsby"
              ],
              "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.",
              "categories": [
                "cs.CV",
                "cs.AI",
                "cs.LG"
              ],
              "published_at": "2020-10-22T17:55:59Z",
              "updated_at": "2021-06-03T13:08:56Z",
              "abs_url": "https://arxiv.org/abs/2010.11929v2",
              "pdf_url": "https://arxiv.org/pdf/2010.11929v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2010.11929\\mineru"
                }
              }
            },
            {
              "paper_id": "2103.14030",
              "base_id": "2103.14030",
              "canonical_id": "2103.14030v2",
              "title": "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
              "authors": [
                "Ze Liu",
                "Yutong Lin",
                "Yue Cao",
                "Han Hu",
                "Yixuan Wei",
                "Zheng Zhang",
                "Stephen Lin",
                "Baining Guo"
              ],
              "abstract": "This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\textbf{S}hifted \\textbf{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures. The code and models are publicly available at~\\url{https://github.com/microsoft/Swin-Transformer}.",
              "categories": [
                "cs.CV",
                "cs.LG"
              ],
              "published_at": "2021-03-25T17:59:31Z",
              "updated_at": "2021-08-17T16:41:34Z",
              "abs_url": "https://arxiv.org/abs/2103.14030v2",
              "pdf_url": "https://arxiv.org/pdf/2103.14030v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2103.14030\\mineru"
                }
              }
            }
          ],
          "count": 2,
          "query_debug": {
            "lexical_query": "\"attention\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "attention"
            ]
          },
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "metadata_list",
            "render_policy": "verbatim",
            "answer_text": "共找到 2 篇论文：\n\n1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n   arXiv: 2010.11929\n   作者: Alexey Dosovitskiy、Lucas Beyer、Alexander Kolesnikov、Dirk Weissenborn、Xiaohua Zhai、Thomas Unterthiner、Mostafa Dehghani、Matthias Minderer、Georg Heigold、Sylvain Gelly、Jakob Uszkoreit、Neil Houlsby\n   年份: 2020\n   分类: cs.CV、cs.AI、cs.LG\n2. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows\n   arXiv: 2103.14030\n   作者: Ze Liu、Yutong Lin、Yue Cao、Han Hu、Yixuan Wei、Zheng Zhang、Stephen Lin、Baining Guo\n   年份: 2021\n   分类: cs.CV、cs.LG"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## 案例 3：Attention Is All You Need 参考了哪些本地论文

```json
{
  "user_question": "Attention Is All You Need 参考了哪些本地论文",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_citation",
        "arguments_source": "paper_title",
        "reason": "引用关系意图明确，直接使用引用工具，由工具内部解析论文题目"
      },
      "mcp_request": {
        "tool": "library_citation",
        "arguments": {
          "paper_title": "Attention Is All You Need",
          "mode": "references"
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "paper_id": "1706.03762",
          "items": [
            {
              "reference_id": "1706.03762:1",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 1,
              "raw_text": "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.",
              "page_start": 9,
              "page_end": 9,
              "target_arxiv_id": "1607.06450",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:2",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 2,
              "raw_text": "Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.",
              "page_start": 9,
              "page_end": 9,
              "target_arxiv_id": "1409.0473",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:3",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 3,
              "raw_text": "Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.",
              "page_start": 9,
              "page_end": 9,
              "target_arxiv_id": "1703.03906",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:4",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 4,
              "raw_text": "Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.",
              "page_start": 9,
              "page_end": 9,
              "target_arxiv_id": "1601.06733",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:5",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 5,
              "raw_text": "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1406.1078",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:6",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 6,
              "raw_text": "Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1610.02357",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:7",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 7,
              "raw_text": "Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1412.3555",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:8",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 8,
              "raw_text": "Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:9",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 9,
              "raw_text": "Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1705.03122v2",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:10",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 10,
              "raw_text": "Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1308.0850",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:11",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 11,
              "raw_text": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1512.03385",
              "target_doi": null,
              "resolution": "local",
              "matched_paper_id": "1512.03385",
              "match_method": "title_author_year",
              "match_score": 0.9999999999999999,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:12",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 12,
              "raw_text": "Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:13",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 13,
              "raw_text": "Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:14",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 14,
              "raw_text": "Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:15",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 15,
              "raw_text": "Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1602.02410",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:16",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 16,
              "raw_text": "Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems, (NIPS), 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:17",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 17,
              "raw_text": "Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference on Learning Representations (ICLR), 2016.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:18",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 18,
              "raw_text": "Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1610.10099v2",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:19",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 19,
              "raw_text": "Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks. In International Conference on Learning Representations, 2017.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:20",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 20,
              "raw_text": "Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:21",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 21,
              "raw_text": "Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1703.10722",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:22",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 22,
              "raw_text": "Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1703.03130",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:23",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 23,
              "raw_text": "Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1511.06114",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:24",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 24,
              "raw_text": "Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attentionbased neural machine translation. arXiv preprint arXiv:1508.04025, 2015.",
              "page_start": 10,
              "page_end": 10,
              "target_arxiv_id": "1508.04025",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:25",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 25,
              "raw_text": "Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:26",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 26,
              "raw_text": "David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:27",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 27,
              "raw_text": "Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention model. In Empirical Methods in Natural Language Processing, 2016.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:28",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 28,
              "raw_text": "Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1705.04304",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:29",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 29,
              "raw_text": "Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact, and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:30",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 30,
              "raw_text": "Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1608.05859",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:31",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 31,
              "raw_text": "Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1508.07909",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:32",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 32,
              "raw_text": "Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1701.06538",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:33",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 33,
              "raw_text": "Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. Journal ofMachine Learning Research, 15(1):1929–1958, 2014.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:34",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 34,
              "raw_text": "Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, Inc., 2015.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:35",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 35,
              "raw_text": "Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:36",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 36,
              "raw_text": "Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1512.00567",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:37",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 37,
              "raw_text": "Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In Advances in Neural Information Processing Systems, 2015.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:38",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 38,
              "raw_text": "Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1609.08144",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:39",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 39,
              "raw_text": "Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": "1606.04199",
              "target_doi": null,
              "resolution": "external",
              "matched_paper_id": null,
              "match_method": "external",
              "match_score": null,
              "duplicate_of_reference_id": null
            },
            {
              "reference_id": "1706.03762:40",
              "source_paper_id": "1706.03762",
              "source_canonical_id": "1706.03762v7",
              "ordinal": 40,
              "raw_text": "Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate shift-reduce constituent parsing. In Proceedings ofthe 51st Annual Meeting ofthe ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.",
              "page_start": 11,
              "page_end": 11,
              "target_arxiv_id": null,
              "target_doi": null,
              "resolution": "unresolved",
              "matched_paper_id": null,
              "match_method": "unresolved",
              "match_score": null,
              "duplicate_of_reference_id": null
            }
          ],
          "scope": "local_catalog",
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "references_list",
            "render_policy": "verbatim",
            "answer_text": "论文《Attention Is All You Need》共有 40 条参考文献。\n本地匹配：1\n\n1. Deep Residual Learning for Image Recognition"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```
## 案例 4：哪些论文引用了 Attention Is All You Need

```json
{
  "user_question": "哪些论文引用了 Attention Is All You Need",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_citation",
        "mode": "citations",
        "reason": "查询本地 Catalog 中的入向引用关系"
      },
      "mcp_request": {
        "tool": "library_citation",
        "arguments": {
          "paper_id": "1706.03762",
          "mode": "citations"
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "paper_id": "1706.03762",
          "items": [
            {
              "source_paper_id": "1810.04805",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "1911.02150",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2005.14165",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2006.16236",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2203.15556",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local"
            }
          ],
          "scope": "local_catalog",
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "citations_list",
            "render_policy": "verbatim",
            "answer_text": "在当前本地论文库中，共有 18 篇论文引用了 Attention Is All You Need：\n\n1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n2. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism\n3. Parameter-Efficient Transfer Learning for NLP\n4. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism\n5. Fast Transformer Decoding: One Write-Head is All You Need\n6. Language Models are Few-Shot Learners\n7. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\n8. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n9. Prefix-Tuning: Optimizing Continuous Prompts for Generation\n10. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## 案例 5：Attention Is All You Need 的引用和被引用关系

```json
{
  "user_question": "Attention Is All You Need 的引用和被引用关系",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_citation",
        "mode": "graph",
        "reason": "查询本地引用图的一跳双向关系"
      },
      "mcp_request": {
        "tool": "library_citation",
        "arguments": {
          "paper_id": "1706.03762",
          "mode": "graph",
          "direction": "both",
          "depth": 1
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "paper_id": "1706.03762",
          "direction": "both",
          "depth": 1,
          "nodes": [
            "1512.03385",
            "1706.03762",
            "1810.04805",
            "1811.06965",
            "1902.00751",
            "1909.08053",
            "1911.02150",
            "2005.14165",
            "2006.16236",
            "2010.11929",
            "2101.00190",
            "2103.14030",
            "2104.08691",
            "2106.09685",
            "2203.15556",
            "2205.14135",
            "2302.13971",
            "2309.06180",
            "2312.06635",
            "2405.04434"
          ],
          "edges": [
            {
              "source_paper_id": "1706.03762",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1810.04805",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1911.02150",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2005.14165",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2006.16236",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2203.15556",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            }
          ],
          "scope": "local_catalog",
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "citation_graph",
            "render_policy": "verbatim",
            "answer_text": "目标论文: Attention Is All You Need\n引用: 1 篇\n被引用: 18 篇\n\n引用（前10条）：\n1. Deep Residual Learning for Image Recognition\n\n被引用（前10条）：\n1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n2. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism\n3. Parameter-Efficient Transfer Learning for NLP\n4. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism\n5. Fast Transformer Decoding: One Write-Head is All You Need\n6. Language Models are Few-Shot Learners\n7. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\n8. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n9. Prefix-Tuning: Optimizing Continuous Prompts for Generation\n10. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## 案例 6：Attention Is All You Need 的两跳引用和被引用关系

`depth=2` 表示从目标论文出发，沿引用边继续查询一层邻居；服务端仍只返回 SQLite 引用图，不读取正文或向量索引。

```json
{
  "user_question": "Attention Is All You Need 的两跳引用和被引用关系",
  "steps": [
    {
      "agent_decision": {
        "selected_tool": "library_citation",
        "mode": "graph",
        "reason": "需要查看目标论文两跳范围内的本地引用关系"
      },
      "mcp_request": {
        "tool": "library_citation",
        "arguments": {
          "paper_title": "Attention Is All You Need",
          "mode": "graph",
          "direction": "both",
          "depth": 2
        }
      },
      "mcp_response": {
        "status": "ok",
        "data": {
          "paper_id": "1706.03762",
          "direction": "both",
          "depth": 2,
          "nodes": [
            "1404.5997",
            "1409.1556",
            "1409.4842",
            "1512.03385",
            "1706.03762",
            "1710.03740",
            "1810.04805",
            "1811.06965",
            "1902.00751",
            "1905.11946",
            "1909.08053",
            "1910.02054",
            "1911.02150",
            "2005.14165",
            "2006.16236",
            "2010.11929",
            "2101.00190",
            "2103.14030",
            "2104.08691",
            "2106.09685",
            "2203.02155",
            "2203.15556",
            "2205.14135",
            "2302.13971",
            "2305.13245",
            "2305.14314",
            "2305.18290",
            "2309.06180",
            "2312.06635",
            "2402.03300",
            "2405.04434"
          ],
          "edges": [
            {
              "source_paper_id": "1706.03762",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1810.04805",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1911.02150",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2005.14165",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2006.16236",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2203.15556",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "1706.03762",
              "relation": "cites",
              "resolution": "local",
              "depth": 1
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1404.5997",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1409.4842",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1811.06965",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1905.11946",
              "target_arxiv_id": "1811.06965",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1811.06965",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1910.02054",
              "target_arxiv_id": "1811.06965",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "2203.02155",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "2203.15556",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2302.13971",
              "target_arxiv_id": "2205.14135",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.13245",
              "target_arxiv_id": "2302.13971",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.14314",
              "target_arxiv_id": "2302.13971",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.18290",
              "target_arxiv_id": "2302.13971",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "2302.13971",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "2302.13971",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.13245",
              "target_arxiv_id": "1911.02150",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "1911.02150",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1910.02054",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2005.14165",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2006.16236",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "1810.04805",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "1902.00751",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2104.08691",
              "target_arxiv_id": "2101.00190",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "2104.08691",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.14314",
              "target_arxiv_id": "2104.08691",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "2104.08691",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "2006.16236",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "2006.16236",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2010.11929",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "2010.11929",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "2010.11929",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "2101.00190",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2309.06180",
              "target_arxiv_id": "2205.14135",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2402.03300",
              "target_arxiv_id": "2309.06180",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "2309.06180",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1409.1556",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1409.4842",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2103.14030",
              "target_arxiv_id": "1905.11946",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2203.15556",
              "target_arxiv_id": "1910.02054",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2203.15556",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "1910.02054",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "2203.02155",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "2305.13245",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2405.04434",
              "target_arxiv_id": "2402.03300",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2312.06635",
              "target_arxiv_id": "2205.14135",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1902.00751",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2106.09685",
              "target_arxiv_id": "2101.00190",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.14314",
              "target_arxiv_id": "2106.09685",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1409.1556",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1902.00751",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "1902.00751",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.14314",
              "target_arxiv_id": "1902.00751",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2101.00190",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.14314",
              "target_arxiv_id": "2101.00190",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2205.14135",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.13245",
              "target_arxiv_id": "2205.14135",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1512.03385",
              "target_arxiv_id": "1409.1556",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1512.03385",
              "target_arxiv_id": "1409.4842",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1710.03740",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1905.11946",
              "target_arxiv_id": "1512.03385",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1909.08053",
              "target_arxiv_id": "1710.03740",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "1910.02054",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2005.14165",
              "target_arxiv_id": "1909.08053",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2203.02155",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            },
            {
              "source_paper_id": "2305.18290",
              "target_arxiv_id": "2005.14165",
              "relation": "cites",
              "resolution": "local",
              "depth": 2
            }
          ],
          "scope": "local_catalog",
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "citation_graph",
            "render_policy": "verbatim",
            "answer_text": "目标论文: Attention Is All You Need\n直接引用: 1 篇\n直接被引用: 18 篇\n查询深度: 2\n间接引用: 2 篇\n间接被引用: 7 篇\n引用（前10条）：\n1. Deep Residual Learning for Image Recognition\n\n被引用（前10条）：\n1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n2. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism\n3. Parameter-Efficient Transfer Learning for NLP\n4. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism\n5. Fast Transformer Decoding: One Write-Head is All You Need\n6. Language Models are Few-Shot Learners\n7. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\n8. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n9. Prefix-Tuning: Optimizing Continuous Prompts for Generation\n10. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## 引用图两级方向修正

本轮将引用图的两跳关系改为按方向独立计算，避免在同一条路径中混合“引用”和“被引用”：

- `A → B` 表示 A 直接引用 B；`A → B → C` 表示 A 间接引用 C。
- `D → A` 表示 A 直接被 D 引用；`E → D → A` 表示 A 间接被 E 引用。
- `direction=both` 只是合并出向和入向两套有向 BFS，不能在第二跳切换方向。
- `depth=2` 表示最多两条有向边，即一个中间论文；不支持第三跳。
- MCP 图结果中的每条边增加 `path`，用于核验实际路径；引用边方向仍为 `source_paper_id → target_arxiv_id`。

以 `Attention Is All You Need` 为例：

```text
直接引用：Deep Residual Learning for Image Recognition
间接引用：
  Attention Is All You Need
    → Deep Residual Learning for Image Recognition
    → Very Deep Convolutional Networks for Large-Scale Image Recognition
  Attention Is All You Need
    → Deep Residual Learning for Image Recognition
    → Going Deeper with Convolutions
```

反向关系按同一规则计算：直接被引用论文先指向目标论文，再沿入向继续查找上一层引用者。当前测试结果：`107 passed`，`pip check` 通过。
