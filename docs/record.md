# 混合检索 MCP 测试记录

以下结果由真实 stdio MCP 服务 `paper_rag.mcp.server` 返回。

```json
{
  "generated_at": "2026-10-05T15:31:51.589437+00:00",
  "transport": "stdio",
  "server": "paper_rag.mcp.server",
  "index_status": {
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
  },
  "cases": [
    {
      "name": "fact",
      "query": "What is the main advantage of grouped-query attention over multi-head attention?",
      "task": "fact",
      "response": {
        "status": "ok",
        "data": {
          "query": "What is the main advantage of grouped-query attention over multi-head attention?",
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
          "items": [
            {
              "chunk_id": "ade6d1279ca8557c7ceefe38",
              "paper_id": "2305.13245",
              "canonical_id": "2305.13245v3",
              "ordinal": 22,
              "region": "content",
              "chapter_number": "5",
              "chapter_title": "Conclusion",
              "section_path": [
                "content",
                "5 Conclusion"
              ],
              "section_label": "5 Conclusion",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "126d20074ce576662db4ba0bb99dab2550d0e88f68fd74d3f90fa7bbc5ea87e2",
              "source_blocks": [
                {
                  "index": 56,
                  "page_idx": 4,
                  "bbox": [
                    112,
                    111,
                    490,
                    288
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "59647da5b3b118e5265ea453eb76c62da26cbfdcd0562524c30bafc5c9074d01",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Language models are expensive for inference primarily due to the memory bandwidth overhead from loading keys and values. Multi-query attention reduces this overhead at the cost of decreased model capacity and quality. We propose to convert multi-head attention models to multi-query models with a small fraction of original pre-training compute. Moreover, we introduce grouped-query attention, an interpolation of multi-query and multi-head attention that achieves quality close to multi-head at comparable speed to multi-query attention.",
              "score": 0.03200204813108039,
              "semantic_score": 0.6334823369979858,
              "lexical_rank": 2,
              "semantic_rank": 3,
              "rrf_score": 0.03200204813108039,
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "ceb1b63e04d2b9038323ef62",
              "paper_id": "2305.13245",
              "canonical_id": "2305.13245v3",
              "ordinal": 6,
              "region": "content",
              "chapter_number": "2.1",
              "chapter_title": "Uptraining",
              "section_path": [
                "content",
                "2 Method",
                "2.1 Uptraining"
              ],
              "section_label": "2.1 Uptraining",
              "type": "image",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "4a34e347bb4bf6871c7b4d82fe9fddb552825653be1469b2666efc1e3ae19d61",
              "source_blocks": [
                {
                  "index": 19,
                  "page_idx": 1,
                  "bbox": [
                    166,
                    89,
                    845,
                    244
                  ],
                  "type": "image",
                  "text_format": null,
                  "context_before": "The converted checkpoint is then pre-trained for a small proportion α of its original training steps on the same pre-training recipe.",
                  "context_after": ""
                }
              ],
              "asset_refs": [
                "images/f16486dc2e17b4fd7b67c63cabe29c74414f66d4732d7dfd5183698df2f41efa.jpg"
              ],
              "retrieval_text_hash": "4821f74f99687a3b7469b983689f820e285f490fc198cfc8efe17fb94d09c698",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Figure 2: Overview of grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.",
              "score": 0.03200204813108039,
              "semantic_score": 0.6346269249916077,
              "lexical_rank": 3,
              "semantic_rank": 2,
              "rrf_score": 0.03200204813108039,
              "page_start_display": 2,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "918f5c6c6b608f283cc9b52c",
              "paper_id": "2305.13245",
              "canonical_id": "2305.13245v3",
              "ordinal": 7,
              "region": "content",
              "chapter_number": "2.2",
              "chapter_title": "Grouped-query attention",
              "section_path": [
                "content",
                "2 Method",
                "2.2 Grouped-query attention"
              ],
              "section_label": "2.2 Grouped-query attention",
              "type": "text",
              "page_start": 1,
              "page_end": 1,
              "content_hash": "d98054406f4084e79beef188844065e58b8f02b569eabab7d6f5c60da2325e68",
              "source_blocks": [
                {
                  "index": 22,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    387,
                    489,
                    581
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 23,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    581,
                    490,
                    806
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 24,
                  "page_idx": 1,
                  "bbox": [
                    112,
                    806,
                    490,
                    920
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 26,
                  "page_idx": 1,
                  "bbox": [
                    507,
                    370,
                    884,
                    436
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "2db665137ca541aee3e5ccf4cd09ea9e843c064e810c7e91a606ec74ae32c4e0",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. Figure 2 shows a comparison of grouped-query attention and multihead/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by meanpooling all the original heads within that group.\n\nAn intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity.",
              "score": 0.031544957774465976,
              "semantic_score": 0.6765561103820801,
              "lexical_rank": 6,
              "semantic_rank": 1,
              "rrf_score": 0.031544957774465976,
              "page_start_display": 2,
              "page_end_display": 2,
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
              "chunk_id": "07e73ff2789e524f88a393a8",
              "paper_id": "2305.13245",
              "canonical_id": "2305.13245v3",
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
              "content_hash": "4742918eeeaee34fd0a99da1344f9e032fc87a3e60f0e4c6fa84e0f2ad215c5a",
              "source_blocks": [
                {
                  "index": 4,
                  "page_idx": 0,
                  "bbox": [
                    141,
                    275,
                    460,
                    502
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "87abdf4758f35825a038a92319c11af5edc27fd0f5c13f6bf3c60681347159d4",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.",
              "score": 0.031009615384615385,
              "semantic_score": 0.5715029239654541,
              "lexical_rank": 4,
              "semantic_rank": 5,
              "rrf_score": 0.031009615384615385,
              "page_start_display": 1,
              "page_end_display": 1,
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "66fe79e0315ad637eeaeae25",
              "paper_id": "2405.04434",
              "canonical_id": "2405.04434v5",
              "ordinal": 20,
              "region": "content",
              "chapter_number": "2.1.1",
              "chapter_title": "Preliminaries: Standard Multi-Head Attention",
              "section_path": [
                "content",
                "2. Architecture",
                "2.1. Multi-Head Latent Attention: Boosting Inference Efficiency",
                "2.1.1. Preliminaries: Standard Multi-Head Attention"
              ],
              "section_label": "2.1.1. Preliminaries: Standard Multi-Head Attention",
              "type": "image",
              "page_start": 6,
              "page_end": 6,
              "content_hash": "31625e6ecbb6ad5124bf9da1c3aef430722d2d5760e2bc3ccdcc7c37391c0a05",
              "source_blocks": [
                {
                  "index": 39,
                  "page_idx": 6,
                  "bbox": [
                    119,
                    111,
                    877,
                    246
                  ],
                  "type": "image",
                  "text_format": null,
                  "context_before": "",
                  "context_after": "Then, $\\mathbf { q } _ { t } , \\mathbf { k } _ { t } , \\mathbf { v } _ { t }$ will be sliced into $n _ { h }$ heads for the multi-head attention computation:"
                }
              ],
              "asset_refs": [
                "images/b35529d551f082ce4b28f6ae10da428fbeca0a4394b31d52656a841f9ea96107.jpg"
              ],
              "retrieval_text_hash": "24dd8517a6a6aa6b8cd8149e721e79b7fbed17413bad32634091c120531e5c68",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Figure 3 | Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference.",
              "score": 0.030679156908665108,
              "semantic_score": 0.5391427278518677,
              "lexical_rank": 1,
              "semantic_rank": 10,
              "rrf_score": 0.030679156908665108,
              "page_start_display": 7,
              "page_end_display": 7,
              "evidence_role": "direct",
              "source_id": "S5"
            }
          ],
          "count": 5,
          "context_text": "[S1] 2305.13245v3 p.5: Language models are expensive for inference primarily due to the memory bandwidth overhead from loading keys and values. Multi-query attention reduces this overhead at the cost of decreased model capacity and quality. We propose to convert multi-head attention models to multi-query models with a small fraction of original pre-training compute. Moreover, we introduce grouped-query attention, an interpolation of multi-query and multi-head attention that achieves quality close to multi-head at comparable speed to multi-query attention.\n\n[S2] 2305.13245v3 p.2: Figure 2: Overview of grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.\n\n[S3] 2305.13245v3 p.2: Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. Figure 2 shows a comparison of grouped-query attention and multihead/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by meanpooling all the original heads within that group.\n\nAn intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity.\n\n[S4] 2305.13245v3 p.1: Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.\n\n[S5] 2405.04434v5 p.7: Figure 3 | Simplified illustration of Multi-Head Attention (MHA), Grouped-Query Attention (GQA), Multi-Query Attention (MQA), and Multi-head Latent Attention (MLA). Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference.",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"grouped query attention\" OR \"multi head attention\" OR \"main advantage\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "grouped query attention",
              "multi head attention",
              "main advantage"
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
      }
    },
    {
      "name": "reason",
      "query": "Why does PagedAttention reduce memory fragmentation when serving large language models?",
      "task": "reason",
      "response": {
        "status": "ok",
        "data": {
          "query": "Why does PagedAttention reduce memory fragmentation when serving large language models?",
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
            }
          ],
          "items": [
            {
              "chunk_id": "cfc7f7426551e739e6436b6f",
              "paper_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "ordinal": 44,
              "region": "content",
              "chapter_number": "4.3",
              "chapter_title": "Decoding with PagedAttention and vLLM",
              "section_path": [
                "content",
                "4 Method",
                "4.3 Decoding with PagedAttention and vLLM"
              ],
              "section_label": "4.3 Decoding with PagedAttention and vLLM",
              "type": "image",
              "page_start": 5,
              "page_end": 5,
              "content_hash": "4e28e29c47d62dc9f6cd7f5bd335345f5851cc5003c994fa91df60deae11f230",
              "source_blocks": [
                {
                  "index": 84,
                  "page_idx": 5,
                  "bbox": [
                    516,
                    90,
                    911,
                    210
                  ],
                  "type": "image",
                  "text_format": null,
                  "context_before": "oring multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the efect of block size in §7.2.",
                  "context_after": "Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can"
                }
              ],
              "asset_refs": [
                "images/6516761dfc198bdf927f8a353afd82efbde27f1a3e9e9a32c549892e49c63ac7.jpg"
              ],
              "retrieval_text_hash": "c7c81e996cbab4a84ac4c5f168606dbf67a06041bc112e918dd0463b3fbb12c2",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "Figure 7. Storing the KV cache of two requests at the same time in vLLM.",
              "score": 0.03177805800756621,
              "semantic_score": 0.6616333723068237,
              "lexical_rank": 5,
              "semantic_rank": 1,
              "rrf_score": 0.03177805800756621,
              "page_start_display": 6,
              "page_end_display": 6,
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "8d79baeaf210eefb0147d776",
              "paper_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "ordinal": 43,
              "region": "content",
              "chapter_number": "4.3",
              "chapter_title": "Decoding with PagedAttention and vLLM",
              "section_path": [
                "content",
                "4 Method",
                "4.3 Decoding with PagedAttention and vLLM"
              ],
              "section_label": "4.3 Decoding with PagedAttention and vLLM",
              "type": "text",
              "text": "ion kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks. Storing multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the efect of block size in §7.2.",
              "retrieval_text": "content\n4 Method\n4.3 Decoding with PagedAttention and vLLM\nion kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks. Storing multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the efect of block size in §7.2.",
              "page_start": 5,
              "page_end": 5,
              "page_start_display": 6,
              "page_end_display": 6,
              "source_blocks": [
                {
                  "index": 82,
                  "page_idx": 5,
                  "bbox": [
                    81,
                    454,
                    483,
                    830
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 83,
                  "page_idx": 5,
                  "bbox": [
                    81,
                    830,
                    483,
                    907
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "6329a7313d0952786aadbf38d662bec2be03f1e422ba9643c489b0254067a65a",
              "retrieval_text_hash": "658e379d38946e568eac6f429cdebec838c74eeb53c4617d6599561fa4daef66",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "evidence_role": "context",
              "source_id": "S2"
            },
            {
              "chunk_id": "cd9d334fa60f2fe3273b914f",
              "paper_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "ordinal": 45,
              "region": "content",
              "chapter_number": "4.3",
              "chapter_title": "Decoding with PagedAttention and vLLM",
              "section_path": [
                "content",
                "4 Method",
                "4.3 Decoding with PagedAttention and vLLM"
              ],
              "section_label": "4.3 Decoding with PagedAttention and vLLM",
              "type": "text",
              "text": "Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can efectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests. In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to diferent physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be efectively utilized by both sequences.",
              "retrieval_text": "content\n4 Method\n4.3 Decoding with PagedAttention and vLLM\nAgain, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can efectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests. In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to diferent physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be efectively utilized by both sequences.",
              "page_start": 5,
              "page_end": 5,
              "page_start_display": 6,
              "page_end_display": 6,
              "source_blocks": [
                {
                  "index": 86,
                  "page_idx": 5,
                  "bbox": [
                    511,
                    436,
                    916,
                    694
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "906de5321a7fd99136be817759a566cd8c157e115c3fd9236857499595be7823",
              "retrieval_text_hash": "9ce6d7fda00c3ba654b859fd3325e133f0bc26f33a4c937d5e8fd9884e78658e",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "evidence_role": "context",
              "source_id": "S3"
            },
            {
              "chunk_id": "9df736b4b1722dc1427823d8",
              "paper_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "ordinal": 42,
              "region": "content",
              "chapter_number": "4.3",
              "chapter_title": "Decoding with PagedAttention and vLLM",
              "section_path": [
                "content",
                "4 Method",
                "4.3 Decoding with PagedAttention and vLLM"
              ],
              "section_label": "4.3 Decoding with PagedAttention and vLLM",
              "type": "text",
              "text": "ration phase. ○2 In the first autoregressive decoding step, vLLM generates the new token with the PagedAttention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s #filled record is updated. ○3 At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table.\n\nGlobally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in §4.5), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokens of the current iteration (i.e., all tokens for prompt phase requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the PagedAttention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks.",
              "retrieval_text": "content\n4 Method\n4.3 Decoding with PagedAttention and vLLM\nration phase. ○2 In the first autoregressive decoding step, vLLM generates the new token with the PagedAttention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s #filled record is updated. ○3 At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table.\n\nGlobally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in §4.5), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokens of the current iteration (i.e., all tokens for prompt phase requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the PagedAttention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks.",
              "page_start": 5,
              "page_end": 5,
              "page_start_display": 6,
              "page_end_display": 6,
              "source_blocks": [
                {
                  "index": 82,
                  "page_idx": 5,
                  "bbox": [
                    81,
                    454,
                    483,
                    830
                  ],
                  "type": "text",
                  "text_format": null
                },
                {
                  "index": 83,
                  "page_idx": 5,
                  "bbox": [
                    81,
                    830,
                    483,
                    907
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "431103c862608e519c0569f8c43d6f4feb2f4da0e16da0c96ccf5d2517945974",
              "retrieval_text_hash": "211ff5fbe8e0df6b08505fcc6cf9a2aef8679b49d5f7db6a13fe4744681b3da5",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "evidence_role": "context",
              "source_id": "S4"
            },
            {
              "chunk_id": "7532055ab6dedd850a6e9772",
              "paper_id": "2309.06180",
              "canonical_id": "2309.06180v1",
              "ordinal": 34,
              "region": "content",
              "chapter_number": "4.1",
              "chapter_title": "PagedAttention",
              "section_path": [
                "content",
                "4 Method",
                "4.1 PagedAttention"
              ],
              "section_label": "4.1 PagedAttention",
              "type": "text",
              "page_start": 4,
              "page_end": 4,
              "content_hash": "d30edec00efb75642deec9c91f70c4675d90db5e3287ad1a8ec5579983d3869f",
              "source_blocks": [
                {
                  "index": 67,
                  "page_idx": 4,
                  "bbox": [
                    81,
                    700,
                    485,
                    821
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "retrieval_text_hash": "c63b10ae6073abae95270cd1f59714c70e8227408bc6300a434617bbf7786ab6",
              "chunk_rule_version": "content-list-regions-v5-1200-table-text",
              "text": "To address the memory challenges in §3, we introduce PagedAttention, an attention algorithm inspired by the classic idea of paging [25] in operating systems. Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space. Specifically, PagedAttention partitions the KV cache of each sequence into KV blocks. Each block contains the key and value vectors for a fixed number oftokens,<sup>1</sup> which we denote as KV block size (�). Denote the key block $K _ { j } = ( k _ { ( j - 1 ) B + 1 } , \\ldots , k _ { j B } )$ and value block $V _ { j } = \\big ( \\boldsymbol { v } _ { ( j - 1 ) B + 1 } , \\ldots , \\boldsymbol { v } _ { j B } \\big )$ . The attention computation in Eq. 4 can be transformed into the following blockwise computation:",
              "score": 0.031099324975891997,
              "semantic_score": 0.608878493309021,
              "lexical_rank": 1,
              "semantic_rank": 8,
              "rrf_score": 0.031099324975891997,
              "page_start_display": 5,
              "page_end_display": 5,
              "evidence_role": "direct",
              "source_id": "S5"
            }
          ],
          "count": 5,
          "context_text": "[S1] 2309.06180v1 p.6: Figure 7. Storing the KV cache of two requests at the same time in vLLM.\n\n[S2] 2309.06180v1 p.6: ion kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks. Storing multiple tokens within a KV block (block size > 1) enables the PagedAttention kernel to process the KV cache across more positions in parallel, thus increasing the hardware utilization and reducing latency. However, a larger block size also increases memory fragmentation. We study the efect of block size in §7.2.\n\n[S3] 2309.06180v1 p.6: Again, vLLM dynamically assigns new physical blocks to logical blocks as more tokens and their KV cache are generated. As all the blocks are filled from left to right and a new physical block is only allocated when all previous blocks are full, vLLM limits all the memory wastes for a request within one block, so it can efectively utilize all the memory, as shown in Fig. 2. This allows more requests to fit into memory for batching—hence improving the throughput. Once a request finishes its generation, its KV blocks can be freed to store the KV cache of other requests. In Fig. 7, we show an example of vLLM managing the memory for two sequences. The logical blocks of the two sequences are mapped to diferent physical blocks within the space reserved by the block engine in GPU workers. The neighboring logical blocks of both sequences do not need to be contiguous in physical GPU memory and the space of physical blocks can be efectively utilized by both sequences.\n\n[S4] 2309.06180v1 p.6: ration phase. ○2 In the first autoregressive decoding step, vLLM generates the new token with the PagedAttention algorithm on physical blocks 7 and 1. Since one slot remains available in the last logical block, the newly generated KV cache is stored there, and the block table’s #filled record is updated. ○3 At the second decoding step, as the last logical block is full, vLLM stores the newly generated KV cache in a new logical block; vLLM allocates a new physical block (physical block 3) for it and stores this mapping in the block table.\n\nGlobally, for each decoding iteration, vLLM first selects a set of candidate sequences for batching (more in §4.5), and allocates the physical blocks for the newly required logical blocks. Then, vLLM concatenates all the input tokens of the current iteration (i.e., all tokens for prompt phase requests and the latest tokens for generation phase requests) as one sequence and feeds it into the LLM. During LLM’s computation, vLLM uses the PagedAttention kernel to access the previous KV cache stored in the form of logical KV blocks and saves the newly generated KV cache into the physical KV blocks.\n\n[S5] 2309.06180v1 p.5: To address the memory challenges in §3, we introduce PagedAttention, an attention algorithm inspired by the classic idea of paging [25] in operating systems. Unlike the traditional attention algorithms, PagedAttention allows storing continuous keys and values in non-contiguous memory space. Specifically, PagedAttention partitions the KV cache of each sequence into KV blocks. Each block contains the key and value vectors for a fixed number oftokens,<sup>1</sup> which we denote as KV block size (�). Denote the key block $K _ { j } = ( k _ { ( j - 1 ) B + 1 } , \\ldots , k _ { j B } )$ and value block $V _ { j } = \\big ( \\boldsymbol { v } _ { ( j - 1 ) B + 1 } , \\ldots , \\boldsymbol { v } _ { j B } \\big )$ . The attention computation in Eq. 4 can be transformed into the following blockwise computation:",
          "truncated": false,
          "retrieval_debug": {
            "lexical_query": "\"pagedattention\" OR \"reduce memory fragmentation\" OR \"serving large language models\"",
            "translation_used": false,
            "translation_provider": null,
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "pagedattention",
              "reduce memory fragmentation",
              "serving large language models"
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
      }
    },
    {
      "name": "summary",
      "query": "Summarize the core contributions of LoRA.",
      "task": "summary",
      "response": {
        "status": "not_found",
        "data": {
          "query": "Summarize the core contributions of LoRA.",
          "task": "summary",
          "papers": [],
          "items": [],
          "count": 0,
          "context_text": "",
          "truncated": false,
          "retrieval_debug": {},
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
      }
    },
    {
      "name": "comparison",
      "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
      "task": "comparison",
      "response": {
        "status": "ok",
        "data": {
          "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
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
              "paper_id": "2305.14314",
              "base_id": "2305.14314",
              "canonical_id": "2305.14314v1",
              "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
              "authors": [
                "Tim Dettmers",
                "Artidoro Pagnoni",
                "Ari Holtzman",
                "Luke Zettlemoyer"
              ],
              "abstract": "We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimziers to manage memory spikes. We use QLoRA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models). Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. We release all of our models and code, including CUDA kernels for 4-bit training.",
              "categories": [
                "cs.LG"
              ],
              "published_at": "2023-05-23T17:50:33Z",
              "updated_at": "2023-05-23T17:50:33Z",
              "abs_url": "https://arxiv.org/abs/2305.14314v1",
              "pdf_url": "https://arxiv.org/pdf/2305.14314v1.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2305.14314\\mineru"
                }
              }
            },
            {
              "paper_id": "2106.09685",
              "base_id": "2106.09685",
              "canonical_id": "2106.09685v2",
              "title": "LoRA: Low-Rank Adaptation of Large Language Models",
              "authors": [
                "Edward J. Hu",
                "Yelong Shen",
                "Phillip Wallis",
                "Zeyuan Allen-Zhu",
                "Yuanzhi Li",
                "Shean Wang",
                "Lu Wang",
                "Weizhu Chen"
              ],
              "abstract": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA.",
              "categories": [
                "cs.CL",
                "cs.AI",
                "cs.LG"
              ],
              "published_at": "2021-06-17T17:37:18Z",
              "updated_at": "2021-10-16T18:40:34Z",
              "abs_url": "https://arxiv.org/abs/2106.09685v2",
              "pdf_url": "https://arxiv.org/pdf/2106.09685v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2106.09685\\mineru"
                }
              }
            },
            {
              "paper_id": "1902.00751",
              "base_id": "1902.00751",
              "canonical_id": "1902.00751v2",
              "title": "Parameter-Efficient Transfer Learning for NLP",
              "authors": [
                "Neil Houlsby",
                "Andrei Giurgiu",
                "Stanislaw Jastrzebski",
                "Bruna Morrone",
                "Quentin de Laroussilhe",
                "Andrea Gesmundo",
                "Mona Attariyan",
                "Sylvain Gelly"
              ],
              "abstract": "Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter's effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.",
              "categories": [
                "cs.LG",
                "cs.CL",
                "stat.ML"
              ],
              "published_at": "2019-02-02T16:29:47Z",
              "updated_at": "2019-06-13T17:48:30Z",
              "abs_url": "https://arxiv.org/abs/1902.00751v2",
              "pdf_url": "https://arxiv.org/pdf/1902.00751v2.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\1902.00751\\mineru"
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
            },
            {
              "paper_id": "2005.14165",
              "base_id": "2005.14165",
              "canonical_id": "2005.14165v4",
              "title": "Language Models are Few-Shot Learners",
              "authors": [
                "Tom B. Brown",
                "Benjamin Mann",
                "Nick Ryder",
                "Melanie Subbiah",
                "Jared Kaplan",
                "Prafulla Dhariwal",
                "Arvind Neelakantan",
                "Pranav Shyam",
                "Girish Sastry",
                "Amanda Askell",
                "Sandhini Agarwal",
                "Ariel Herbert-Voss",
                "Gretchen Krueger",
                "Tom Henighan",
                "Rewon Child",
                "Aditya Ramesh",
                "Daniel M. Ziegler",
                "Jeffrey Wu",
                "Clemens Winter",
                "Christopher Hesse",
                "Mark Chen",
                "Eric Sigler",
                "Mateusz Litwin",
                "Scott Gray",
                "Benjamin Chess",
                "Jack Clark",
                "Christopher Berner",
                "Sam McCandlish",
                "Alec Radford",
                "Ilya Sutskever",
                "Dario Amodei"
              ],
              "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.",
              "categories": [
                "cs.CL"
              ],
              "published_at": "2020-05-28T17:29:03Z",
              "updated_at": "2020-07-22T19:47:17Z",
              "abs_url": "https://arxiv.org/abs/2005.14165v4",
              "pdf_url": "https://arxiv.org/pdf/2005.14165v4.pdf",
              "doi": null,
              "metadata_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\metadata.json",
              "pdf_path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\paper.pdf",
              "mineru_dir": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\mineru",
              "state": "ingested",
              "assets": {
                "pdf": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\paper.pdf"
                },
                "metadata": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\metadata.json"
                },
                "mineru": {
                  "present": true,
                  "path": "E:\\Pythonproject\\paper_RAG\\data\\sources\\arxiv\\2005.14165\\mineru"
                }
              }
            }
          ],
          "items": [
            {
              "chunk_id": "9c2b240ca82b699d6fb9d9c2",
              "paper_id": "2305.14314",
              "canonical_id": "2305.14314v1",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLORA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g.",
              "retrieval_text": "abstract\nWe present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLORA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g.",
              "page_start": 0,
              "page_end": 0,
              "page_start_display": 1,
              "page_end_display": 1,
              "source_blocks": [
                {
                  "index": 7,
                  "page_idx": 0,
                  "bbox": [
                    228,
                    356,
                    767,
                    689
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "2f13f8b230f8114fcfe734863e4fc6f43ceb312ab35c59bbdfb1f82e0c04b731",
              "retrieval_text_hash": "2ff3cd9173ee488319073e62b9bdec2477451d5e08ecafcf5098df1600ec9d08",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
              "evidence_role": "direct",
              "source_id": "S1"
            },
            {
              "chunk_id": "b94582d3bc30b72aa5c31d70",
              "paper_id": "2106.09685",
              "canonical_id": "2106.09685v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.",
              "retrieval_text": "abstract\nAn important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.",
              "page_start": 0,
              "page_end": 0,
              "page_start_display": 1,
              "page_end_display": 1,
              "source_blocks": [
                {
                  "index": 4,
                  "page_idx": 0,
                  "bbox": [
                    228,
                    334,
                    767,
                    585
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "07c85ae95a72dad3d3d9b705dde91c6d7c80364c62b8df3615bf8715259dfc37",
              "retrieval_text_hash": "b7137d231559452c76441d15448af177a6c5046b5663fee3d87dba262a94b4d2",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
              "evidence_role": "direct",
              "source_id": "S2"
            },
            {
              "chunk_id": "790e80f2134e03847cd9441e",
              "paper_id": "1902.00751",
              "canonical_id": "1902.00751v2",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter’s effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.<sup>1</sup>",
              "retrieval_text": "abstract\nFine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter’s effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.<sup>1</sup>",
              "page_start": 0,
              "page_end": 0,
              "page_start_display": 1,
              "page_end_display": 1,
              "source_blocks": [
                {
                  "index": 3,
                  "page_idx": 0,
                  "bbox": [
                    117,
                    258,
                    444,
                    578
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "068a33a144b4415b35868d6a71a75f4093da1d062e68d27a45263554d46445d9",
              "retrieval_text_hash": "e1e411ef24d20977bc9d9e0ffc4f464500736dd7a41f06f1887de6632e550851",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
              "evidence_role": "direct",
              "source_id": "S3"
            },
            {
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
              "page_start_display": 1,
              "page_end_display": 1,
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
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
              "evidence_role": "direct",
              "source_id": "S4"
            },
            {
              "chunk_id": "599b1b013a32462b571862f0",
              "paper_id": "2005.14165",
              "canonical_id": "2005.14165v4",
              "ordinal": 0,
              "region": "abstract",
              "chapter_number": null,
              "chapter_title": null,
              "section_path": [
                "abstract"
              ],
              "section_label": "abstract",
              "type": "text",
              "text": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.",
              "retrieval_text": "abstract\nRecent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.",
              "page_start": 0,
              "page_end": 0,
              "page_start_display": 1,
              "page_end_display": 1,
              "source_blocks": [
                {
                  "index": 12,
                  "page_idx": 0,
                  "bbox": [
                    169,
                    545,
                    828,
                    810
                  ],
                  "type": "text",
                  "text_format": null
                }
              ],
              "asset_refs": [],
              "content_hash": "c9f02b44966638ebb37a6c19ee3a216bd2c9a1a9fe125604a848a22bd8b9442e",
              "retrieval_text_hash": "f2b2bbe6bc82d5f26818c4cbdf5369e9e15cf18478f81098112289cbf1d28df0",
              "score": null,
              "semantic_score": null,
              "lexical_rank": null,
              "semantic_rank": null,
              "rrf_score": null,
              "query": "Compare LoRA and QLoRA in terms of quantization and fine-tuning.",
              "evidence_role": "direct",
              "source_id": "S5"
            }
          ],
          "count": 5,
          "context_text": "[S1] 2305.14314v1 p.1: We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance. QLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA). Our best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU. QLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes. We use QLORA to finetune more than 1,000 models, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and model scales that would be infeasible to run with regular finetuning (e.g.\n\n[S2] 2106.09685v2 p.1: An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.\n\n[S3] 1902.00751v2 p.1: Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter’s effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task.<sup>1</sup>\n\n[S4] 2101.00190v1 p.1: Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. However, it modifies all the language model parameters and therefore necessitates storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were “virtual tokens”. We apply prefix-tuning to GPT-2 for table-to-text generation and to BART for summarization. We find that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topics unseen during training.\n\n[S5] 2005.14165v4 p.1: Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.",
          "truncated": false,
          "retrieval_debug": {},
          "presentation": {
            "template_version": "library-answer-v1",
            "answer_type": "rag_evidence",
            "render_policy": "compose",
            "answer_text": ""
          }
        },
        "warnings": [],
        "read_only": true
      }
    }
  ]
}
```

## Milvus 增量同步统计

本次增量更新完成后，Catalog、Milvus Collection 与 embedding cache 的 Chunk ID 已校验一致。

```json
{
  "operation": "milvus_incremental_sync",
  "status": "completed",
  "mode": "incremental",
  "catalog_chunk_count": 3241,
  "indexed_count": 3241,
  "embedding_model": "qwen3.7-text-embedding-flash",
  "embedding_dimensions": 1024,
  "sync_stats": {
    "reused": 1492,
    "added": 1395,
    "updated": 354,
    "deleted": 404,
    "failed": 0
  },
  "consistency_check": {
    "catalog_ids": 3241,
    "milvus_ids": 3241,
    "embedding_cache_ids": 3241,
    "missing_ids": 0,
    "extra_ids": 0
  }
}
```
