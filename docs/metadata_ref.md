# 元数据与引用 MCP 真实调用记录

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
