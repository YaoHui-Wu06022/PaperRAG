[
  {
    "user_question": "论文库中有哪些和注意力机制相关的论文",
    "agent_decision": {
      "selected_tool": "library_search"
    },
    "mcp_request": {
      "tool": "library_search",
      "arguments": {
        "query": "论文库中有哪些和注意力机制相关的论文",
        "limit": 10
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "论文库中有哪些和注意力机制相关的论文",
        "items": [
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
          }
        ],
        "count": 1,
        "query_debug": {
          "lexical_query": "\"attention mechanism\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "attention mechanism"
          ],
          "rewriter_error": null
        },
        "presentation": {
          "template_version": "library-answer-v1",
          "answer_type": "metadata_list",
          "render_policy": "verbatim",
          "answer_text": "共找到 1 篇论文：\n\n1. Attention Is All You Need\n   arXiv: 1706.03762\n   作者: Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、Llion Jones、Aidan N. Gomez、Lukasz Kaiser、Illia Polosukhin\n   年份: 2017\n   分类: cs.CL、cs.LG"
        }
      },
      "warnings": [],
      "read_only": true
    },
    "is_error": false
  },
  {
    "user_question": "Attention Is All You Need 参考了哪些本地论文",
    "agent_decision": {
      "selected_tool": "library_citation",
      "mode": "references"
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
    },
    "is_error": false
  },
  {
    "user_question": "GQA 的方法是什么",
    "agent_decision": {
      "selected_tool": "library_retrieve",
      "task": "auto"
    },
    "mcp_request": {
      "tool": "library_retrieve",
      "arguments": {
        "query": "GQA 的方法是什么",
        "task": "auto"
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "GQA 的方法是什么",
        "task": "fact",
        "mode": "hybrid",
        "routing": {
          "route_intent": "retrieve",
          "task": "fact",
          "provider": "jev",
          "fallback_used": false,
          "confidence": 0.65
        },
        "retrieval_debug": {
          "query_debug": {
            "lexical_query": "\"gqa method\" OR \"grouped query attention method\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "gqa method",
              "grouped query attention method"
            ],
            "rewriter_error": null
          },
          "candidate_discovery": {
            "metadata_count": 0,
            "chunk_count": 0,
            "entity_hits": {
              "GQA": [
                "2305.13245"
              ]
            },
            "fallback_used": false,
            "chunk_search_used": true,
            "chunk_match_count": 0,
            "selected_paper_ids": [
              "2305.13245"
            ],
            "lexical_query": "\"gqa method\" OR \"grouped query attention method\"",
            "candidate_match_source": {
              "2305.13245": "title_exact"
            },
            "title_exact_hit": [
              "2305.13245"
            ],
            "abstract_exact_hit": [
              "2305.13245"
            ],
            "chunk_exact_hit": [],
            "table_ref": null,
            "figure_ref": null
          },
          "evidence_fallback_used": false,
          "lexical_query": "\"gqa method\" OR \"grouped query attention method\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "gqa method",
            "grouped query attention method"
          ],
          "rewriter_error": null
        },
        "evidence": [
          {
            "source_id": "S1",
            "paper_id": "2305.13245",
            "chunk_id": "9d59efb82ff64eaa74f8b259",
            "text": "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. Figure 2 shows a comparison of grouped-query attention and multihead/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by meanpooling all the original heads within that group.",
            "type": "text",
            "section_path": [
              "content",
              "2 Method",
              "2.2 Grouped-query attention"
            ],
            "page_start": 1,
            "page_end": 1,
            "score": 0.018993442622950822
          },
          {
            "source_id": "S2",
            "paper_id": "2305.13245",
            "chunk_id": "8708de1bc8f2039b1d699b6a",
            "text": "This work contains two contributions for faster inference with large language models. First, we show that language model checkpoints with multihead attention (MHA) can be uptrained (Komatsuzaki et al., 2022) to use MQA with a small fraction of original training compute. This presents a cost-effective method to obtain fast multi-query as well as high-quality MHA checkpoints.\n\nSecond, we propose grouped-query attention (GQA), an interpolation between multi-head and multi-query attention with single key and value heads per subgroup of query heads. We show that uptrained GQA achieves quality close to multihead attention while being almost as fast as multiquery attention.",
            "type": "text",
            "section_path": [
              "content",
              "1 Introduction"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": 0.01847301587301587
          },
          {
            "source_id": "S3",
            "paper_id": "2305.13245",
            "chunk_id": "fd155127fc6950ab0a3c3129",
            "text": "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": 0.017975
          },
          {
            "source_id": "S4",
            "paper_id": "2305.13245",
            "chunk_id": "908f3ee9d59e32fc115fb9d7",
            "text": "We note that GQA is not applied to the encoder self-attention layers; encoder representations are computed in parallel, and memory bandwidth is therefore generally not the primary bottleneck.",
            "type": "text",
            "section_path": [
              "content",
              "2 Method",
              "2.2 Grouped-query attention"
            ],
            "page_start": 1,
            "page_end": 1,
            "score": 0.017634615384615384
          },
          {
            "source_id": "S5",
            "paper_id": "2305.13245",
            "chunk_id": "8510645f75b366fffd870e56",
            "text": "An intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.\n\nMoreover, larger models suffer relatively less from memory bandwidth overhead from attention, as the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension. Finally, standard sharding for large models replicates the single key and value head by the number of model partitions (Pope et al., 2022); GQA removes the waste from such partitioning. Therefore, we expect GQA to present a particularly good trade-off for larger models.",
            "type": "text",
            "section_path": [
              "content",
              "2 Method",
              "2.2 Grouped-query attention"
            ],
            "page_start": 1,
            "page_end": 1,
            "score": 0.01740151515151515
          },
          {
            "source_id": "S6",
            "paper_id": "2305.13245",
            "chunk_id": "49d4d97ce6ed92412e43fe1d",
            "text": "Figure 3 shows average performance over all datasets as a function of average inference time for MHA T5-Large and T5-XXL, and uptrained MQA and GQA-8 XXL models with uptraining proportion $\\alpha \\ : = \\ : 0 . 0 5$ . We see that a larger uptrained MQA model provides a favorable tradeoff relative to MHA models, with higher quality and faster inference than MHA-Large. Moreover, GQA achieves significant additional quality gains, achieving performance close to MHA-XXL with speed close to MQA. Table 1 contains full results for all datasets.",
            "type": "text",
            "section_path": [
              "content",
              "3 Experiments",
              "3.2 Main results"
            ],
            "page_start": 2,
            "page_end": 2,
            "score": 0.017305882352941178
          },
          {
            "source_id": "S7",
            "paper_id": "2305.13245",
            "chunk_id": "8b4ffb560a8f94b4cafd155f",
            "text": "Checkpoint conversion Figure 4 compares the performance of different methods for checkpoint conversion. Mean pooling appears to work best, followed by selecting a single head and then random initialization. Intuitively, results are ordered by the degree to which information is preserved from the pre-trained model.\n\nUptraining steps Figure 5 shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA. First, we note that GQA already achieves reasonable performance after conversion while MQA requires uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%.",
            "type": "text",
            "section_path": [
              "content",
              "3 Experiments",
              "3.3 Ablations"
            ],
            "page_start": 2,
            "page_end": 2,
            "score": 0.016742753623188406
          },
          {
            "source_id": "S8",
            "paper_id": "2305.13245",
            "chunk_id": "275560d9af68c3f9f9b99378",
            "text": "This paper focuses on ameliorating the memory bandwidth overhead from loading keys and values. This overhead is most important when generating longer sequences, for which quality is inherently difficult to evaluate. For summarization we employ Rouge score, which we know is a flawed evaluation that does not tell the whole story; for that reason, it is difficult to be certain our trade-offs are correct. Due to limited computation, we also do not compare our XXL GQA model to a comparitive model trained from scratch, so we do not know the relative performance of uptraining vs training from scratch. Finally, we evaluate the impact of uptraining and GQA only on encoder-decoder models. Recently, decoder-only models are extremely popular, and since these models do not have separate self-attention and cross-attention, we expect GQA to have a stronger advantage over MQA.",
            "type": "text",
            "section_path": [
              "content",
              "5 Conclusion",
              "Limitations"
            ],
            "page_start": 4,
            "page_end": 4,
            "score": 0.01668450704225352
          }
        ],
        "count": 8,
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
    "is_error": false
  },
  {
    "user_question": "为什么 GQA 能在 MQA 和 MHA 之间取得平衡",
    "agent_decision": {
      "selected_tool": "library_retrieve",
      "task": "auto"
    },
    "mcp_request": {
      "tool": "library_retrieve",
      "arguments": {
        "query": "为什么 GQA 能在 MQA 和 MHA 之间取得平衡",
        "task": "auto"
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "为什么 GQA 能在 MQA 和 MHA 之间取得平衡",
        "task": "reason",
        "mode": "hybrid",
        "routing": {
          "route_intent": "retrieve",
          "task": "reason",
          "provider": "jev",
          "fallback_used": false,
          "confidence": 0.72
        },
        "retrieval_debug": {
          "query_debug": {
            "lexical_query": "\"gqa strike balance between mqa mha\" OR \"gqa mqa mha balance\" OR \"grouped query attention multi query attention multi head attention balance\" OR \"gqa compromise mechanism between mqa mha\" OR \"grouped query attention trade off between mqa mha\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "why",
              "can",
              "a",
              "and"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "gqa strike balance between mqa mha",
              "gqa mqa mha balance",
              "grouped query attention multi query attention multi head attention balance",
              "gqa compromise mechanism between mqa mha",
              "grouped query attention trade off between mqa mha"
            ],
            "rewriter_error": null
          },
          "candidate_discovery": {
            "metadata_count": 0,
            "chunk_count": 0,
            "entity_hits": {
              "GQA": [
                "2305.13245"
              ],
              "MQA": [
                "2305.13245"
              ]
            },
            "fallback_used": false,
            "chunk_search_used": true,
            "chunk_match_count": 0,
            "selected_paper_ids": [
              "2305.13245"
            ],
            "lexical_query": "\"gqa strike balance between mqa mha\" OR \"gqa mqa mha balance\" OR \"grouped query attention multi query attention multi head attention balance\" OR \"gqa compromise mechanism between mqa mha\" OR \"grouped query attention trade off between mqa mha\"",
            "candidate_match_source": {
              "2305.13245": "title_exact"
            },
            "title_exact_hit": [
              "2305.13245"
            ],
            "abstract_exact_hit": [
              "2305.13245"
            ],
            "chunk_exact_hit": [],
            "table_ref": null,
            "figure_ref": null
          },
          "evidence_fallback_used": false,
          "lexical_query": "\"gqa strike balance between mqa mha\" OR \"gqa mqa mha balance\" OR \"grouped query attention multi query attention multi head attention balance\" OR \"gqa compromise mechanism between mqa mha\" OR \"grouped query attention trade off between mqa mha\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [
            "why",
            "can",
            "a",
            "and"
          ],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "gqa strike balance between mqa mha",
            "gqa mqa mha balance",
            "grouped query attention multi query attention multi head attention balance",
            "gqa compromise mechanism between mqa mha",
            "grouped query attention trade off between mqa mha"
          ],
          "rewriter_error": null
        },
        "evidence": [
          {
            "source_id": "S1",
            "paper_id": "2305.13245",
            "chunk_id": "49d4d97ce6ed92412e43fe1d",
            "text": "Figure 3 shows average performance over all datasets as a function of average inference time for MHA T5-Large and T5-XXL, and uptrained MQA and GQA-8 XXL models with uptraining proportion $\\alpha \\ : = \\ : 0 . 0 5$ . We see that a larger uptrained MQA model provides a favorable tradeoff relative to MHA models, with higher quality and faster inference than MHA-Large. Moreover, GQA achieves significant additional quality gains, achieving performance close to MHA-XXL with speed close to MQA. Table 1 contains full results for all datasets.",
            "type": "text",
            "section_path": [
              "content",
              "3 Experiments",
              "3.2 Main results"
            ],
            "page_start": 2,
            "page_end": 2,
            "source_chunk_ids": [
              "49d4d97ce6ed92412e43fe1d"
            ],
            "score": 0.018993442622950822
          },
          {
            "source_id": "S2",
            "paper_id": "2305.13245",
            "chunk_id": "9d59efb82ff64eaa74f8b259",
            "text": "Grouped-query attention divides query heads into G groups, each of which shares a single key head and value head. GQA-G refers to grouped-query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to number of heads, is equivalent to MHA. Figure 2 shows a comparison of grouped-query attention and multihead/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by meanpooling all the original heads within that group.\n\nAn intermediate number of groups leads to an interpolated model that is higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.\n\nMoreover, larger models suffer relatively less from memory bandwidth overhead from attention, as the KV-cache scales with model dimension while model FLOPs and parameters scale with the square of model dimension. Finally, standard sharding for large models replicates the single key and value head by the number of model partitions (Pope et al., 2022); GQA removes the waste from such partitioning. Therefore, we expect GQA to present a particularly good trade-off for larger models.",
            "type": "text",
            "section_path": [
              "content",
              "2 Method",
              "2.2 Grouped-query attention"
            ],
            "page_start": 1,
            "page_end": 1,
            "source_chunk_ids": [
              "9d59efb82ff64eaa74f8b259",
              "8510645f75b366fffd870e56"
            ],
            "score": 0.018729032258064514
          },
          {
            "source_id": "S3",
            "paper_id": "2305.13245",
            "chunk_id": "fd155127fc6950ab0a3c3129",
            "text": "Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "source_chunk_ids": [
              "fd155127fc6950ab0a3c3129"
            ],
            "score": 0.017975
          }
        ],
        "count": 3,
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
    "is_error": false
  },
  {
    "user_question": "请总结 LoRA 这篇论文的方法和主要贡献",
    "agent_decision": {
      "selected_tool": "library_retrieve",
      "task": "auto"
    },
    "mcp_request": {
      "tool": "library_retrieve",
      "arguments": {
        "query": "请总结 LoRA 这篇论文的方法和主要贡献",
        "task": "auto"
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "请总结 LoRA 这篇论文的方法和主要贡献",
        "task": "summary",
        "mode": "hybrid",
        "routing": {
          "route_intent": "retrieve",
          "task": "summary",
          "provider": "jev",
          "fallback_used": false,
          "confidence": 0.99
        },
        "retrieval_debug": {
          "query_debug": {
            "lexical_query": "\"main contributions lora methodology\" OR \"lora low rank adaptation method contributions\" OR \"contribution lora low rank adaptation methods\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "of",
              "paper"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "main contributions lora methodology",
              "lora low rank adaptation method contributions",
              "contribution lora low rank adaptation methods"
            ],
            "rewriter_error": null
          },
          "candidate_discovery": {
            "metadata_count": 0,
            "chunk_count": 0,
            "entity_hits": {
              "LoRA": [
                "2106.09685",
                "2305.14314"
              ]
            },
            "fallback_used": false,
            "chunk_search_used": true,
            "chunk_match_count": 0,
            "selected_paper_ids": [
              "2106.09685"
            ],
            "lexical_query": "\"main contributions lora methodology\" OR \"lora low rank adaptation method contributions\" OR \"contribution lora low rank adaptation methods\"",
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
          "evidence_fallback_used": false,
          "lexical_query": "\"main contributions lora methodology\" OR \"lora low rank adaptation method contributions\" OR \"contribution lora low rank adaptation methods\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [
            "of",
            "paper"
          ],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "main contributions lora methodology",
            "lora low rank adaptation method contributions",
            "contribution lora low rank adaptation methods"
          ],
          "rewriter_error": null
        },
        "evidence": [
          {
            "source_id": "S1",
            "paper_id": "2106.09685",
            "chunk_id": "7f475dda898c0404d731dd7d",
            "text": "We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.\n\nWe release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at https://github.com/microsoft/LoRA.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": 0.017525373134328358
          },
          {
            "source_id": "S2",
            "paper_id": "2106.09685",
            "chunk_id": "b64ef3dd069a4d2b933bdcaf",
            "text": "An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains.\n\nAs we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible.\n\nUsing GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive.\n\nWe propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.\n\nCompared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.\n\nLoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.\n\nWe also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": 0.016535714285714286
          },
          {
            "source_id": "S3",
            "paper_id": "2106.09685",
            "chunk_id": "0be95b55edde3ff291331d5a",
            "text": "We describe the simple design of LoRA and its practical benefits. The principles outlined here apply to any dense layers in deep learning models, though we only focus on certain weights in Transformer language models in our experiments as the motivating use case.",
            "type": "text",
            "section_path": [
              "content",
              "4 OUR METHOD"
            ],
            "page_start": 3,
            "page_end": 3,
            "score": 0.018743442622950822
          },
          {
            "source_id": "S4",
            "paper_id": "2106.09685",
            "chunk_id": "94f8e7070ff0758be918c3dc",
            "text": "Practical Benefits and Limitations. The most significant benefit comes from the reduction in memory and storage usage. For a large Transformer trained with Adam, we reduce that VRAM usage by up to $2 / 3 \\ \\mathrm { i f } \\ r \\ll \\ d _ { m o d e l }$ as we do not need to store the optimizer states for the frozen parameters. On GPT-3 175B, we reduce the VRAM consumption during training from 1.2TB to 350GB. With $r = 4$ and only the query and value projection matrices being adapted, the checkpoint size is reduced by roughly 10,000× (from 350GB to 35MB)<sup>4</sup>. This allows us to train with significantly fewer GPUs and avoid I/O bottlenecks. Another benefit is that we can switch between tasks while deployed at a much lower cost by only swapping the LoRA weights as opposed to all the parameters. This allows for the creation of many customized models that can be swapped in and out on the fly on machines that store the pre-trained weights in VRAM. We also observe a 25% speedup during training on GPT-3 175B compared to full fine-tuning<sup>5</sup> as we do not need to calculate the gradient for the vast majority of the parameters.",
            "type": "text",
            "section_path": [
              "content",
              "4 OUR METHOD",
              "4.2 APPLYING LORA TO TRANSFORMER"
            ],
            "page_start": 4,
            "page_end": 4,
            "score": 0.018601515151515154
          },
          {
            "source_id": "S5",
            "paper_id": "2106.09685",
            "chunk_id": "f5922348ccbfd3458f304739",
            "text": "Fine-tuning enormous language models is prohibitively expensive in terms of the hardware required and the storage/switching cost for hosting independent instances for different tasks. We propose LoRA, an efficient adaptation strategy that neither introduces inference latency nor reduces input sequence length while retaining high model quality. Importantly, it allows for quick task-switching when deployed as a service by sharing the vast majority of the model parameters. While we focused on Transformer language models, the proposed principles are generally applicable to any neural networks with dense layers.",
            "type": "text",
            "section_path": [
              "content",
              "8 CONCLUSION AND FUTURE WORK"
            ],
            "page_start": 11,
            "page_end": 11,
            "score": 0.018479032258064517
          },
          {
            "source_id": "S6",
            "paper_id": "2106.09685",
            "chunk_id": "5f4c7047eaf462c69b4887e1",
            "text": "• LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.\n\n• Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.\n\n• LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning. We provide an example in Appendix E.",
            "type": "text",
            "section_path": [
              "content",
              "1 INTRODUCTION"
            ],
            "page_start": 1,
            "page_end": 1,
            "score": 0.018223015873015874
          },
          {
            "source_id": "S7",
            "paper_id": "2106.09685",
            "chunk_id": "ef7a0b26d7f5b130a73f7344",
            "text": "There are many directions for future works. 1) LoRA can be combined with other efficient adaptation methods, potentially providing orthogonal improvement. 2) The mechanism behind fine-tuning or LoRA is far from clear – how are features learned during pre-training transformed to do well on downstream tasks? We believe that LoRA makes it more tractable to answer this than full finetuning. 3) We mostly depend on heuristics to select the weight matrices to apply LoRA to. Are there more principled ways to do it? 4) Finally, the rank-deficiency of ∆W suggests that W could be rank-deficient as well, which can also be a source of inspiration for future works.",
            "type": "text",
            "section_path": [
              "content",
              "8 CONCLUSION AND FUTURE WORK"
            ],
            "page_start": 11,
            "page_end": 11,
            "score": 0.017625000000000002
          },
          {
            "source_id": "S8",
            "paper_id": "2106.09685",
            "chunk_id": "b208fc392b96f50554e7efa2",
            "text": "LoRA also has its limitations. For example, it is not straightforward to batch inputs to different tasks with different A and B in a single forward pass, if one chooses to absorb A and B into W to eliminate additional inference latency. Though it is possible to not merge the weights and dynamically choose the LoRA modules to use for samples in a batch for scenarios where latency is not critical.",
            "type": "text",
            "section_path": [
              "content",
              "4 OUR METHOD",
              "4.2 APPLYING LORA TO TRANSFORMER"
            ],
            "page_start": 4,
            "page_end": 4,
            "score": 0.017592753623188406
          }
        ],
        "count": 8,
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
    "is_error": false
  },
  {
    "user_question": "比较 LoRA 和 QLoRA 的训练方法",
    "agent_decision": {
      "selected_tool": "library_retrieve",
      "task": "auto"
    },
    "mcp_request": {
      "tool": "library_retrieve",
      "arguments": {
        "query": "比较 LoRA 和 QLoRA 的训练方法",
        "task": "auto"
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "比较 LoRA 和 QLoRA 的训练方法",
        "task": "comparison",
        "mode": "hybrid",
        "routing": {
          "route_intent": "retrieve",
          "task": "comparison",
          "provider": "jev",
          "fallback_used": false,
          "confidence": 0.99
        },
        "retrieval_debug": {
          "query_debug": {
            "lexical_query": "\"comparison lora qlora training methods\"",
            "translation_used": true,
            "translation_provider": "tencent",
            "translation_fallback": false,
            "stopwords_removed": [
              "of"
            ],
            "rewriter_used": true,
            "rewriter_fallback": false,
            "core_terms": [
              "comparison lora qlora training methods"
            ],
            "rewriter_error": null
          },
          "candidate_discovery": {
            "metadata_count": 0,
            "chunk_count": 0,
            "entity_hits": {
              "LoRA": [
                "2106.09685",
                "2305.14314"
              ],
              "QLoRA": [
                "2305.14314"
              ]
            },
            "fallback_used": false,
            "chunk_search_used": true,
            "chunk_match_count": 0,
            "selected_paper_ids": [
              "2106.09685",
              "2305.14314"
            ],
            "lexical_query": "\"comparison lora qlora training methods\"",
            "candidate_match_source": {
              "2106.09685": "title_exact",
              "2305.14314": "title_exact"
            },
            "title_exact_hit": [
              "2106.09685",
              "2305.14314"
            ],
            "abstract_exact_hit": [
              "2106.09685",
              "2305.14314"
            ],
            "chunk_exact_hit": [],
            "table_ref": null,
            "figure_ref": null
          },
          "evidence_fallback_used": false,
          "lexical_query": "\"comparison lora qlora training methods\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [
            "of"
          ],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "comparison lora qlora training methods"
          ],
          "rewriter_error": null
        },
        "evidence": [
          {
            "source_id": "S1",
            "paper_id": "2106.09685",
            "chunk_id": "b64ef3dd069a4d2b933bdcaf",
            "text": "An important paradigm of natural language processing consists of large-scale pretraining on general domain data and adaptation to particular tasks or domains.\n\nAs we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible.\n\nUsing GPT-3 175B as an example – deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive.\n\nWe propose Low-Rank Adaptation, or LoRA, which freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.\n\nCompared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.\n\nLoRA performs on-par or better than finetuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.\n\nWe also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": null
          },
          {
            "source_id": "S2",
            "paper_id": "2106.09685",
            "chunk_id": "5f4c7047eaf462c69b4887e1",
            "text": "• LoRA makes training more efficient and lowers the hardware barrier to entry by up to 3 times when using adaptive optimizers since we do not need to calculate the gradients or maintain the optimizer states for most parameters. Instead, we only optimize the injected, much smaller low-rank matrices.\n\n• Our simple linear design allows us to merge the trainable matrices with the frozen weights when deployed, introducing no inference latency compared to a fully fine-tuned model, by construction.\n\n• LoRA is orthogonal to many prior methods and can be combined with many of them, such as prefix-tuning. We provide an example in Appendix E.",
            "type": "text",
            "section_path": [
              "content",
              "1 INTRODUCTION"
            ],
            "page_start": 1,
            "page_end": 1,
            "score": 0.018224999999999998
          },
          {
            "source_id": "S3",
            "paper_id": "2106.09685",
            "chunk_id": "d5e5cb1ec9a4af42837af976",
            "text": "LoRA adds trainable pairs of rank decomposition matrices in parallel to existing weight matrices. As mentioned in Section 4.2, we only apply LoRA to $W _ { q }$ and $\\hat { W _ { v } }$ in most experiments for simplicity. The number of trainable parameters is determined by the rank r and the shape of the original weights: $| \\Theta | = 2 \\times \\hat { L } _ { L o R A } \\times d _ { m o d e l } \\times r .$ , where $\\hat { L } _ { L o R A }$ is the number of weight matrices we apply LoRA to.",
            "type": "text",
            "section_path": [
              "content",
              "5 EMPIRICAL EXPERIMENTS",
              "5.1 BASELINES"
            ],
            "page_start": 5,
            "page_end": 5,
            "score": 0.017305882352941178
          },
          {
            "source_id": "S4",
            "paper_id": "2106.09685",
            "chunk_id": "318064dbe7c0d0363ec869bd",
            "text": "A Generalization of Full Fine-tuning. A more general form of fine-tuning allows the training of a subset of the pre-trained parameters. LoRA takes a step further and does not require the accumulated gradient update to weight matrices to have full-rank during adaptation. This means that when applying LoRA to all weight matrices and training all biases<sup>2</sup>, we roughly recover the expressiveness of full fine-tuning by setting the LoRA rank r to the rank of the pre-trained weight matrices. In other words, as we increase the number of trainable parameters <sup>3</sup>, training LoRA roughly converges to training the original model, while adapter-based methods converges to an MLP and prefix-based methods to a model that cannot take long input sequences.",
            "type": "text",
            "section_path": [
              "content",
              "4 OUR METHOD",
              "4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES"
            ],
            "page_start": 3,
            "page_end": 3,
            "score": 0.01542051282051282
          },
          {
            "source_id": "S5",
            "paper_id": "2305.14314",
            "chunk_id": "e02264b1a96ae177e7cbcb34",
            "text": "We present QLORA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.\n\nQLORA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).\n\nOur best model family, which we name Guanaco, outperforms all previous openly released models on the Vicuna benchmark, reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU.\n\nQLORA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) Double Quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) Paged Optimizers to manage memory spikes.",
            "type": "text",
            "section_path": [
              "abstract"
            ],
            "page_start": 0,
            "page_end": 0,
            "score": null
          },
          {
            "source_id": "S6",
            "paper_id": "2305.14314",
            "chunk_id": "0c971e77882e66207738bfcd",
            "text": "We have discussed how QLoRA works and how it can significantly reduce the required memory for finetuning models. The main question now is whether QLoRA can perform as well as full-model finetuning. Furthermore, we want to analyze the components of QLoRA including the impact of NormalFloat4 over standard Float4. The following sections will discuss the experiments that aimed at answering these questions.",
            "type": "text",
            "section_path": [
              "content",
              "4 QLoRA vs. Standard Finetuning"
            ],
            "page_start": 4,
            "page_end": 5,
            "score": 0.019829032258064518
          },
          {
            "source_id": "S7",
            "paper_id": "2305.14314",
            "chunk_id": "174ff68df2e1d5843e84eb35",
            "text": "To summarize, QLORA has one storage data type (usually 4-bit NormalFloat) and a computation data type (16-bit BrainFloat). We dequantize the storage data type to the computation data type to perform the forward and backward pass, but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.",
            "type": "text",
            "section_path": [
              "content",
              "3 QLORA Finetuning"
            ],
            "page_start": 4,
            "page_end": 4,
            "score": 0.019573015873015874
          },
          {
            "source_id": "S8",
            "paper_id": "2305.14314",
            "chunk_id": "362ea97891463d52312c0bd6",
            "text": "Training Setup To avoid confounding effects from different training objectives, we perform QLoRA finetuning with cross-entropy loss (supervised learning) without reinforcement learning, even for datasets that include human judgments of different responses. For datasets that have a clear distinction between instruction and response, we finetune only on the response (see ablations in Appendix B). For OASST1 and HH-RLHF, multiple responses are available. We then select the top response at every level of the conversation tree and finetune on the full selected conversation, including the instructions. In all of our experiments, we use NF4 QLORA with double quantization and paged optimizers to prevent memory spikes during gradient checkpointing. We do small hyperparameter searches for the 13B and 33B LLaMA models and we find that all hyperparameter settings found at 7B generalize (including number of epochs) except learning rate and batch size. We halve the learning rate for 33B and 65B while doubling the batch size.",
            "type": "text",
            "section_path": [
              "content",
              "5 Pushing the Chatbot State-of-the-art with QLoRA",
              "5.1 Experimental setup"
            ],
            "page_start": 7,
            "page_end": 7,
            "score": 0.018993442622950822
          }
        ],
        "count": 8,
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
    "is_error": false
  },
  {
    "user_question": "2020年以后有哪些计算机视觉论文和注意力相关",
    "agent_decision": {
      "selected_tool": "library_search",
      "reason": "元数据发现，使用类别和年份约束"
    },
    "mcp_request": {
      "tool": "library_search",
      "arguments": {
        "query": "2020年以后有哪些计算机视觉论文和注意力相关",
        "filters": {
          "category": "cs.CV",
          "year_from": "2020"
        },
        "limit": 20
      }
    },
    "mcp_response": {
      "status": "ok",
      "data": {
        "query": "2020年以后有哪些计算机视觉论文和注意力相关",
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
          "lexical_query": "\"computer vision\" OR \"attention mechanism\" OR \"after 2020\"",
          "translation_used": true,
          "translation_provider": "tencent",
          "translation_fallback": false,
          "stopwords_removed": [
            "paper"
          ],
          "rewriter_used": true,
          "rewriter_fallback": false,
          "core_terms": [
            "computer vision",
            "attention mechanism",
            "after 2020"
          ],
          "rewriter_error": null
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
    },
    "is_error": false
  },
  {
    "user_question": "Attention Is All You Need 的引用和被引用关系",
    "agent_decision": {
      "selected_tool": "library_citation",
      "mode": "graph",
      "direction": "both",
      "depth": 2
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
          "1409.1556",
          "1409.4842",
          "1512.03385",
          "1706.03762",
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
            "depth": 1,
            "path": [
              "1706.03762",
              "1512.03385"
            ]
          },
          {
            "source_paper_id": "1810.04805",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1811.06965",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "1811.06965",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1902.00751",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "1902.00751",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1909.08053",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1911.02150",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "1911.02150",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2005.14165",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2006.16236",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2006.16236",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2010.11929",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2010.11929",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2101.00190",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2101.00190",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2103.14030",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2103.14030",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2104.08691",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2104.08691",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2106.09685",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2203.15556",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2203.15556",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2205.14135",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2309.06180",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2312.06635",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2312.06635",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2405.04434",
            "target_arxiv_id": "1706.03762",
            "relation": "cites",
            "resolution": "local",
            "depth": 1,
            "path": [
              "2405.04434",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1512.03385",
            "target_arxiv_id": "1409.1556",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1706.03762",
              "1512.03385",
              "1409.1556"
            ]
          },
          {
            "source_paper_id": "1512.03385",
            "target_arxiv_id": "1409.4842",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1706.03762",
              "1512.03385",
              "1409.4842"
            ]
          },
          {
            "source_paper_id": "1811.06965",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1811.06965",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1902.00751",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1902.00751",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1905.11946",
            "target_arxiv_id": "1811.06965",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1905.11946",
              "1811.06965",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1909.08053",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1909.08053",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1909.08053",
            "target_arxiv_id": "1811.06965",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1909.08053",
              "1811.06965",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1910.02054",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1910.02054",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1910.02054",
            "target_arxiv_id": "1811.06965",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1910.02054",
              "1811.06965",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "1910.02054",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "1910.02054",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2005.14165",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2005.14165",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2005.14165",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2005.14165",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2006.16236",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2006.16236",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2010.11929",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2010.11929",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2010.11929",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2010.11929",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2101.00190",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2101.00190",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2101.00190",
            "target_arxiv_id": "1902.00751",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2101.00190",
              "1902.00751",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2101.00190",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2101.00190",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2103.14030",
            "target_arxiv_id": "2010.11929",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2103.14030",
              "2010.11929",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2104.08691",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2104.08691",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2104.08691",
            "target_arxiv_id": "1902.00751",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2104.08691",
              "1902.00751",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2104.08691",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2104.08691",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2104.08691",
            "target_arxiv_id": "2101.00190",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2104.08691",
              "2101.00190",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "1902.00751",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "1902.00751",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "2101.00190",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "2101.00190",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2106.09685",
            "target_arxiv_id": "2104.08691",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2106.09685",
              "2104.08691",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2203.02155",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2203.02155",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2203.15556",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2203.15556",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2205.14135",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2205.14135",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2205.14135",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "2006.16236",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2205.14135",
              "2006.16236",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2205.14135",
            "target_arxiv_id": "2010.11929",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2205.14135",
              "2010.11929",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "1810.04805",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2302.13971",
              "1810.04805",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2302.13971",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2302.13971",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "2203.15556",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2302.13971",
              "2203.15556",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2302.13971",
            "target_arxiv_id": "2205.14135",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2302.13971",
              "2205.14135",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.13245",
            "target_arxiv_id": "1911.02150",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.13245",
              "1911.02150",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.13245",
            "target_arxiv_id": "2205.14135",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.13245",
              "2205.14135",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.13245",
            "target_arxiv_id": "2302.13971",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.13245",
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.14314",
            "target_arxiv_id": "1902.00751",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.14314",
              "1902.00751",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.14314",
            "target_arxiv_id": "2101.00190",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.14314",
              "2101.00190",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.14314",
            "target_arxiv_id": "2104.08691",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.14314",
              "2104.08691",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.14314",
            "target_arxiv_id": "2106.09685",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.14314",
              "2106.09685",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.14314",
            "target_arxiv_id": "2302.13971",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.14314",
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.18290",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.18290",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2305.18290",
            "target_arxiv_id": "2302.13971",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2305.18290",
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "1909.08053",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "1909.08053",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "2005.14165",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "2005.14165",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "2101.00190",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "2101.00190",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "2104.08691",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "2104.08691",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "2205.14135",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "2205.14135",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2309.06180",
            "target_arxiv_id": "2302.13971",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2309.06180",
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2312.06635",
            "target_arxiv_id": "2006.16236",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2312.06635",
              "2006.16236",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2312.06635",
            "target_arxiv_id": "2205.14135",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2312.06635",
              "2205.14135",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2312.06635",
            "target_arxiv_id": "2302.13971",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2312.06635",
              "2302.13971",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2402.03300",
            "target_arxiv_id": "2309.06180",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2402.03300",
              "2309.06180",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2405.04434",
            "target_arxiv_id": "1911.02150",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2405.04434",
              "1911.02150",
              "1706.03762"
            ]
          },
          {
            "source_paper_id": "2405.04434",
            "target_arxiv_id": "2309.06180",
            "relation": "cites",
            "resolution": "local",
            "depth": 2,
            "path": [
              "2405.04434",
              "2309.06180",
              "1706.03762"
            ]
          }
        ],
        "scope": "local_catalog",
        "presentation": {
          "template_version": "library-answer-v1",
          "answer_type": "citation_graph",
          "render_policy": "verbatim",
          "answer_text": "目标论文：Attention Is All You Need\n直接引用：1 篇\n直接被引用：18 篇\n查询深度：2\n间接引用：2 篇\n间接被引用：7 篇\n\n引用（前10条）：\n1. Deep Residual Learning for Image Recognition\n────────\n2. Very Deep Convolutional Networks for Large-Scale Image Recognition\n3. Going Deeper with Convolutions\n\n被引用（前10条）：\n1. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n2. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism\n3. Parameter-Efficient Transfer Learning for NLP\n4. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism\n5. Fast Transformer Decoding: One Write-Head is All You Need\n6. Language Models are Few-Shot Learners\n7. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention\n8. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\n9. Prefix-Tuning: Optimizing Continuous Prompts for Generation\n10. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
        }
      },
      "warnings": [],
      "read_only": true
    },
    "is_error": false
  }
]
