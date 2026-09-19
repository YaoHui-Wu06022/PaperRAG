"""Materialize host-reviewed questions and exact source quotes as versioned spans.

These questions were authored after reading the stored MinerU abstracts. This is
a source-support benchmark, not verification of the papers' reported results.
"""

import json
from pathlib import Path

from paper_rag.library.catalog import Catalog
from paper_rag.library.common import evidence_id, digest
from paper_rag.library.settings import LibrarySettings

# Title selector, two independently sourced question/quote pairs.
REVIEWED = [
    (
        "Inception-v4",
        [
            (
                "How do residual connections affect Inception training speed?",
                "training with residual connections accelerates the training of Inception networks significantly",
            ),
            (
                "How does Inception-ResNet stabilize very wide residual networks?",
                "proper activation scaling stabilizes the training of very wide residual Inception networks",
            ),
        ],
    ),
    (
        "Dropout:",
        [
            (
                "What does Dropout randomly remove during training?",
                "randomly drop units (along with their connections) from the neural network during training",
            ),
            (
                "How does Dropout approximate averaging thinned networks at test time?",
                "using a single unthinned network that has smaller weights",
            ),
        ],
    ),
    (
        "Reconcile Once",
        [
            (
                "What maintained assets does the Reconcile Once librarian build?",
                "evidence cards, an authoritative metric ledger, and a claim graph",
            ),
            (
                "How many sources and evidence cards does Reconcile Once report?",
                "6,130 sources yielding 555,926 evidence cards",
            ),
        ],
    ),
    (
        "Going deeper",
        [
            (
                "How deep is GoogLeNet in the original Inception submission?",
                "GoogLeNet, a 22 layers deep network",
            ),
            (
                "What happened to the computational budget when Inception increased depth and width?",
                "increased the depth and width of the network while keeping the computational budget constant",
            ),
        ],
    ),
    (
        "Learning on the Job",
        [
            (
                "How does Learning on the Job retain feedback with frozen model weights?",
                "external memory that distils each episode into retrievable natural-language rules",
            ),
            (
                "What baseline-relative success did learning from outcome verdicts and corrections achieve?",
                "1.6× the baseline, and learning from corrections to 2.6×",
            ),
        ],
    ),
    (
        "ECA-Net",
        [
            (
                "How does ECA implement local cross-channel interaction?",
                "local crosschannel interaction strategy without dimensionality reduction, which can be efficiently implemented via 1D convolution",
            ),
            (
                "How does ECA determine the coverage of local cross-channel interaction?",
                "adaptively select kernel size of 1D convolution",
            ),
        ],
    ),
    (
        "Squeeze-and-Excitation",
        [
            (
                "What does the SE block explicitly model?",
                "interdependencies between channels",
            ),
            (
                "What ILSVRC 2017 top-5 error is reported by SENets?",
                "top-5 error to 2.251%",
            ),
        ],
    ),
    (
        "NormFace",
        [
            (
                "What similarity does NormFace optimize instead of inner-product?",
                "optimizes cosine similarity instead of inner-product",
            ),
            (
                "How does NormFace reformulate metric learning?",
                "introducing an agent vector for each class",
            ),
        ],
    ),
    (
        "A Discriminative Feature",
        [
            (
                "What distances does center loss penalize?",
                "distances between the deep features and their corresponding class centers",
            ),
            (
                "What signals jointly supervise the center-loss CNN?",
                "joint supervision of softmax loss and center loss",
            ),
        ],
    ),
    (
        "KGVoyager",
        [
            (
                "What loop does KGVoyager use to refine SPARQL queries?",
                "think–act–observe loop with search, exploration, and execution tools",
            ),
            ("What index does KGVoyager require?", "only a lightweight class index"),
        ],
    ),
    (
        "ImageNet Classification",
        [
            (
                "How many convolutional and fully connected layers did AlexNet use?",
                "five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers",
            ),
            (
                "What ILSVRC-2012 top-5 test error did the AlexNet variant achieve?",
                "winning top-5 test error rate of 15.3%",
            ),
        ],
    ),
    (
        "Only Ask",
        [
            (
                "What are the three GDP-RAG planning design choices?",
                "(1) preliminary retrieval to ground planning before execution, (2) a gap-conditioned planning prompt that asks only for missing information, and (3) a skeletal trajectory",
            ),
            (
                "What accuracy and cost-of-pass does GDP-RAG report?",
                "accuracy (60.63%) among all compared systems while maintaining a cost-of-pass of 0.51",
            ),
        ],
    ),
    (
        "BERT:",
        [
            (
                "How does BERT condition its bidirectional pretraining representations?",
                "jointly conditioning on both left and right context in all layers",
            ),
            (
                "What SQuAD v1.1 Test F1 does BERT report?",
                "SQuAD v1.1 question answering Test F1 to 93.2",
            ),
        ],
    ),
    (
        "Deep Residual",
        [
            (
                "How does ResNet reformulate layer learning?",
                "learning residual functions with reference to the layer inputs",
            ),
            (
                "How deep were the residual nets evaluated on ImageNet?",
                "a depth of up to 152 layers",
            ),
        ],
    ),
    (
        "An Image is Worth",
        [
            (
                "What input sequence does a pure Vision Transformer process?",
                "sequences of image patches",
            ),
            (
                "Which image benchmarks are listed for Vision Transformer transfer?",
                "ImageNet, CIFAR-100, VTAB",
            ),
        ],
    ),
    (
        "Aggregated Residual",
        [
            (
                "How is cardinality defined in ResNeXt?",
                "the size of the set of transformations",
            ),
            (
                "What comparison does ResNeXt report for increasing cardinality versus depth or width?",
                "increasing cardinality is more effective than going deeper or wider",
            ),
        ],
    ),
    (
        "Long Short-Term",
        [
            (
                "What maintains constant error flow in the original LSTM?",
                '"constant error carrousels" within special units',
            ),
            (
                "What is LSTM computational complexity per time step and weight?",
                "computational complexity per time step and weight is  O(1)",
            ),
        ],
    ),
    (
        "Auto-Encoding",
        [
            (
                "What enables stochastic gradient optimization in Auto-Encoding Variational Bayes?",
                "a reparameterization of the variational lower bound",
            ),
            (
                "What is the approximate inference model also called in Auto-Encoding Variational Bayes?",
                "a recognition model",
            ),
        ],
    ),
    (
        "Supervised Contrastive",
        [
            (
                "How are same-class and different-class clusters treated in supervised contrastive learning?",
                "Clusters of points belonging to the same class are pulled together in embedding space, while simultaneously pushing apart clusters of samples from different classes",
            ),
            (
                "What ImageNet top-1 accuracy does SupCon report on ResNet-200?",
                "top-1 accuracy of 81.4%",
            ),
        ],
    ),
    (
        "Adam:",
        [
            (
                "What estimates form the basis of Adam?",
                "adaptive estimates of lower-order moments",
            ),
            ("What norm defines the AdaMax variant?", "infinity norm"),
        ],
    ),
    (
        "Attention is All",
        [
            (
                "Which sequence modelling components does the Transformer dispense with?",
                "dispensing with recurrence and convolutions entirely",
            ),
            (
                "What WMT 2014 English-to-French BLEU did the Transformer report and after how much training?",
                "BLEU score of 41.0 after training for 3.5 days on eight GPUs",
            ),
        ],
    ),
    (
        "Citation-Enforced",
        [
            (
                "What provenance is preserved by Citation-Enforced RAG source-first ingestion?",
                "document identifiers and page-level provenance",
            ),
            (
                "When may Citation-Enforced RAG decline to answer?",
                "when document support is insuficient",
            ),
        ],
    ),
    (
        "EfficientNet:",
        [
            (
                "Which dimensions are jointly scaled by EfficientNet?",
                "depth/width/resolution",
            ),
            (
                "What ImageNet top-1 accuracy is reported for EfficientNet-B7?",
                "84.3% top-1 accuracy",
            ),
        ],
    ),
    (
        "Batch Normalization:",
        [
            (
                "At what granularity does Batch Normalization perform normalization during training?",
                "each training mini-batch",
            ),
            (
                "How many fewer steps did the Batch Normalization image classification model need for the same accuracy?",
                "14 times fewer training steps",
            ),
        ],
    ),
    (
        "Generative Adversarial",
        [
            (
                "What does the discriminator in the original GAN estimate?",
                "the probability that a sample came from the training data rather than G",
            ),
            (
                "Does the original GAN require Markov chains during training or generation?",
                "There is no need for any Markov chains",
            ),
        ],
    ),
    (
        "Exponential Moving",
        [
            (
                "Why does EMA require less learning rate decay compared with SGD in the reported study?",
                "averaging naturally reduces noise, introducing a form of implicit regularization",
            ),
            (
                "Besides generalization, what benefits are reported for EMA models?",
                "robustness to noisy labels, ii) prediction consistency, iii) calibration and iv) transfer learning",
            ),
        ],
    ),
]


def main():
    settings = LibrarySettings.load(Path.cwd())
    cases = []
    with Catalog(settings) as cat:
        docs = cat.rows("SELECT * FROM documents WHERE active_revision IS NOT NULL")
        reviewed_sources = json.loads(
            (settings.root / "eval/reviewed_sources.json").read_text(encoding="utf-8")
        )
        for doc in docs:
            title = json.loads(doc["metadata"])["title"]
            if title in reviewed_sources:
                identity = cat.one(
                    "SELECT source_hash,parser_hash FROM revisions WHERE revision_id=?",
                    (doc["active_revision"],),
                )
                if identity != reviewed_sources[title]:
                    raise ValueError(
                        "Source or MinerU output changed; re-review benchmark before compiling: "
                        + title
                    )
        for index, (selector, pairs) in enumerate(REVIEWED):
            matches = [
                d
                for d in docs
                if json.loads(d["metadata"])["title"].startswith(selector)
            ]
            assert len(matches) == 1, selector
            doc = matches[0]
            blocks = cat.rows(
                "SELECT * FROM blocks WHERE revision_id=?", (doc["active_revision"],)
            )
            for question, quote in pairs:
                matches = [b for b in blocks if quote in b["text"]]
                assert matches, (selector, quote)
                b = next((b for b in matches if b["region"] == "abstract"), matches[0])
                start = b["text"].index(quote)
                ref = evidence_id(
                    doc["document_id"],
                    doc["active_revision"],
                    b["block_id"],
                    start,
                    start + len(quote),
                )
                cases.append(
                    {
                        "id": f"fact-{len(cases)+1:03}",
                        "query": question,
                        "evidence_ids": [ref],
                        "expected_quote": quote,
                        "category": "fact",
                        "reviewed": True,
                        "reviewer": "host-agent-source-span-review-2026-09-19",
                        "group": doc["document_id"],
                        "split": "tune" if index < 4 else "test",
                    }
                )
        structured = [
            (
                "Attention is All",
                "b000042",
                "What is the scaled dot product attention formula in terms of Q K V?",
                "formula",
                None,
            ),
            (
                "Attention is All",
                "b000050",
                "How are attention heads concatenated and projected in MultiHead attention?",
                "formula",
                None,
            ),
            (
                "Attention is All",
                "b000067",
                "What maximum path length does Table 1 give for recurrent versus self-attention layers?",
                "table",
                None,
            ),
            (
                "BERT:",
                "b000068",
                "What BERTLARGE RTE accuracy is reported in GLUE Table 1?",
                "table",
                "RTE2.5k = 70.1",
            ),
            (
                "Adam:",
                "b000145",
                "How does the Adam appendix use the lemma to construct the regret proof?",
                "appendix",
                None,
            ),
            (
                "BERT:",
                "b000194",
                "Why does the BERT appendix keep the selected word unchanged ten percent of the time?",
                "appendix",
                None,
            ),
        ]
        for selector, bid, question, category, quote in structured:
            doc = next(
                d
                for d in docs
                if json.loads(d["metadata"])["title"].startswith(selector)
            )
            b = cat.one(
                "SELECT * FROM blocks WHERE revision_id=? AND block_id=?",
                (doc["active_revision"], bid),
            )
            quote = quote or b["text"]
            start = b["text"].index(quote)
            cases.append(
                {
                    "id": f"structured-{len(cases)+1}",
                    "query": question,
                    "category": category,
                    "evidence_ids": [
                        evidence_id(
                            doc["document_id"],
                            doc["active_revision"],
                            bid,
                            start,
                            start + len(quote),
                        )
                    ],
                    "expected_quote": quote,
                    "regions": ["abstract", "body", "appendix"],
                    "reviewed": True,
                    "group": doc["document_id"],
                    "split": "test",
                }
            )
        for question, selectors in [
            (
                "Compare channel modelling in Squeeze-and-Excitation and ECA-Net.",
                ["Squeeze-and-Excitation", "ECA-Net"],
            ),
            (
                "Compare the input representations in Vision Transformer and BERT.",
                ["An Image is Worth", "BERT:"],
            ),
        ]:
            references = []
            source_groups = []
            for selector in selectors:
                doc = next(
                    d
                    for d in docs
                    if json.loads(d["metadata"])["title"].startswith(selector)
                )
                references.extend(
                    next(
                        c["evidence_ids"]
                        for c in cases
                        if c["group"] == doc["document_id"]
                    )
                )
                source_groups.append(doc["document_id"])
            cases.append(
                {
                    "id": f"comparison-{len(cases)+1}",
                    "query": question,
                    "evidence_ids": references,
                    "category": "comparison",
                    "reviewed": True,
                    "group": "comparison-" + str(len(cases)),
                    "source_groups": source_groups,
                    "split": "test",
                }
            )
        for question in [
            "What is the private MinerU account password used by the authors?",
            "What exact accuracy will a new unpublished experiment achieve tomorrow?",
        ]:
            cases.append(
                {
                    "id": f"unanswerable-{len(cases)+1}",
                    "query": question,
                    "evidence_ids": [],
                    "category": "unanswerable",
                    "reviewed": True,
                    "group": "unanswerable",
                    "split": "test",
                    "expected_behavior": "Abstain: the source library cannot establish this answer; do not infer absence from top-k.",
                }
            )
        report = {
            "schema_version": 1,
            "corpus_version": cat.snapshot(),
            "cases": cases,
            "limitations": "Host-agent reviewed source-support benchmark. Includes parsed tables and formulas; OCR support does not independently replicate paper results. Metadata, graph and long-read coverage are separate integration checks.",
        }
        by_title = {json.loads(d["metadata"])["title"]: d for d in docs}
        lstm = by_title["Long Short-Term Memory"]
        resnet = by_title["Deep Residual Learning for Image Recognition"]
        full_text = "".join(
            b["text"]
            for b in cat.rows(
                "SELECT text FROM blocks WHERE revision_id=? ORDER BY ordinal,block_id",
                (lstm["active_revision"],),
            )
        )
        report["tool_cases"] = [
            {
                "id": "metadata-count",
                "command": "papers_count",
                "request": {"filters": {"source_kind": "paper"}},
                "expected_count": 26,
                "reviewed": True,
            },
            {
                "id": "metadata-title",
                "command": "papers_count",
                "request": {"query": "Long Short-Term Memory"},
                "expected_count": 1,
                "reviewed": True,
            },
            {
                "id": "reference-edge",
                "command": "citations",
                "request": {
                    "document_id": resnet["document_id"],
                    "direction": "outgoing",
                },
                "expected_target": lstm["document_id"],
                "expected_quote": "S. Hochreiter and J. Schmidhuber. Long short-term memory.",
                "reviewed": True,
            },
            {
                "id": "long-paged-read",
                "command": "read",
                "request": {"document_id": lstm["document_id"], "max_chars": 997},
                "expected_hash": digest(full_text),
                "reviewed": True,
            },
        ]
    destination = settings.home / "eval" / "reviewed-facts.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(cases), "path": str(destination)}))


if __name__ == "__main__":
    main()
