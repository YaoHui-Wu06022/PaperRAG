import json

from paper_rag.ingest.citation_graph import build_citation_graph

from conftest import add_paper, save_manifest


def test_citation_graph_builds_only_conservative_local_edges(settings):
    target = add_paper(
        settings,
        file_hash="target-hash",
        paper_id="resnet",
        title="Deep Residual Learning for Image Recognition",
        authors=["Kaiming He"],
        year={"preprint_year": 2015, "publish_year": 2016},
        venue="CVPR",
    )
    source = add_paper(
        settings,
        file_hash="source-hash",
        paper_id="vit",
        title="An Image is Worth 16x16 Words",
        authors=["Alexey Dosovitskiy"],
        year={"preprint_year": 2020, "publish_year": 2021},
        venue="ICLR",
        references=[
            {
                "reference_id": "ref_001",
                "ref_index": 1,
                "raw_text": "[1] He et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
                "page": 9,
                "source_block_id": "b000009",
            },
            {
                "reference_id": "ref_002",
                "ref_index": 2,
                "raw_text": "[2] Deep Residual Learning for Image Recognition without author or year.",
                "page": 9,
                "source_block_id": "b000009",
            },
        ],
    )
    manifest = save_manifest(settings, [target, source])

    result = build_citation_graph(settings, manifest)

    graph = json.loads(result.path.read_text(encoding="utf-8"))
    assert result.node_count == 2
    assert result.edge_count == 1
    assert graph["edges"][0]["source_paper_id"] == "vit"
    assert graph["edges"][0]["target_paper_id"] == "resnet"
    assert graph["edges"][0]["ref_index"] == 1
