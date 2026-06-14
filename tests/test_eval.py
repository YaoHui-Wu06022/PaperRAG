import json

from paper_rag.test import eval as eval_module


def test_eval_metrics_from_fake_plan_payload(settings):
    cases = [
        {
            "id": "metadata_bert_year",
            "query": "BERT 是哪一年发表的？",
            "expected_route": "metadata",
            "expected_intent": "lookup",
            "expected_papers": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"],
            "expected_fields": ["year"],
            "expected_values": {"year": "2019"},
        },
        {
            "id": "content_resnet",
            "query": "ResNet 的结构是什么？",
            "expected_route": "content",
            "expected_intent": "lookup",
            "expected_papers": ["Deep Residual Learning for Image Recognition"],
            "expected_terms": ["shortcut", "identity"],
        },
    ]

    def fake_plan(_settings, query, *, debug, corpus):
        assert debug is True
        assert corpus is not None
        if "BERT" in query:
            return {
                "route": "metadata",
                "intent": "lookup",
                "status": "ok",
                "plan": {"return_fields": ["year"], "scope": ["paper=BERT"]},
                "resolved": {
                    "aliases": [
                        {
                            "alias": "BERT",
                            "canonical": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                        }
                    ]
                },
                "results": {
                    "items": [
                        {
                            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                            "values": {"year": {"preprint_year": 2018, "publish_year": 2019}},
                        }
                    ]
                },
                "debug": {"timings_ms": {"top_parser": 1.0}},
            }
        return {
            "route": "content",
            "intent": "lookup",
            "status": "ok",
            "plan": {"scope": ["paper=ResNet"]},
            "resolved": {
                "aliases": [
                    {
                        "alias": "ResNet",
                        "canonical": "Deep Residual Learning for Image Recognition",
                    }
                ]
            },
            "results": {
                "contexts": [
                    {
                        "chunk_id": "resnet::chunk_0001",
                        "title": "Deep Residual Learning for Image Recognition",
                        "section_path": ["3 Deep Residual Learning"],
                        "text": "identity shortcut connections perform residual learning.",
                    },
                ]
            },
            "debug": {"timings_ms": {"dense": 2.0}},
        }

    report = eval_module.run_eval(settings, cases, planner=fake_plan)

    summary = report["summary"]
    assert summary["route_accuracy"]["score"] == 1.0
    assert summary["intent_accuracy"]["score"] == 1.0
    assert summary["slot_accuracy"]["score"] == 1.0
    assert summary["paper_scope_recall"]["score"] == 1.0
    assert summary["recall_at_5"]["score"] == 1.0


def test_eval_continues_when_single_case_fails(settings):
    cases = [
        {"id": "ok", "query": "ok", "expected_route": "metadata"},
        {"id": "bad", "query": "bad", "expected_route": "content"},
    ]

    def fake_plan(_settings, query, *, debug, corpus):
        if query == "bad":
            raise RuntimeError("service unavailable")
        return {"route": "metadata", "status": "ok", "results": {}, "debug": {"timings_ms": {}}}

    report = eval_module.run_eval(settings, cases, planner=fake_plan)

    assert report["summary"]["cases"] == 2
    assert report["summary"]["errors"] == 1
    assert report["failures"][0]["id"] == "bad"
    assert "service unavailable" in report["case_results"][1]["error"]


def test_eval_main_writes_json_report(monkeypatch, tmp_path):
    report = eval_module.build_report([])
    output_path = tmp_path / "eval" / "latest.json"

    monkeypatch.setattr(eval_module.Settings, "load", lambda _root: object())
    monkeypatch.setattr(eval_module, "load_cases", lambda _path: [])
    monkeypatch.setattr(eval_module, "run_eval", lambda _settings, _cases: report)

    assert eval_module.main(["--save-json", str(output_path)]) == 0

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["summary"]["cases"] == 0
    assert "case_results" in saved
    assert "failures" in saved
