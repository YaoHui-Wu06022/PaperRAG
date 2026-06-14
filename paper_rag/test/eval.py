"""轻量 plan 链路评测入口。

默认运行当前包内的小规模 golden set：

    python -m paper_rag.test.eval
    python -m paper_rag.test.eval --save-json data/eval/latest.json

第一版只评估 plan evidence，不做最终答案 judge。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.plan import run_plan


CASE_TOP_K = 5
DEFAULT_CASES_PATH = Path(__file__).with_name("eval_cases.json")
MetricValue = bool | None
PlanRunner = Callable[..., dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-rag-eval", description="运行轻量 plan 链路评测")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="项目根目录，默认当前目录")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="评测 case JSON 文件")
    parser.add_argument("--save-json", type=Path, help="把完整评测报告保存为 JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load(args.project_root)
    cases = load_cases(args.cases)
    report = run_eval(settings, cases)
    print_report(report)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n已保存 JSON 报告：{args.save_json}")
    return 0


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("eval cases 文件必须是 JSON list")
    cases: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个 case 必须是对象")
        if not item.get("id") or not item.get("query") or not item.get("expected_route"):
            raise ValueError(f"第 {index} 个 case 缺少 id/query/expected_route")
        cases.append(item)
    return cases


def run_eval(
    settings: Settings,
    cases: list[dict[str, Any]],
    *,
    planner: PlanRunner = run_plan,
) -> dict[str, Any]:
    corpus = CorpusContext(settings)
    case_results = [
        run_case(settings, case, corpus=corpus, planner=planner)
        for case in cases
    ]
    return build_report(case_results)


def run_case(
    settings: Settings,
    case: dict[str, Any],
    *,
    corpus: CorpusContext,
    planner: PlanRunner = run_plan,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        evidence = planner(settings, case["query"], debug=True, corpus=corpus)
    except Exception as exc:
        wall_ms = elapsed_ms(start)
        return error_case_result(case, exc, wall_ms)
    wall_ms = elapsed_ms(start)
    return evaluate_case(case, evidence, wall_ms)


def evaluate_case(case: dict[str, Any], evidence: dict[str, Any], wall_ms: float) -> dict[str, Any]:
    contexts = top_contexts(evidence, CASE_TOP_K)
    result = {
        "id": case["id"],
        "query": case["query"],
        "expected_route": case.get("expected_route"),
        "actual_route": evidence.get("route"),
        "expected_intent": case.get("expected_intent"),
        "actual_intent": evidence.get("intent"),
        "status": evidence.get("status"),
        "route_ok": evidence.get("route") == case.get("expected_route"),
        "intent_ok": metric_equals(evidence.get("intent"), case.get("expected_intent")),
        "slot_ok": check_slot_accuracy(case, evidence),
        "paper_scope_recall": check_expected_papers(case, evidence),
        "recall_at_5": check_recall_at_5(case, contexts),
        "contexts_at_5": len(contexts),
        "wall_ms": wall_ms,
        "timings_ms": extract_timings(evidence),
        "warnings": evidence.get("warnings") or [],
        "top_contexts": summarize_contexts(contexts, case),
    }
    return result


def error_case_result(case: dict[str, Any], error: Exception, wall_ms: float) -> dict[str, Any]:
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_route": case.get("expected_route"),
        "actual_route": None,
        "expected_intent": case.get("expected_intent"),
        "actual_intent": None,
        "status": "error",
        "route_ok": False,
        "intent_ok": False if case.get("expected_intent") else None,
        "slot_ok": False if has_slot_expectations(case) else None,
        "paper_scope_recall": False if case.get("expected_papers") else None,
        "recall_at_5": False if case.get("expected_route") == "content" else None,
        "contexts_at_5": 0,
        "wall_ms": wall_ms,
        "timings_ms": {},
        "warnings": [],
        "top_contexts": [],
        "error": str(error),
    }


def build_report(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": summarize_results(case_results),
        "by_route": {
            route: summarize_results([result for result in case_results if result.get("expected_route") == route])
            for route in sorted({str(result.get("expected_route")) for result in case_results})
        },
        "case_results": case_results,
        "failures": failure_rows(case_results),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    timings = stage_timings(results)
    return {
        "cases": len(results),
        "errors": sum(1 for result in results if result.get("error")),
        "route_accuracy": metric_summary(results, "route_ok"),
        "intent_accuracy": metric_summary(results, "intent_ok"),
        "slot_accuracy": metric_summary(results, "slot_ok"),
        "paper_scope_recall": metric_summary(results, "paper_scope_recall"),
        "recall_at_5": metric_summary(results, "recall_at_5"),
        "latency_ms": {
            "wall_p50": percentile([float(result["wall_ms"]) for result in results], 50),
            "wall_p95": percentile([float(result["wall_ms"]) for result in results], 95),
            "stages": {
                stage: {
                    "p50": percentile(values, 50),
                    "p95": percentile(values, 95),
                }
                for stage, values in timings.items()
            },
        },
    }


def metric_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [result.get(key) for result in results if result.get(key) is not None]
    correct = sum(1 for value in values if value is True)
    return {
        "correct": correct,
        "total": len(values),
        "score": safe_divide(correct, len(values)),
    }


def stage_timings(results: list[dict[str, Any]]) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {}
    for result in results:
        for stage, value in (result.get("timings_ms") or {}).items():
            if isinstance(value, int | float):
                timings.setdefault(stage, []).append(float(value))
    return timings


def failure_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for result in results:
        failed_metrics = [
            key
            for key in ["route_ok", "intent_ok", "slot_ok", "paper_scope_recall", "recall_at_5"]
            if result.get(key) is False
        ]
        if result.get("error") or failed_metrics:
            failures.append({
                "id": result["id"],
                "query": result["query"],
                "expected_route": result.get("expected_route"),
                "actual_route": result.get("actual_route"),
                "failed_metrics": failed_metrics,
                "error": result.get("error"),
                "warnings": result.get("warnings") or [],
            })
    return failures


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("轻量评测完成")
    print(f"Cases: {summary['cases']}，Errors: {summary['errors']}")
    for label, key in [
        ("Route Accuracy", "route_accuracy"),
        ("Intent Accuracy", "intent_accuracy"),
        ("Slot Accuracy", "slot_accuracy"),
        ("Paper Scope Recall", "paper_scope_recall"),
        ("Recall@5", "recall_at_5"),
    ]:
        print_metric(label, summary[key])
    latency = summary["latency_ms"]
    print(f"Latency wall p50/p95: {format_ms(latency['wall_p50'])} / {format_ms(latency['wall_p95'])}")
    if latency["stages"]:
        print("Stage p50/p95:")
        for stage, values in sorted(latency["stages"].items()):
            print(f"  - {stage}: {format_ms(values['p50'])} / {format_ms(values['p95'])}")
    if report["failures"]:
        print("\n失败 case:")
        for failure in report["failures"]:
            failed = ",".join(failure["failed_metrics"]) or "error"
            print(f"- {failure['id']} [{failed}] expected={failure['expected_route']} actual={failure['actual_route']}")


def print_metric(label: str, metric: dict[str, Any]) -> None:
    score = metric.get("score")
    if score is None:
        print(f"{label}: N/A")
        return
    numerator = metric.get("correct")
    denominator = metric.get("total")
    print(f"{label}: {score:.3f} ({numerator}/{denominator})")


def top_contexts(evidence: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    results = evidence.get("results") if isinstance(evidence.get("results"), dict) else {}
    contexts = results.get("contexts") if isinstance(results, dict) else []
    if not isinstance(contexts, list):
        return []
    debug_contexts = debug_context_units_by_id(evidence)
    merged = []
    for context in contexts[:top_k]:
        if not isinstance(context, dict):
            continue
        chunk_id = str(context.get("chunk_id") or "")
        merged.append({**debug_contexts.get(chunk_id, {}), **context})
    return merged


def debug_context_units_by_id(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    debug = evidence.get("debug") if isinstance(evidence.get("debug"), dict) else {}
    context_units = debug.get("context_units") if isinstance(debug, dict) else []
    if not isinstance(context_units, list):
        return {}
    return {
        str(unit.get("chunk_id") or ""): unit
        for unit in context_units
        if isinstance(unit, dict) and unit.get("chunk_id")
    }


def summarize_contexts(contexts: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": context.get("chunk_id"),
            "title": context.get("title"),
            "section_path": context.get("section_path"),
            "pages": context.get("pages"),
            "matched_terms": matched_terms(context, case.get("expected_terms") or []),
        }
        for context in contexts
    ]


def check_slot_accuracy(case: dict[str, Any], evidence: dict[str, Any]) -> MetricValue:
    if not has_slot_expectations(case):
        return None
    haystack = searchable_evidence_text(evidence)
    checks: list[bool] = []
    expected_return_side = case.get("expected_return_side")
    if expected_return_side:
        plan = evidence.get("plan") if isinstance(evidence.get("plan"), dict) else {}
        checks.append(plan.get("return_side") == expected_return_side)
    expected_fields = case.get("expected_fields") or []
    checks.extend(contains_text(haystack, field) for field in expected_fields)
    expected_values = case.get("expected_values") or {}
    if isinstance(expected_values, dict):
        checks.extend(check_expected_value(evidence, field, value) for field, value in expected_values.items())
    else:
        checks.extend(contains_text(haystack, value) for value in expected_values)
    expected_papers = case.get("expected_papers") or []
    checks.extend(contains_text(haystack, paper) for paper in expected_papers)
    return all(checks) if checks else None


def has_slot_expectations(case: dict[str, Any]) -> bool:
    return any(case.get(key) for key in ["expected_papers", "expected_fields", "expected_values", "expected_return_side"])


def check_expected_value(evidence: dict[str, Any], field: str, expected: Any) -> bool:
    matches = values_for_key({
        "plan": evidence.get("plan"),
        "resolved": evidence.get("resolved"),
        "results": evidence.get("results"),
    }, field)
    if matches:
        return any(contains_text(value, expected) for value in matches)
    return contains_text(searchable_evidence_text(evidence), expected)


def values_for_key(value: Any, field: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if normalize_text(key) == normalize_text(field):
                values.append(item)
            values.extend(values_for_key(item, field))
    elif isinstance(value, list):
        for item in value:
            values.extend(values_for_key(item, field))
    return values


def check_expected_papers(case: dict[str, Any], evidence: dict[str, Any]) -> MetricValue:
    expected_papers = case.get("expected_papers") or []
    if not expected_papers:
        return None
    haystack = searchable_evidence_text(evidence)
    return all(contains_text(haystack, paper) for paper in expected_papers)


def check_recall_at_5(case: dict[str, Any], contexts: list[dict[str, Any]]) -> MetricValue:
    if case.get("expected_route") != "content":
        return None
    if not contexts:
        return False
    terms = case.get("expected_terms") or []
    papers = case.get("expected_papers") or []
    if not terms and not papers:
        return None
    return any(matched_terms(context, terms) or any(contains_text(flatten_text(context), paper) for paper in papers) for context in contexts)


def matched_terms(context: dict[str, Any], terms: list[str]) -> list[str]:
    text = flatten_text(context)
    return [term for term in terms if contains_text(text, term)]


def searchable_evidence_text(evidence: dict[str, Any]) -> str:
    return flatten_text({
        "plan": evidence.get("plan"),
        "resolved": evidence.get("resolved"),
        "results": evidence.get("results"),
    })


def flatten_text(value: Any) -> str:
    parts: list[str] = []
    collect_text(value, parts)
    return " ".join(parts)


def collect_text(value: Any, parts: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            parts.append(str(key))
            collect_text(item, parts)
    elif isinstance(value, list):
        for item in value:
            collect_text(item, parts)
    elif value is not None:
        parts.append(str(value))


def metric_equals(actual: Any, expected: Any) -> MetricValue:
    if expected is None:
        return None
    return actual == expected


def contains_text(haystack: Any, needle: Any) -> bool:
    return normalize_text(needle) in normalize_text(haystack)


def normalize_text(value: Any) -> str:
    text = flatten_text(value) if isinstance(value, dict | list) else str(value or "")
    return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", text.casefold()))


def extract_timings(evidence: dict[str, Any]) -> dict[str, float]:
    debug = evidence.get("debug") if isinstance(evidence.get("debug"), dict) else {}
    timings = debug.get("timings_ms") if isinstance(debug, dict) else {}
    if not isinstance(timings, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in timings.items()
        if isinstance(value, int | float)
    }


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} ms"


if __name__ == "__main__":
    raise SystemExit(main())
