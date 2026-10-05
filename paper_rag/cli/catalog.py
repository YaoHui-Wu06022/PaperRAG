"""Catalog 同步和状态 CLI。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random

from paper_rag.config import Settings
from paper_rag.presentation import attach_presentation, citation_presentation


def add_catalog_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册 Catalog 子命令。"""

    catalog = subparsers.add_parser("catalog", help="维护本地论文 Catalog")
    commands = catalog.add_subparsers(dest="catalog_command", required=True)

    sync = commands.add_parser("sync", help="重建 Catalog 派生索引")
    sync.add_argument("--dry-run", action="store_true", help="只显示当前状态")
    sync.add_argument("--json", action="store_true", help="输出 JSON")
    sync.set_defaults(handler=handle_sync)

    status = commands.add_parser("status", help="查看 Catalog 状态")
    status.add_argument("--json", action="store_true", help="输出 JSON")
    status.set_defaults(handler=handle_status)

    chunks = commands.add_parser("chunks", help="查看本地 Chunk JSON 或抽样检查切分结果")
    chunks.add_argument("--paper-id", help="按论文读取 chunks.json")
    chunks.add_argument("--sample", type=int, help="从所有论文中随机抽样")
    chunks.add_argument("--seed", type=int, default=None, help="抽样随机种子")
    chunks.add_argument("--type", dest="chunk_type", choices=("text", "table", "image", "figure", "chart", "equation", "formula", "code", "list", "unknown"))
    chunks.add_argument("--limit", type=int, default=10000)
    chunks.add_argument("--json", action="store_true", help="输出 JSON")
    chunks.set_defaults(handler=handle_chunks)

    index = subparsers.add_parser("index", help="维护 LlamaIndex 向量索引")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    index_status = index_commands.add_parser("status", help="查看 LlamaIndex 状态")
    index_status.add_argument("--json", action="store_true")
    index_status.set_defaults(handler=handle_index_status)
    index_rebuild = index_commands.add_parser("rebuild", help="重建 LlamaIndex 向量索引")
    index_rebuild.add_argument("--mode", choices=("auto", "incremental", "full"), default="auto")
    index_rebuild.add_argument("--json", action="store_true")
    index_rebuild.set_defaults(handler=handle_index_rebuild)

    citation = subparsers.add_parser("citation", help="查询本地引用关系")
    citation_commands = citation.add_subparsers(dest="citation_command", required=True)
    graph = citation_commands.add_parser("graph", help="查询本地引用关系图")
    graph.add_argument("--paper-id", required=True)
    graph.add_argument("--direction", choices=("in", "out", "both"), default="both")
    graph.add_argument("--depth", type=int, default=1)
    graph.add_argument("--json", action="store_true")
    graph.set_defaults(handler=handle_citation_graph)


def handle_sync(args: argparse.Namespace) -> int:
    """预览或重建本地 Catalog。"""

    from paper_rag.catalog.service import catalog_status, rebuild_catalog

    settings = Settings.load(args.project_root)
    payload = {"status": "dry_run", "current": catalog_status(settings)} if args.dry_run else rebuild_catalog(settings)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_catalog(payload))
    return 0


def handle_status(args: argparse.Namespace) -> int:
    """读取 Catalog 状态。"""

    from paper_rag.catalog.service import catalog_status

    payload = catalog_status(Settings.load(args.project_root))
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_catalog(payload))
    return 0


def handle_chunks(args: argparse.Namespace) -> int:
    """读取每篇论文的 chunks.json，并支持可复现抽样。"""

    from paper_rag.catalog.service import get_metadata, scan_catalog

    settings = Settings.load(args.project_root)
    if args.paper_id:
        record = get_metadata(settings, args.paper_id)
        paths = [record.mineru_dir / "chunks.json"] if record and record.mineru_dir else []
    else:
        paths = [record.mineru_dir / "chunks.json" for record in scan_catalog(settings) if record.mineru_dir and (record.mineru_dir / "chunks.json").is_file()]
    items: list[dict] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        values = payload.get("chunks", []) if isinstance(payload, dict) else []
        if isinstance(values, list):
            items.extend(item for item in values if isinstance(item, dict))
    if args.chunk_type:
        items = [item for item in items if item.get("type") == args.chunk_type]
    if args.paper_id and not items:
        payload = {"status": "not_found", "data": {"items": [], "count": 0, "stats": {}}}
        _print_catalog(payload, args.json)
        return 1
    if args.sample is not None:
        if args.sample < 1:
            payload = {"status": "invalid_input", "data": {"items": [], "count": 0}, "warnings": ["sample must be positive"]}
            _print_catalog(payload, args.json)
            return 1
        if len(items) > args.sample:
            items = random.Random(args.seed).sample(items, args.sample)
    else:
        items = items[: max(1, int(args.limit))]
    payload = {"status": "ok", "data": {"items": items, "count": len(items), "stats": _chunk_stats(items)}}
    _print_catalog(payload, args.json)
    return 0


def _chunk_stats(items: list[dict]) -> dict:
    """生成抽样的类型、区域、资源和路径重复统计。"""

    duplicate_tail = 0
    for item in items:
        path = item.get("section_path") or []
        if len(path) > 1 and path[-1] == path[-2]:
            duplicate_tail += 1
    return {
        "types": dict(Counter(str(item.get("type") or "unknown") for item in items)),
        "regions": dict(Counter(str(item.get("region") or "unknown") for item in items)),
        "asset_refs": sum(bool(item.get("asset_refs")) for item in items),
        "duplicate_section_tail": duplicate_tail,
    }


def _print_catalog(payload: dict, as_json: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2) if as_json else format_catalog(payload))


def format_catalog(payload: dict) -> str:
    return f"Catalog：{payload.get('status', 'ready')}，论文数量：{payload.get('papers', payload.get('current', {}).get('papers', 0))}"


def handle_index_status(args: argparse.Namespace) -> int:
    from paper_rag.llamaindex.service import index_status

    payload = index_status(Settings.load(args.project_root))
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_catalog(payload.get("data", payload)))
    return 0 if payload.get("status") == "ok" else 1


def handle_index_rebuild(args: argparse.Namespace) -> int:
    from paper_rag.llamaindex.service import rebuild_index

    payload = rebuild_index(Settings.load(args.project_root), mode=args.mode)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_catalog(payload.get("data", payload)))
    return 0 if payload.get("status") == "completed" else 1


def handle_citation_graph(args: argparse.Namespace) -> int:
    """读取 SQLite 本地引用图，不访问正文索引。"""

    from paper_rag.catalog.service import CatalogIndexNotReady, citation_graph, get_metadata

    try:
        settings = Settings.load(args.project_root)
        data = citation_graph(settings, args.paper_id, args.direction, args.depth)
        record = get_metadata(settings, args.paper_id)
        titles = {}
        if record:
            for identifier in (record.paper_id, record.base_id, record.canonical_id):
                if identifier:
                    titles[str(identifier).casefold()] = record.title
        payload = attach_presentation({"status": "ok", "data": data, "warnings": [], "read_only": True}, citation_presentation(data, "graph", title_lookup=titles))
    except CatalogIndexNotReady:
        data = {"paper_id": args.paper_id, "nodes": [], "edges": []}
        payload = attach_presentation({"status": "catalog_not_ready", "data": data, "warnings": [], "read_only": True}, citation_presentation(data, "graph"))
    except ValueError as exc:
        payload = {"status": "invalid_input", "data": {"paper_id": args.paper_id}, "warnings": [str(exc)], "read_only": True}
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_catalog(payload.get("data", payload)))
    return 0 if payload["status"] == "ok" else 1


__all__ = ["add_catalog_parser", "handle_chunks", "handle_citation_graph", "handle_index_rebuild", "handle_index_status", "handle_status", "handle_sync"]
