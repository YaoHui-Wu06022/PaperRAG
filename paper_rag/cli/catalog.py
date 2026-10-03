"""Catalog 同步和状态 CLI。"""

from __future__ import annotations

import argparse
import json

from paper_rag.config import Settings


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


def format_catalog(payload: dict) -> str:
    return f"Catalog：{payload.get('status', 'ready')}，论文数量：{payload.get('papers', payload.get('current', {}).get('papers', 0))}"


__all__ = ["add_catalog_parser", "handle_status", "handle_sync"]
