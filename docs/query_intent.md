# Query 意图分类

Paper RAG 的 `paper_query` 是只读知识查询入口。非空 query 优先调用 Jev System One；只有 Jev 不可用时，才使用少量高置信度短句规则回退。最后从当前 ArXiv Catalog 返回结构化论文上下文。

```text
paper_query(query)
  -> Jev choice/noul
  -> 高置信度短句规则（Jev 故障回退）
  -> QueryIntent
  -> Catalog / Reading Service
```

固定意图：

- `paper_discovery`：查找和筛选论文
- `paper_summary`：总结论文
- `paper_comparison`：比较论文、方法或实验结果
- `paper_content`：回答论文正文问题
- `metadata_lookup`：查询作者、日期、分类和版本
- `clarify`：信息不足，需要补充
- `unsupported`：当前系统不支持

下载、MinerU 解析、任务状态、论文资产状态和索引管理属于 MCP 操作层，使用专用工具，不由 `paper_query` 读取或执行。

当前阶段的 `paper_discovery` 使用显式同步的 SQLite FTS5 元数据索引；索引不存在时返回
`index_not_ready`，查询不会自动写入数据库。`paper_summary`、`paper_comparison` 和
`paper_content` 暂不读取正文，只返回元数据和正文资产可用性提示。

## 配置

```text
JEV_ENABLED=true
JEV_BASE_URL=https://jevmodel.org/v1/systemone
JEV_API_KEY=
JEV_MODEL=jev-1.13.0
JEV_TIMEOUT_SECONDS=15
JEV_RETRY_COUNT=2
JEV_ROUTE_PROBABILITY_THRESHOLD=0.65
```

密钥只能放在本地 `.env` 或环境变量中，不会进入配置表示、日志或查询结果。

## CLI

```powershell
python -m paper_rag query "找出关于 LoRA 的论文" --json
python -m paper_rag query "总结 2106.09685" --json
python -m paper_rag catalog sync --dry-run --json
python -m paper_rag catalog sync --json
```

CLI 和 MCP 共用同一个查询服务；查询不会写入 PDF、MinerU 目录或任务日志。
