# MinerU ArXiv 入库

MinerU 入库只处理已经保存到 `data/sources/arxiv/<base_id>/paper.pdf` 的 ArXiv 论文。客户端使用 MinerU v4 签名上传接口：申请上传 URL、PUT 上传本地 PDF、轮询批次结果，再把结果 ZIP 解压到同一论文目录的 `mineru/` 子目录。

先在本地 `.env` 配置 `MINERU_API_KEY` 和 `MINERU_API_BASE_URL`。密钥不作为命令行参数，也不会写入结果 manifest。

预览不会访问 MinerU：

```powershell
python -X utf8 -m paper_rag ingest arxiv 1706.03762 --dry-run --json
```

确认执行时，任务会同步逐项处理；MCP 调用则先使用 `paper_arxiv_ingest(confirm=false)` 预览，再用 `confirm=true` 创建异步任务：

```powershell
python -X utf8 -m paper_rag ingest arxiv 1706.03762 2106.09685 --json
```

每篇论文的 `mineru/manifest.json` 会记录源 PDF SHA-256、ArXiv 版本、MinerU 模型、语言和远端任务批次。源 PDF、metadata 或解析配置变化时会重新解析；相同资产会跳过。解析失败不会替换已有成功结果。

