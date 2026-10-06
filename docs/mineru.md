# MinerU 解析与入库

本文记录 ArXiv PDF 进入 MinerU、解析结果落盘、区域识别以及引用图输入的实现。MinerU 阶段不生成向量，Catalog 和索引见 [Chunk、Catalog 与索引](chunk.md)。

## 输入与输出

输入是 data/sources/arxiv/<base_id>/paper.pdf。客户端使用 MinerU v4 签名上传接口：申请上传 URL、PUT 上传 PDF、创建或轮询批次、下载 ZIP，并解压到 mineru/。

典型输出包括 content_list.json、full.md、图片资源和 manifest.json。manifest 记录源 PDF SHA-256、ArXiv 版本、MinerU 模型、语言和远端任务批次。源 PDF、metadata 或解析配置未变化时跳过重复解析；失败不会替换已有成功结果。

## CLI 与 MCP

先在 .env 配置 MINERU_API_KEY 和 MINERU_API_BASE_URL。预览不会访问 MinerU：

    python -X utf8 -m paper_rag ingest arxiv 1706.03762 --dry-run --json

确认执行：

    python -X utf8 -m paper_rag ingest arxiv 1706.03762 2106.09685 --json

MCP 先调用 paper_arxiv_ingest(confirm=false) 预览，再使用 confirm=true 创建任务；通过 library_job_status 查询异步状态。

## 区域与章节识别

Catalog 读取 content_list.json，并维护 abstract、content、appendix、reference 和 metadata 区域。Reference 不生成正文证据，但其中的条目会进入 references 表和 citation_edges。

References 后出现的 A、D.1、H.4 等字母编号章节会切换或恢复到 appendix 并保留层级。致谢之后直接出现 A、A1、A.1、A2.1 等附录标题时也会切换到 appendix；致谢后的 Contributions 保持为 content 顶层章节。

## 原始结构保留

解析结果保留章节编号、章节标题、页码、bbox、原始 block index 和 block type。表格、公式、图片、图表和列表不在 MinerU 阶段合并为普通段落，后续由 Chunk 层保持独立类型。

每篇论文的 Catalog 同步会在 mineru/chunks.json 生成检查副本：

    paper-rag catalog chunks --paper-id 2106.09685 --json
    paper-rag catalog chunks --sample 50 --seed 20261005 --json
    paper-rag catalog chunks --sample 50 --type table --json

引用查询入口是 library_citation(mode=references|citations|graph)，服务只返回结构化引用数据。