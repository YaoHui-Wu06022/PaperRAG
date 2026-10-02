# ArXiv 原始资料获取

数据获取工具只接收 ArXiv ID 或 `abs`/`pdf` URL，先查询当前版本的 Atom 元数据，再下载 PDF，最后把完整原始资产交给后续 MinerU 入库流程。

默认资产目录为 `data/sources/arxiv/`。每个 ArXiv 论文只保留一个当前版本，目录中包含 `paper.pdf` 和 `metadata.json`，同级的 `manifest.jsonl` 用于去重和版本比较。

MCP 调用先使用 `paper_arxiv_download(confirm=false)` 获取预览，再用 `confirm=true` 创建异步下载任务；CLI 可使用 `paper-rag acquire arxiv` 或 `--dry-run`。

本项目使用 ArXiv 的开放接口和开放获取内容，感谢 ArXiv 提供开放访问互操作能力：

> Thank you to arXiv for use of its open access interoperability.

