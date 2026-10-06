# ArXiv 论文获取

本文记录从 ArXiv 获取原始论文资产的实现、目录结构和运行方式。获取阶段只负责元数据与 PDF，不负责 MinerU 解析、Chunk 生成或向量化。

## 处理流程

1. 接收 ArXiv ID，或可解析出 ID 的 abs/pdf URL。
2. 查询 ArXiv Atom 元数据，确定当前版本、标题、作者、分类、发布时间和更新时间。
3. 下载当前版本 PDF。
4. 将 PDF 和元数据写入 data/sources/arxiv/<base_id>/。
5. 生成版本和源文件信息，供后续 MinerU 入库判断是否需要重新解析。

每个论文目录至少包含 paper.pdf、metadata.json 和后续创建的 mineru/。默认总目录为 data/sources/arxiv/，同一 base_id 只保留当前版本。

## CLI 与 MCP

    paper-rag acquire arxiv 1706.03762 --dry-run --json
    paper-rag acquire arxiv 1706.03762 --json

MCP 先使用 paper_arxiv_download(confirm=false) 获取预览，再使用 confirm=true 创建异步下载任务；通过 library_job_status 查询任务状态。预览不下载 PDF，确认后才执行外部请求和本地写入。API 密钥从环境读取，不作为命令行参数，也不写入 metadata。

## 更新与失败处理

同版本 PDF 和 metadata 已存在时跳过重复资产。版本变化、PDF 缺失或元数据不完整时重新获取。下载失败时保留上一份完整资产，不用半成品覆盖现有文件。

完成后进入 [MinerU 解析与入库](mineru.md)。