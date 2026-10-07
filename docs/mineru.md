# MinerU PDF 解析与本地结果

实现位于 `paper_rag/ingest/mineru.py`。客户端把已下载 PDF 提交 MinerU v4，验证远端结果并保存在本地。解析阶段不切 Chunk、不更新 SQLite，也不生成 Embedding。

## 1. 输入及解析复用

输入为已存在的 ArXiv 论文目录，需要 paper.pdf 和 metadata.json。可传 ID 或可解析的 ArXiv URL，但实际解析版本以本地 metadata.canonical_id 为准，不另外下载指定历史版本。

计划读取本地 PDF 哈希，检查 mineru/full.md 和 manifest.json 是否存在，并比较 canonical_id、source_sha256、model_version、language。相同则 skipped，PDF、版本或配置变化则重新解析；源文件缺失返回 missing_source/failed 等逐项状态。

当前复用检查主要依赖 full.md 和 manifest，不单独检查 content_list.json 是否仍存在。Catalog 入库有更严格的三个文件和 manifest 检查：若结果目录不完整，即使解析预览显示可跳过，也不能据此认为已有有效 Chunk。应核对原始结果，不能用旧 Catalog 成功状态代替完整性校验。

## 2. 真实远端链路

```text
本地计划
  → POST /file-urls/batch：申请签名上传 URL
  → HTTPS PUT：上传 PDF
  → GET /extract-results/batch/<batch_id>：轮询
  → done 且 full_zip_url 非空
  → 下载 ZIP、解压验证
  → 规范化文件名、写本地 manifest
  → 替换 mineru 结果目录
```

提交参数包含 model_version、language，并开启公式和表格解析。文件项携带 data_id，用于在批次结果中定位；返回必须有 batch_id 和对应的上传 URL。上传使用签名 URL，不把 API Key 写入本地解析文件。

轮询根据 data_id/文件名识别本次结果，done 才接受结果下载地址，failed 返回远端错误；超出总时限停止。默认每 10 秒轮询一次，总等待上限 1800 秒，不是每次 HTTP 请求都等待 1800 秒。

## 3. 本地输出与校验

```text
data/sources/arxiv/<base_id>/
  paper.pdf
  metadata.json
  mineru/
    full.md
    content_list.json
    manifest.json
    images/ 或远端结果中的图片目录
    chunks.json                  # Catalog 同步后生成
```

下载和解压阶段限制 ZIP/解压结果大小，并拒绝不安全成员路径和符号链接。要求识别出唯一、非空的 full.md 和可读取的主 content_list JSON；保留其他原始结果及资源，复制并规范化成上述主文件名。当前大小上限为 200 MiB。

manifest.json 保存 schema_version、source、base_id、canonical_id、source_sha256、task_id、batch_id、model_version、language、state=done、processed_at。它是解析来源清单，不是 LlamaIndex manifest，也不是论文元数据文件。

结果先在临时目录下载、解压和验证，再替换现有 mineru 目录；常规失败保留上一份可用结果并清理临时文件。批次结果含 items、succeeded、skipped、failed，各论文独立完成。进度信息记录申请、上传、等待、下载等动作。

## 4. CLI 与 MCP

先在根目录 .env 配置 MinerU 服务和凭证；不将 Key 放到 CLI 参数。预览只读取本地资产，不访问 MinerU：

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag ingest arxiv 1706.03762 --dry-run --json
conda run -n RAG_project python -X utf8 -m paper_rag ingest arxiv 1706.03762 2106.09685 --json
```

CLI 子命令是 ingest arxiv，不是 ingest mineru；同步执行，failed 不为零时退出码为 1。未写 --json 时输出人工可读统计。

MCP 工具 library_ingest_mineru 属于默认关闭的 ingestion 组：

```json
{"inputs": ["1706.03762"], "confirm": false}
```

confirm=false 返回本地预览；confirm=true 排队解析，使用 library_job_status 查询。worker 返回后的 succeeded 状态不替代批次逐项结果检查。工具组与作业生命周期见 [mcp.md](mcp.md)。

## 5. 配置及超时

| 配置 | 默认值 |
|---|---|
| MINERU_API_KEY | 空，真实提交必需 |
| MINERU_API_BASE_URL | https://mineru.net/api/v4 |
| MINERU_MODEL_VERSION | vlm |
| MINERU_LANGUAGE | en，英文语料解析 |
| MINERU_REQUEST_TIMEOUT_SECONDS | 60 |
| MINERU_UPLOAD_TIMEOUT_SECONDS | 300 |
| MINERU_POLL_INTERVAL_SECONDS | 10 |
| MINERU_POLL_TIMEOUT_SECONDS | 1800 |

配置影响解析复用判断；修改模型/语言后不能把旧结果无条件作为新配置结果使用。JSON API 使用公共 HTTP 客户端；文件上传、ZIP 下载和任务轮询保留专门实现，轮询与 HTTP 重试不是同一个循环。

## 6. 解析后的 Catalog 行为

解析成功后，单独执行：

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag catalog sync --json
conda run -n RAG_project python -X utf8 -m paper_rag catalog chunks --paper-id 1706.03762 --json
```

Catalog 从 content_list 识别 metadata、abstract、content、appendix、reference，维护章节树，生成正文 Chunk/FTS 和可查看 chunks.json。Reference 由独立引用匹配处理，进入 references/citation_edges，不参与正文检索。

表格 HTML、公式 LaTeX、媒体 caption 与资源原样追溯；不是让 MinerU 生成回答。媒体无 caption 与引用未匹配会成为同步 issues，它们不必然意味着整篇入库失败，具体含义见 [chunk.md](chunk.md) 和 [citation.md](citation.md)。

Catalog 同步后若要启用新正文的语义检索，再单独检查并更新 [LlamaIndex/Milvus 索引](index.md)。仅解析成功不会让新论文自动出现在正文向量结果里。
