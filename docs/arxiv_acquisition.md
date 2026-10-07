# ArXiv 论文获取

实现入口为 `paper_rag/acquisition/arxiv.py`。本阶段获取论文最新版本的 Atom 元数据和 PDF，保存本地资产及版本 manifest；不提交 MinerU，不生成 Chunk、Catalog 或向量。

## 1. 输入与版本含义

支持现代 ArXiv ID、带 vN 的版本 ID、旧式分类/编号以及 arxiv.org 的 abs/pdf URL。例如：

```text
1706.03762
1706.03762v7
https://arxiv.org/abs/1706.03762
https://arxiv.org/pdf/1706.03762
```

规范化后拆分 base_id 和 requested_version。服务以 base_id 查询当前最新版本，Atom 的 entry ID/链接用于确定实际 canonical_id。输入 vN 不表示任意下载并长期保留某个历史版本；远端身份以最新元数据为准。比较时优先判断远端是否有升级；没有升级且请求版本低于本地版本时标记 downgrade_rejected 并跳过。

同批次多个输入指向同一 base_id 时去重，不重复下载。ArXiv 不要求 API Key；请求带可配置 User-Agent。网络解析失败、无有效 entry 或非法输入以逐项 failed 返回，其他输入仍继续处理。

## 2. 规划和执行

```text
规范化输入
  → 读取本地 manifest
  → 联网 resolve_latest
  → 比较版本和本地资产存在情况
  → 预览，或按计划下载
  → 临时 PDF + metadata
  → 校验、替换论文目录
  → 更新资产 manifest
```

plan/preview 也访问 ArXiv 元数据接口。dry-run 的含义是不下载 PDF、不写资产，并不是完全离线。

主要计划状态包括 new、upgrade_available、already_latest、downgrade_rejected、duplicate、failed。已有版本相同且 paper.pdf/metadata.json 存在时跳过；manifest 指向的资产缺失时按未完整获取处理。存在检查不重新证明已有 PDF 内容完整，因此最终获取状态要结合文件和后续解析判断。

执行结果包含 items 及 downloaded/skipped/failed 数量。每项有 input、canonical_id、status、PDF/元数据路径、SHA-256 或错误。批次允许部分成功，不因一篇失败撤回已经完成的其他论文。

## 3. 目录与元数据

默认目录：

```text
data/sources/arxiv/
  manifest.jsonl
  1706.03762/
    paper.pdf
    metadata.json
    mineru/                 # 后续解析才生成
```

旧式 ID 的斜杠转换成双下划线作为目录名，逻辑 base_id 仍保留原形式。metadata.json 保存标题、作者、摘要、分类、发布时间、更新时间、base/canonical ID、abs/pdf URL，以及下载来源和源文件哈希等信息；不写入凭证。

manifest.jsonl 按 base_id 保存当前版本和资产位置，不是任务日志；异步任务日志由 MCP JobManager 单独保存。后续 MinerU 使用 metadata 中实际版本和源 PDF 哈希判断是否可复用解析。

## 4. 下载校验与替换边界

PDF 分块下载，限制总大小；检查响应最后落到允许的 ArXiv 地址，并验证 PDF 文件头。默认最大 100 MB，超时 300 秒，不把半下载文件直接当作正式 paper.pdf。

新 PDF 和 metadata 在临时目录准备，旧论文目录通过备份和目录重命名方式替换。替换失败时尝试恢复旧目录，成功后清理备份；manifest 使用临时 JSONL 文件原子替换。目录与 manifest 不是跨文件系统的统一事务。

更新版本时替换的是论文资产目录，不把旧 MinerU 输出作为新版本的有效解析继续复制。因此后续应重新解析新 PDF，再更新 Catalog 和索引；不能认为下载新版本后引用图、Chunk 和向量已自动更新。

## 5. CLI 与 MCP

CLI 使用 RAG_project，参数均为 ASCII ID：

```powershell
conda run -n RAG_project python -X utf8 -m paper_rag acquire arxiv 1706.03762 --dry-run --json
conda run -n RAG_project python -X utf8 -m paper_rag acquire arxiv 1706.03762 2106.09685 --json
```

CLI 同步执行，有任一 failed 时返回非零退出码。未写 --json 时输出人工可读的预览或统计。

MCP 工具是 library_acquire_arxiv，属于默认关闭的 acquisition 组：

```json
{"inputs": ["1706.03762"], "confirm": false}
```

预览返回 confirmation_required；确认输入相同 inputs、confirm=true 后提交 queued 作业，由 library_job_status(job_id) 查看。Job succeeded 还需检查逐项 failed，不等于每篇都成功。启用方式及 MCP 配置见 [mcp.md](mcp.md)。

## 6. 配置

| 配置 | 默认值/含义 |
|---|---|
| ARXIV_DATA_DIR | data/sources/arxiv |
| ARXIV_API_BASE_URL | https://export.arxiv.org/api/query |
| ARXIV_REQUEST_DELAY_SECONDS | 3，元数据请求间隔 |
| ARXIV_TIMEOUT_SECONDS | 30，元数据请求超时 |
| ARXIV_DOWNLOAD_TIMEOUT_SECONDS | 300，PDF 下载超时 |
| ARXIV_MAX_DOWNLOAD_MB | 100，文件大小上限 |
| ARXIV_USER_AGENT | 项目默认标识，可在 .env 配置 |

HTTP JSON 层与 Atom/PDF 流式下载的路径不同，不能把公共 JSON 重试行为假定为所有下载阶段完全一致。版本/资产错误保留在 items.error 中。

完成获取后进入 [MinerU 解析](mineru.md)，随后按需执行 Catalog 同步和 [索引更新](index.md)。
