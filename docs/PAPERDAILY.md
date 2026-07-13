# PaperDaily MVP 使用指南

PaperDaily 是本 Fork 在 PaperFlow 之上增加的本地优先 arXiv 工作流。它面向“按研究话题持续追踪新论文”的场景，当前可完成：

- 按 arXiv 分类、精确短语、关键词、上下文词和负关键词召回论文；
- 正常日报从 arXiv RSS 获取当天新公告（含可选 cross-list），再通过官方 API 补齐元数据；历史补推仍使用官方 API 日期查询；
- 使用真实 Embedding 时加入话题/用户画像语义相似度；配置真实 LLM 后，仅重排基础候选前 N 篇，并用 MMR 控制重复；
- 只对最终推荐生成中文短摘要；未配置真实 LLM 时明确回退到原始英文摘要；
- 将日报输出到终端和本地 Markdown，飞书文本推送可选；
- 记录感兴趣、不相关、稍后阅读、收藏和已读反馈；
- 通过 Codex CLI 或 Claude Code CLI 对 arXiv PDF 生成带证据位置的中文阅读笔记。

PaperDaily 不替代原有 `paperflow` 命令，两套 CLI 共用 PaperFlow 的本地 SQLite 和用户画像。当前是 MVP：尚未加入飞书交互卡片或全文中英对照排版；本地只读优先的 stdio MCP Server 已可用。

## 1. Windows PowerShell 安装

需要 Python 3.10+、Git，以及能够访问 arXiv 的网络。下面以 `D:\PaperFlow` 为例：

```powershell
git clone https://github.com/Cat-blizzard/PaperFlow.git D:\PaperFlow
Set-Location D:\PaperFlow

py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[all]"
```

`.[all]` 包含联网 Provider、论文来源和 PDF 解析依赖。只使用日报主流程时可以先安装最小包；执行 `paperdaily read` 前至少还要安装解析依赖：

```powershell
pip install -e ".[parsing]"
```

若不想激活虚拟环境，本文中的 `paperdaily` 可替换为：

```powershell
.\.venv\Scripts\paperdaily.exe
```

## 2. 初始化与自检

```powershell
paperdaily init
paperdaily doctor
paperdaily status
```

默认初始化结果：

| 内容 | 默认位置 |
| --- | --- |
| PaperDaily 配置 | `D:\PaperFlow\data\paperdaily\config.yaml` |
| PaperFlow/PaperDaily SQLite | `D:\PaperFlow\data\paperflow.db` |
| arXiv 缓存 | `D:\PaperFlow\data\cache\arxiv\` |
| 日报 | `D:\PaperFlow\data\output\digests\` |
| 精读工作区/PDF/结构化结果 | `D:\PaperFlow\data\workspaces\<arxiv-id>\` |
| 最终中文笔记 | `D:\PaperFlow\data\output\notes\<arxiv-id>.md` |

`init` 默认创建 `user_001` 以及一个“具身智能与 VLA”话题。重复执行不会覆盖已有配置；只有显式使用 `--force` 才会重建默认配置。

如需另一份配置，所有主要命令均支持 `--config`：

```powershell
paperdaily init --config D:\PaperFlow\data\paperdaily\lab.yaml --user-id lab_user
paperdaily doctor --config D:\PaperFlow\data\paperdaily\lab.yaml
```

## 3. 配置真实 Embedding 与中文摘要

PaperDaily 的 Provider 选择沿用 PaperFlow 环境变量，并自动读取仓库根目录的 `.env`。详细选项见 [Provider 文档](providers.md) 和 [配置文档](configuration.md)。

### 3.1 必须理解的默认回退

- 未配置真实 Embedding 时使用 `hash`。它不具备语义理解能力，系统仍可按话题规则召回和排序。
- 未配置可用 LLM 凭据时使用 `mock` 回退。日报会保留原始英文摘要并标出未配置状态，不会伪装成中文摘要。
- 真实语义推荐需要把 `PAPERFLOW_EMBED_PROVIDER` 配成 `openai`、`sentence_transformers` 或 `ollama`。
- 真实中文摘要需要把 `PAPERFLOW_LLM_PROVIDER` 配成 `openai`、`anthropic` 或 `ollama`，并配置对应凭据/服务。

中文短摘要是可跨日报复用的“论文事实层”：v2 Prompt 和缓存只读取规范化后的**标题与摘要**。命中话题、推荐原因、用户 ID 和其他论文元数据既不会发送给摘要模型，也不会写入摘要缓存；日报显示的推荐理由始终来自排序层。因此，同一论文的事实摘要可以安全地被不同用户或不同话题复用，而不会混入前一次的个性化上下文。标题或摘要修订会自动使缓存失效。

同一 LLM Provider 还会对基础排序前 `daily.rerank_limit` 篇候选做一次结构化相关性复核。未配置 Provider、启用 `--dry-run` 或将 `daily.llm_rerank_enabled` 设为 `false` 时，系统不会发起该调用，结果保持规则/Embedding 基础排序。

默认重排参数如下；价格仅用于在日报统计中估算成本，不会影响排序：

```yaml
daily:
  rerank_limit: 30
  llm_rerank_enabled: true
  llm_rerank_weight: 0.25
  llm_rerank_max_tokens: 3000
  llm_rerank_input_cost_per_million_tokens: 0.0
  llm_rerank_output_cost_per_million_tokens: 0.0
  arxiv_rss_enabled: true
  arxiv_rss_include_cross_list: true
  arxiv_rss_cache_ttl_minutes: 15
  arxiv_api_id_batch_size: 20
```

配置完成后必须重新运行 `paperdaily doctor`，确认不再显示 `hash embedding` 或 `mock` 警告。

### 3.2 PowerShell 临时配置示例

下面变量只在当前 PowerShell 会话生效；模型名请替换成你的账户或兼容网关实际支持的值：

```powershell
$env:PAPERFLOW_LLM_PROVIDER = "openai"
$env:PAPERFLOW_LLM_MODEL = "<llm-model>"
$env:PAPERFLOW_EMBED_PROVIDER = "openai"
$env:PAPERFLOW_EMBED_MODEL = "<embedding-model>"
$env:PAPERFLOW_LLM_API_KEY = "<llm-api-key>"
$env:PAPERFLOW_EMBED_API_KEY = "<embedding-api-key>"

paperdaily doctor
```

LLM 与 Embedding 的 OpenAI 兼容凭据和端点可以完全分离：LLM 优先读取
`PAPERFLOW_LLM_API_KEY` / `PAPERFLOW_LLM_BASE_URL`，Embedding 优先读取
`PAPERFLOW_EMBED_API_KEY` / `PAPERFLOW_EMBED_BASE_URL`。两者都未设置时，才依次回退到
`PAPERFLOW_OPENAI_*` 和旧版 `OPENAI_*`。这四个 `PAPERFLOW_*` 变量只供日报的 LLM/
Embedding Provider 使用，**不会**传给 Codex CLI 子进程，因此不会改变已登录 Codex 的额度或认证。

DeepSeek 生成中文摘要、独立 BGE-M3 服务做语义推荐的 `.env` 示例：

```env
PAPERFLOW_LLM_PROVIDER=openai
PAPERFLOW_LLM_MODEL=deepseek-v4-flash
PAPERFLOW_LLM_API_KEY=<your-deepseek-api-key>
PAPERFLOW_LLM_BASE_URL=https://api.deepseek.com

PAPERFLOW_EMBED_PROVIDER=openai
PAPERFLOW_EMBED_MODEL=BAAI/bge-m3
PAPERFLOW_EMBED_DIMENSIONS=1024
PAPERFLOW_EMBED_API_KEY=<your-bge-m3-api-key>
PAPERFLOW_EMBED_BASE_URL=https://<your-bge-m3-compatible-endpoint>/v1
```

这里的 BGE-M3 服务必须提供 OpenAI Embeddings 兼容接口；DeepSeek 文本 API 本身不承担
Embedding。此方案不需要设置 `OPENAI_API_KEY`。不要把真实密钥写入 Git 跟踪文件；持久配置可复制
`.env.example` 为被忽略的 `.env`：

```powershell
Copy-Item .env.example .env
notepad .env
```

本地 Embedding 示例：

```powershell
$env:PAPERFLOW_EMBED_PROVIDER = "sentence_transformers"
$env:PAPERFLOW_EMBED_MODEL = "BAAI/bge-m3"
paperdaily doctor
```

首次使用本地模型会下载权重。若只想验证日期、抓取和规则匹配，保留 `hash` 即可。

## 4. 桌面 GUI：arXiv-only 工作流

在仓库根目录启动本地界面：

```powershell
.\.venv\Scripts\paperflow.exe gui --port 8769
```

打开 `http://127.0.0.1:8769`。PaperDaily GUI 只展示 arXiv 日报流程；会议、
期刊、OpenReview、Semantic Scholar 和自定义 RSS 不会参与这个工作流。

右上角的“研究者空间”是本机隔离，而不是网络账户或密码认证：

- 新建一个研究者 ID 后，会创建 `data/paperdaily/users/<user-id>.yaml`；
- 每个研究者有独立的话题、日报、反馈和阅读笔记输出目录；
- SQLite 仍共用一个文件，但所有推荐和反馈都按 `user_id` 隔离；
- 仅在本机使用时不需要密码；若未来部署到局域网或公网，必须在前面增加真正的认证层。

添加研究话题时只需填关键词，例如：

```text
VLA, vision-language-action, embodied AI, robotic manipulation
```

名称可留空，系统会自动生成；默认会搜索 `cs.RO`、`cs.AI`、`cs.CV`、`cs.LG` 和
`cs.CL`。输入 `VLA` 或 `WAM` 时会自动加上机器人语境词；`WAM` 也会自动排除
`wireless access management` 等常见误报。需要时再展开“高级匹配规则”调整分类、
负关键词、每日上限和阈值。

如果日报里的某篇显示“中文摘要生成失败”，点击日报工具栏中的“重试中文摘要”。
该操作只重试当前日报中失败的摘要并更新 Markdown，不会重新抓取 arXiv、重新排序或
覆盖你的反馈。

## 5. 管理研究话题

查看默认话题：

```powershell
paperdaily topic list
paperdaily topic show embodied-vla
```

添加新话题。PowerShell 的续行符是反引号：

```powershell
paperdaily topic add `
  --id robot-world-model `
  --name "机器人世界模型" `
  --description "关注面向机器人规划和动作预测的世界模型" `
  --category cs.RO `
  --category cs.AI `
  --phrase "world model" `
  --keyword "robot planning" `
  --context-keyword robot `
  --context-keyword action `
  --negative-keyword "wireless network" `
  --daily-limit 8
```

删除话题：

```powershell
paperdaily topic remove robot-world-model
```

修改话题时，未给出的字段会保留；重复传入的 `--category`、`--phrase`、`--keyword` 等列表选项会整体替换对应字段：

```powershell
paperdaily topic edit embodied-vla `
  --name "具身智能、VLA 与世界动作模型" `
  --description "关注具身智能中的 VLA、World Action Model 和机器人操作。" `
  --category cs.RO `
  --category cs.AI `
  --keyword "vision-language-action" `
  --keyword "world action model" `
  --daily-limit 10 `
  --minimum-score 0.5

# 清空一个列表字段
paperdaily topic edit embodied-vla --clear-negative-keywords

# 临时停用或重新启用，保留既有配置和历史
paperdaily topic disable embodied-vla
paperdaily topic enable embodied-vla
```

话题字段含义：

| 字段 | 作用 |
| --- | --- |
| `arxiv_categories` | arXiv 第一层范围，例如 `cs.RO`、`cs.AI` |
| `exact_phrases` | 高权重精确短语，例如 `vision-language-action` |
| `keywords` | 普通关键词或模型名 |
| `context_keywords` | 为 `VLA`、`WAM` 等缩写提供语境，降低误报 |
| `negative_keywords` | 明确排除的含义或领域 |
| `daily_limit` | 该话题在最终结果中的配额 |
| `minimum_score` | 规则召回阈值，取值 `0` 到 `1` |

`VLA`、`WAM` 等缩写不要单独作为唯一条件。应同时配置完整短语和机器人语境词。

## 5. 每日检索

建议第一次先做不调用 LLM、不写业务运行记录、watermark 或渠道投递的估算：

`--dry-run` 不会写入推荐、摘要、重排或投递结果；为完成本地初始化和避免重复请求 arXiv，它仍可能创建 SQLite 元信息或更新本地 arXiv 缓存。因此它不是“零文件副作用”模式。

```powershell
paperdaily run --window 7d --dry-run --limit 20
```

### PDF 资源边界

`paperdaily read` 会在调用完整 PDF 解析器之前，用 PyMuPDF 检查页数。默认最多
`100` 页，所有逐页文本或普通解析器输出合计最多 `500000` 个字符。超过任一上限
会安全失败；超过页数的 PDF 不会进入通用解析器，超过文本上限的解析结果不会交给 Agent。

可在本地配置中按需调整；只建议针对已知的长附录论文临时提高：

```yaml
deep_read:
  max_pdf_pages: 100
  max_extracted_text_chars: 500000
```

新的全文解析需要 PyMuPDF，原因是没有它就无法在昂贵的通用解析之前可靠地确认页数。
请安装 `pip install -e ".[parsing]"`；已缓存且未超过文本上限的工作区仍可离线复用。

正式运行：

```powershell
# 按 watermark 自动选择窗口；正常连续运行时处理昨天
paperdaily run

# 明确只处理前一个完整日期
paperdaily run --window yesterday --limit 12

# 指定日期范围，起止日期均包含
paperdaily run --since 2026-07-01 --until 2026-07-07 --limit 20
```

常用开关：

```powershell
# 不调用中文短摘要 Provider
paperdaily run --window yesterday --no-summary

# 不发送飞书等远程渠道；本地 Markdown 仍会生成
paperdaily run --window yesterday --no-push

# 选择输出渠道；--channel 可重复。latest 是当前 arXiv 公告批次。
paperdaily run --window latest --channel terminal --channel markdown

# 无人值守时不等待交互确认
paperdaily run --non-interactive

# 计划任务专用：始终无人值守，并按 catchup 策略自动选择窗口
paperdaily auto

# 默认会排除已处理/反馈过的论文；显式允许重复推荐
paperdaily run --window 7d --include-handled
```

日期窗口按配置中的时区计算，最新目标是“当前 arXiv 公告批次”。正常日报会先读取分类 RSS（默认包含 cross-list），再按 arXiv ID 分批调用官方 API 补齐标题、摘要、作者与分类；不会依赖脆弱的 HTML 页面爬取。若 RSS 尚未更新到当天，任务会停止且**不会推进 watermark**，请在公告更新后重试。`yesterday` 仍可作为 `latest` 的兼容别名。

系统先按话题规则召回，再在真实 Embedding 可用时计算话题和用户画像语义相似度，同时加入反馈、时效性和轻量质量特征。若配置了真实 LLM，系统只对基础排序前 `rerank_limit` 篇进行一次结构化重排，随后仍执行 MMR 与话题配额。重排只读取标题、摘要和订阅话题；缓存命中、未配置 LLM、失败或 dry-run 都会回退到基础排序。

每次正式运行都会生成本地 Markdown。只有该文件成功写入后，运行才会完成并推进 watermark；飞书失败只会产生警告，不会破坏已生成的本地日报。watermark 表示的是**连续完成的日期边界**：如果你先处理较新的 7 天窗口、而更早日期仍有空档，该较新运行会被保留为成功记录，但不会静默越过空档推进 watermark。下一次 `catchup` 会优先给出最早空档；补齐后，系统会自动合并已完成的后续窗口并推进到新的连续边界。

arXiv 查询若达到 `daily.arxiv_max_results` 安全上限，PaperDaily 会将该次运行标记为失败，并且**绝不会推进 watermark**。这避免补推窗口中未抓到的论文被永久跳过。请缩短日期范围（例如从 `30d` 改为 `7d`），或在确认资源与 arXiv 请求量可接受后提高该上限，再重新运行。

## 6. 断更补推

查看遗漏范围和系统建议：

```powershell
paperdaily status
paperdaily catchup --dry-run
```

执行补推：

```powershell
paperdaily catchup --window 7d --limit 30
paperdaily catchup --window 30d --limit 30
paperdaily catchup --window all --limit 30
```

当前策略：

- 第一次运行默认查看最近 7 天；
- 漏 1～2 天时自动补齐全部；
- 漏 3～30 天时交互推荐最近 7 天；
- 超过 30 天时交互推荐最近 30 天；
- 多日窗口未指定 `--limit` 时，默认最多输出 `catchup.max_papers_per_run`（默认 30）篇综合精选。

`catchup` 当前固定禁用飞书远程推送，但仍生成终端和本地 Markdown 输出。它不会按周拆成多份周报；`overflow_mode` 是为后续策略保留的配置项。

## 7. 反馈与历史

```powershell
paperdaily feedback 2607.08182 interested
paperdaily feedback 2607.08182 irrelevant
paperdaily feedback 2607.08182 later
paperdaily feedback 2607.08182 saved
paperdaily feedback 2607.08182 read

paperdaily history --limit 10
paperdaily status
```

反馈先持久化到 PaperDaily 表，再尽力同步到原 PaperFlow 用户画像。后续排序会按命中话题使用这些反馈；失败的旧画像同步不会丢失 PaperDaily 反馈事件。

## 8. 本地阅读笔记

精读成功后，笔记会保存为本地 Markdown，不依赖飞书或网络。下面的命令只读取本地文件：

```powershell
paperdaily notes list
paperdaily notes show 2607.08182

# 可接受带版本号的标准 arXiv 标识；会定位到同一份本地笔记
paperdaily notes show 2607.08182v2
```

若尚未生成笔记，`notes show` 会返回非零退出码并给出预期文件路径。

## 9. Codex / Claude 论文精读

先安装 PDF 解析依赖，再诊断 CLI Provider：

```powershell
pip install -e ".[parsing]"

paperdaily provider list
paperdaily provider doctor
paperdaily provider test codex
paperdaily provider test claude
```

`provider test` 只检查命令和认证就绪状态，不进行模型调用。`auto` 按配置中的 `fallback_order` 选择第一个就绪 Provider，默认顺序为 Codex、Claude。

### 9.1 Codex CLI

先安装 Codex CLI 并完成其登录，或为子进程提供 `OPENAI_API_KEY`。这里的 Key 是 Codex CLI
自己的 OpenAI API Key；日报使用的 `PAPERFLOW_LLM_API_KEY` / `PAPERFLOW_EMBED_API_KEY` 不会
被转交给 Codex。如果可执行文件不在 `PATH`，在 `config.yaml` 中设置：

```yaml
providers:
  default_provider: auto
  fallback_order: [codex, claude]
  codex:
    command: C:\Users\you\AppData\Roaming\npm\codex.cmd
```

运行：

```powershell
paperdaily read 2607.08182 --provider codex
```

当前适配器通过非交互式 `codex exec` 执行，并固定使用：

- `--ignore-user-config`，避免用户级配置改变自动化行为；
- `--ephemeral`，不保留会话；
- `--sandbox read-only`，不授予 Agent 文件写权限；
- JSON Schema 和 `--output-last-message`，只接收结构化结果。

### 9.2 Claude Code CLI

安全的 `--bare` 模式当前明确要求环境中存在 `ANTHROPIC_API_KEY`：

```powershell
$env:ANTHROPIC_API_KEY = "<your-api-key>"
paperdaily provider test claude
paperdaily read 2607.08182 --provider claude
```

Claude 适配器使用 `--bare`、`dontAsk`、`Read` 工具白名单、JSON Schema 和无会话持久化；可在配置中设置单次预算：

```yaml
providers:
  claude:
    command: claude
    max_budget_usd: 3.0
```

### 9.3 可选 API Key 隔离 Home

默认模式保持与已登录的 ChatGPT/Codex CLI 兼容：子进程会使用当前用户的
`CODEX_HOME` / Home 登录状态。因此默认模式**不是**强隔离边界；它适合个人本地
使用，但不应把论文工作区附近放置敏感文件。

如果希望精读子进程不接触真实的 Home、`CODEX_HOME`、`APPDATA`、
`LOCALAPPDATA` 和用户临时目录，请显式启用 API Key 隔离模式：

```yaml
deep_read:
  isolated_home: true
```

或仅对一次运行启用：

```powershell
$env:OPENAI_API_KEY = "<your-api-key>"
paperdaily provider doctor --isolated-home
paperdaily read 2607.08182 --provider codex --isolated-home
```

隔离模式会在这篇论文的 workspace 内创建一个新的、空的临时 Home、`APPDATA`、
`LOCALAPPDATA`、`CODEX_HOME`、Claude 配置目录和临时目录；完成或失败后立即删除。
系统不会复制真实用户目录、既有 CLI 登录资料或配置文件，认证只通过环境中的直接
API Key 传给对应 Provider。不要把 Key 写进 YAML 配置或论文 workspace。

- Codex 隔离模式必须有 `OPENAI_API_KEY`（不能用 `PAPERFLOW_LLM_API_KEY` 代替）；没有 Key 时会明确失败，绝不会退回读取
  ChatGPT / `CODEX_HOME` 登录资料。
- Claude 的 `--bare` 本来就要求 `ANTHROPIC_API_KEY`；隔离模式仍会为其提供新的空
  Home 和配置目录。
- 可以用 `--shared-home` 临时覆盖已配置的 `deep_read.isolated_home: true`。

这仍不是容器或独立系统账户：它隔离的是隐式的用户认证/配置路径。Codex 保持
`--sandbox read-only`，Claude 保持 `--bare` 和仅 `Read` 工具；如需防止进程读取
任意主机路径，应额外使用容器、虚拟机或独立系统账户。

### 9.4 精读安全边界

PaperDaily 主程序负责下载和解析，并把每篇论文放进独立工作目录，Agent 以该目录为当前工作目录。当前保护包括：

- 仅接受 arXiv ID，仅从 `arxiv.org` 允许列表的 HTTPS 公网地址下载 PDF；
- 每次重定向重新校验目标，默认拒绝超过 50 MiB 的 PDF；
- 每条 evidence 都必须有非空、唯一的 ID、主张、短引文和章节；短引文无论是否带页码都必须能在本地 `paper.md` 找到，带页码时还必须能在该物理 PDF 页的 `pages.json` 逐字定位；
- Codex 使用只读沙箱；Claude 仅允许 `Read` 工具；
- 子进程环境使用白名单，不继承无关密钥；启用 `isolated_home` 时还会移除真实用户 Home 与 Provider 配置路径；
- 论文文本被视为不可信输入，输出会在主机端再次通过固定 JSON Schema 校验，不能只依赖 Agent CLI 的结构化输出选项；
- 含数字的实验结论必须引用有效的 evidence 记录，且该 evidence 引文至少包含一个对应的结果/提升数值，否则笔记校验失败；
- Agent 不负责写数据库、推送飞书或执行论文附带代码。

这不是容器或虚拟机级隔离。Codex 的只读沙箱和 Claude 的 `Read` 工具限制主要防止修改与命令执行，不应被视为对本机所有可读文件的绝对隔离；精读时仍应避免在论文工作区或其附近放置敏感材料，高隔离需求请使用独立系统账户或容器。

`read` 会缓存 PDF 和解析结果。重新解析使用：

```powershell
paperdaily read 2607.08182 --provider auto --force-parse --timeout 1800
```

成功后会自动记录 `reading_note` 正反馈，并生成：

```text
D:\PaperFlow\data\workspaces\2607.08182\reading_note.json
D:\PaperFlow\data\output\notes\2607.08182.md
```

## 10. 本地 MCP Server（Codex / Claude Code）

PaperDaily 也可作为一个本地 `stdio` MCP Server 使用。它不是 Web 服务，不监听端口；每个
请求只访问 `--config` 指定配置对应的本地 SQLite 和笔记目录。

安装 MCP 可选依赖后启动：

```powershell
pip install -e ".[mcp]"

paperdaily mcp serve --config D:\PaperFlow\data\paperdaily\config.yaml
```

正常情况下不要在终端里直接手动运行后再输入日志：标准输出由 MCP JSON-RPC 协议独占，应该
由 Codex 或 Claude Code 启动它。

服务只暴露这些受限业务工具：

| 工具 | 类型 | 作用 |
| --- | --- | --- |
| `get_status` | 只读 | 查看本地运行状态和 MCP 能力边界 |
| `list_topics` | 只读 | 列出配置的研究话题 |
| `get_daily_digest` | 只读 | 读取已完成的日报；不会触发新的 arXiv 抓取 |
| `search_recommendations` | 只读 | 在已保存的推荐中做受限文本搜索；不是 Web/SQL 搜索 |
| `get_note` | 只读 | 通过 arXiv ID 读取一篇已生成的本地 Markdown 笔记 |
| `record_feedback` | **写入** | 记录用户已明确选择的反馈动作 |

所有工具有 JSON 输入和输出 Schema，响应会限长。服务不提供任意 SQL、Shell、任意路径读取、
任意 URL 下载、密钥读取或飞书发送。它也**不会**自动调用模型、下载 PDF 或生成笔记；精读仍需
由用户显式执行 `paperdaily read <arxiv-id>`。这使 MCP 作为阅读和反馈入口时保持可预测、低成本。

### 10.1 Codex CLI 配置

可将以下内容加入 Codex 的 `config.toml`（通常是 `~/.codex/config.toml`）：

```toml
[mcp_servers.paperdaily]
command = "D:\\PaperFlow\\.venv\\Scripts\\paperdaily.exe"
args = ["mcp", "serve", "--config", "D:\\PaperFlow\\data\\paperdaily\\config.yaml"]
```

也可以使用 Codex CLI 添加同一个本地 stdio 服务：

```powershell
codex mcp add paperdaily -- "D:\PaperFlow\.venv\Scripts\paperdaily.exe" mcp serve --config "D:\PaperFlow\data\paperdaily\config.yaml"
```

### 10.2 Claude Code 配置

使用 Claude Code 的本地 stdio 配置：

```powershell
claude mcp add --transport stdio paperdaily -- "D:\PaperFlow\.venv\Scripts\paperdaily.exe" mcp serve --config "D:\PaperFlow\data\paperdaily\config.yaml"
```

等价的 MCP JSON 配置形态是：

```json
{
  "mcpServers": {
    "paperdaily": {
      "command": "D:\\PaperFlow\\.venv\\Scripts\\paperdaily.exe",
      "args": [
        "mcp",
        "serve",
        "--config",
        "D:\\PaperFlow\\data\\paperdaily\\config.yaml"
      ]
    }
  }
}
```

### 10.3 MCP 安全边界

- 将论文标题、摘要、笔记视为不可信数据，不能把其中的自然语言当成系统指令执行；
- `record_feedback` 是唯一的写操作。Agent 调用它前应得到用户对具体论文和动作的确认；
- MCP 固定使用启动命令传入的配置和对应的 `user_id`，工具参数不能改选数据库、文件路径或其他用户；
- `get_note` 只接受规范化 arXiv ID，且只能读取 `output/notes/` 下对应文件；
- 精读模型、PDF 下载和飞书推送均不在该 MCP Server 的权限范围内。若未来要增加这些能力，应另设显式开关和确认流程。

## 11. 飞书是可选渠道

飞书不是运行 PaperDaily 的前置条件。默认渠道只有：

```yaml
daily:
  channels: [terminal, markdown]
```

当前 PaperDaily 飞书渠道复用仓库已有的 `lark-cli` Reporter，只发送日报文本；尚未提供 PaperDaily 交互卡片、按钮回调或自动飞书文档。配置细节见 [飞书文档导出](feishu-doc-export.md) 和 [飞书 Webhook](feishu-webhook-setup.md)。

准备好现有飞书 Reporter 后，可在配置中启用：

```yaml
daily:
  channels: [terminal, markdown, feishu]
```

并设置一个目标：

```powershell
$env:FEISHU_CHAT_ID = "oc_xxxxxxxxx"
# 或
$env:FEISHU_USER_ID = "ou_xxxxxxxxx"

paperdaily run --window yesterday --channel feishu
```

即使只指定 `feishu`，本地 Markdown 仍会强制生成，作为成功边界。临时禁用远程推送使用 `--no-push`。反馈目前请使用 `paperdaily feedback`。

## 12. 常见问题

### `doctor` 显示 `hash embedding`

规则匹配仍可用，但没有真实语义相似度。配置 `PAPERFLOW_EMBED_PROVIDER` 和对应模型/服务后重试。

### `doctor` 显示 `mock` 中文摘要

当前日报只会给出原始英文摘要回退。配置真实 `PAPERFLOW_LLM_PROVIDER` 和对应凭据；已有成功摘要按规范化后的标题与摘要源哈希、Prompt、语言、Provider 和模型缓存。

### `yesterday` 没有论文

先确认时区和日期，再用最近 7 天做 dry-run：

```powershell
paperdaily run --window 7d --dry-run --limit 20
```

arXiv 周末或处理周期内没有新论文属于正常情况。

### 精读找不到 PDF 解析器

```powershell
pip install -e ".[parsing]"
paperdaily doctor
```

### 想在定时任务中运行

使用虚拟环境内的绝对路径和 `auto`：

```powershell
D:\PaperFlow\.venv\Scripts\paperdaily.exe auto
```

`auto` 从不等待终端输入，并使用 `catchup` 的非交互推荐策略：短间隔自动补齐，较长间隔默认选最近 7 天或 30 天。它只使用 `daily.channels` 中明确配置的渠道；默认配置是 `terminal, markdown`，不会自动发送飞书。若计划任务绝不应远程推送，可额外加 `--no-push`：

```powershell
D:\PaperFlow\.venv\Scripts\paperdaily.exe auto --no-push
```

先手工运行并确认配置、网络、Provider 和输出目录都正常，再交给 Windows 任务计划程序。
