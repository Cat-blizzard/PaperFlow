# PaperDaily

PaperDaily 是一个本地优先的 arXiv 日报工具。它按研究话题抓取每日公告，保留英文原题，生成中文短摘要，并将日报和反馈保存在本地。它不会下载或解析全文 PDF。

## 安装

```powershell
git clone https://github.com/Cat-blizzard/PaperFlow.git D:\PaperFlow
Set-Location D:\PaperFlow

py -3 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[all]"

Copy-Item .env.example .env
.\.venv\Scripts\paperdaily.exe init
.\.venv\Scripts\paperdaily.exe doctor
```

没有 API Key 也可以验证抓取、话题匹配和 Markdown 输出；但中文摘要会回退为原始英文摘要，`hash` embedding 不会启用语义召回。

## 配置中文摘要

编辑根目录 `.env`，默认可使用 DeepSeek 的 OpenAI 兼容接口：

```env
PAPERFLOW_LLM_PROVIDER=openai
PAPERFLOW_LLM_MODEL=deepseek-chat
PAPERFLOW_LLM_API_KEY=your-deepseek-api-key
PAPERFLOW_LLM_BASE_URL=https://api.deepseek.com
```

DeepSeek 不提供 embedding。需要理解 Abstract 并召回没有字面关键词的论文时，再选择一个 OpenAI Embeddings 兼容服务或本地模型：

```env
PAPERFLOW_EMBED_PROVIDER=openai
PAPERFLOW_EMBED_MODEL=BAAI/bge-m3
PAPERFLOW_EMBED_API_KEY=your-embedding-api-key
PAPERFLOW_EMBED_BASE_URL=https://your-embedding-endpoint/v1
```

语义召回默认阈值为 `0.58`，每次最多补回 30 篇无字面命中的论文。参数位于 PaperDaily YAML 配置的 `daily` 段：

```yaml
semantic_recall_enabled: true
semantic_recall_threshold: 0.58
semantic_recall_limit: 30
```

标题、Abstract 和话题向量缓存在本地 SQLite；论文内容、Provider、模型或维度变化时才会重新生成。

## GUI 工作流

```powershell
Set-Location D:\PaperFlow
.\.venv\Scripts\paperflow.exe gui --port 8769
```

访问 <http://127.0.0.1:8769> 后：

1. 在侧栏新增研究话题，输入关键词即可，例如 `VLA, vision-language-action, embodied AI, robotic manipulation`。
2. 选择日期范围，先点击“预估候选”确认范围和数量。预估不会创建日报或推进进度。
3. 点击“生成日报”。同一日期会复用已有日报，后台任务锁会阻止重复运行。
4. 在卡片流中阅读中文摘要，打开论文或 PDF，记录“感兴趣”“稍后”“不相关”。
5. 需要外部中文阅读时点击“中文阅读”。系统会复制 arXiv ID，并安全打开 `hjfy.top`，不会上传本地数据或自动提交论文。

## CLI

未激活虚拟环境时，将下面的 `paperdaily` 替换为 `.\.venv\Scripts\paperdaily.exe`。

```powershell
# 检查配置、模型和待处理窗口
paperdaily doctor
paperdaily status

# 预估最近公告批次，不生成摘要也不写入日报
paperdaily run --window latest --dry-run --limit 20

# 生成最近公告批次的日报
paperdaily run --window latest

# 补推最近 7 天，补推默认限制数量以免输出过长
paperdaily catchup --window 7d --limit 30

# 管理话题与记录反馈
paperdaily topic list
paperdaily topic show embodied-vla
paperdaily feedback 2607.08974 interested
paperdaily feedback 2607.08974 irrelevant
```

默认文件位置：

| 内容 | 路径 |
| --- | --- |
| 主配置 | `data/paperdaily/config.yaml` |
| 本地数据库 | `data/paperflow.db` |
| Markdown 日报 | `data/output/digests/` |

## arXiv 抓取与匹配

常规日报优先读取 arXiv RSS 公告，并使用官方 API 补充论文元数据；补推按日期窗口使用官方 API 查询。请求有缓存和限速，时间窗口会重叠并用 arXiv ID 去重，因此任务中断后可以安全重跑。

推荐首先按分类和关键词召回。关键词会同时检查标题与 Abstract，并兼容连字符和常见英文单复数；VLA、WAM 等歧义缩写仍需机器人相关语境。配置真实 embedding 后，系统还会比较每个话题描述与分类范围内全部 Abstract 的语义相似度，补回达到阈值但没有字面关键词的论文；负关键词可以阻止错误的语义补召。随后由 DeepSeek 复核靠前候选，并用 MMR 去重。arXiv 没有统一可靠的作者关键词字段，因此系统使用标题、摘要和官方分类作为输入。

## MCP

可选的本地 MCP Server 只提供已存日报、话题、搜索和反馈操作，不抓取新论文、不调用模型，也不读取任意文件：

```powershell
pip install -e ".[mcp]"
paperdaily mcp serve --config D:\PaperFlow\data\paperdaily\config.yaml
```

## 隐私

- 不要提交 `.env`、API Key、Cookie 或本地数据库。
- LLM 只接收标题和摘要；系统不会自动下载或解析 PDF。
- 论文标题、摘要和外部链接均视为不可信输入，不会自动执行论文附带代码。
