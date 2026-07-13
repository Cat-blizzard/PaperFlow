# PaperDaily

一个面向个人研究者的本地优先 arXiv Daily 工具：订阅研究话题，筛选每天新论文，保留英文原标题，生成中文短摘要，并提供安全的外部中文阅读跳转。

这是 [Cat-blizzard/PaperFlow](https://github.com/Cat-blizzard/PaperFlow) 的 PaperDaily 分支。项目基于 [OpenRaiser/PaperFlow](https://github.com/OpenRaiser/PaperFlow) 开发，保留原项目的 MIT License 与上游署名；本 fork 的产品重心是 **arXiv 日报**，而不是会议/期刊聚合、知识 Wiki、用户画像或本地精读系统。

## 能做什么

- 只跟踪 arXiv：常规日报通过 arXiv RSS 获取最新公告，并用官方 API 补全元数据；历史补推使用官方 API 日期查询。
- 按研究话题订阅：分类、短语、关键词、上下文词和负关键词共同筛选，`VLA`、`WAM` 等缩写会做语境消歧。
- 通过 Abstract 语义召回：配置真实 Embedding 后，即使论文没有出现字面关键词，只要摘要概念与话题接近也能进入候选。
- 输出日报卡片：英文原标题保持不变，显示 arXiv 分类、命中话题、中文短摘要、推荐理由和论文/PDF链接。
- 生成中文短摘要：可用 DeepSeek 或其他 OpenAI 兼容文本 API；未配置真实 LLM 时明确回退到英文原摘要。
- 当天日报默认不限数量：所有通过话题规则、且未推送过的论文都会保留；历史补推仍默认限制数量，避免遗漏多天时产生过长列表。
- 控制重复：已推送论文默认去重；重复生成同一日期时会打开已有日报，而不是制造一个新的空日报。需要回看时可显式勾选“包含已推送论文”。
- 记录反馈：感兴趣、不相关、稍后阅读、收藏和已读会影响后续排序。
- 外部中文阅读：每张 GUI 卡片可安全跳转至 `hjfy.top` 的对应 arXiv 页面；点击时仅复制 arXiv ID 到剪贴板，不会自动提交论文内容或本地数据。
- 本地存储：配置、日报与反馈保存在本机 YAML、SQLite 与 Markdown 中；飞书只是可选的后续输出渠道。

## 工作流

```text
arXiv RSS / API
        |
研究话题召回 (分类 + 关键词 + 语境消歧)
        |
规则召回 + Abstract 语义召回 + DeepSeek 重排 + 去重
        |
中文短摘要 / Markdown 日报 / 本地 GUI
        |
感兴趣、不相关、稍后阅读
```

## 快速开始（Windows PowerShell）

需要 Python 3.10+、Git，以及可以访问 arXiv 的网络。

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

`doctor` 应显示当前使用的摘要与 Embedding Provider。首次安装时即使没有 API Key 也可运行，但会使用 `mock` 摘要和 `hash` embedding，只适合验证流程，不能提供真正的中文摘要或语义召回。

## 配置模型

编辑根目录 `.env`。默认模板已经将文本生成指向 DeepSeek 兼容接口；填入 Key 后，日报会生成中文短摘要。

```env
# 中文短摘要与可选 LLM 重排：DeepSeek / OpenAI 兼容 API
PAPERFLOW_LLM_PROVIDER=openai
PAPERFLOW_LLM_MODEL=deepseek-chat
PAPERFLOW_LLM_API_KEY=your-deepseek-api-key
PAPERFLOW_LLM_BASE_URL=https://api.deepseek.com

# 首次使用可保留 hash；它不具备语义理解能力。
PAPERFLOW_EMBED_PROVIDER=hash
```

DeepSeek 文本 API 不提供 embedding。要让系统理解 Abstract 并补回无关键词论文，请另配一个 OpenAI Embeddings 兼容服务，或使用本地模型：

```env
# 方案 A：独立 Embedding API
PAPERFLOW_EMBED_PROVIDER=openai
PAPERFLOW_EMBED_MODEL=BAAI/bge-m3
PAPERFLOW_EMBED_API_KEY=your-embedding-api-key
PAPERFLOW_EMBED_BASE_URL=https://your-embedding-endpoint/v1

# 方案 B：本地模型（首次会下载模型文件）
# PAPERFLOW_EMBED_PROVIDER=sentence_transformers
# PAPERFLOW_EMBED_MODEL=BAAI/bge-m3
```

语义召回参数位于 `data/paperdaily/config.yaml` 或对应研究者空间配置中：

```yaml
daily:
  semantic_recall_enabled: true
  semantic_recall_threshold: 0.58
  semantic_recall_limit: 30
```

语义向量按论文内容、Provider、模型和维度缓存在 SQLite；同一版本重复运行不会再次请求 Embedding API。

保存后重新启动 GUI 或再次执行：

```powershell
.\.venv\Scripts\paperdaily.exe doctor
```

## GUI 使用

```powershell
Set-Location D:\PaperFlow
.\.venv\Scripts\paperflow.exe gui --port 8769
```

打开 <http://127.0.0.1:8769>，日常操作顺序如下：

1. 在左侧“研究话题”点击添加，只填写关键词即可，例如：`VLA, vision-language-action, embodied AI, robotic manipulation`。
2. 在“检索与补推”选择日期范围。首次可先点“预估候选”，它不会写入日报或推进进度。
3. 确认候选后点“生成日报”。同一个日期重复运行会复用已有日报，避免重复推送与空日报。
4. 在日报卡片上打开论文/PDF，或标记“感兴趣”“稍后”“不相关”。
5. 使用“中文阅读”跳转至外部中文阅读页面，或记录反馈以调整后续推荐。

日报卡片会保留论文的英文原标题，中文仅用于摘要和推荐理由。

## CLI 使用

不激活虚拟环境时，将下面的 `paperdaily` 替换为 `.\.venv\Scripts\paperdaily.exe`。

```powershell
# 查看当前话题、模型配置、下次待处理窗口
paperdaily doctor
paperdaily status

# 先预估最近公告批次，不生成中文摘要、不写入正式日报
paperdaily run --window latest --dry-run --limit 20

# 生成最近公告批次的日报
paperdaily run --window latest --limit 12

# 补推最近 7 天，最多输出 30 篇
paperdaily catchup --window 7d --limit 30

# 查看和管理话题
paperdaily topic list
paperdaily topic show embodied-vla

# 记录反馈
paperdaily feedback 2607.08974 interested
paperdaily feedback 2607.08974 irrelevant

```

常用文件位置：

| 内容 | 默认路径 |
| --- | --- |
| 主配置 | `data/paperdaily/config.yaml` |
| 研究者空间配置 | `data/paperdaily/users/<user-id>.yaml` |
| 本地数据库 | `data/paperflow.db` |
| Markdown 日报 | `data/output/digests/` |

## 推荐逻辑

arXiv 不提供统一、可靠的作者关键词字段。因此日报以论文标题、摘要与官方分类为主要输入：

1. 分类决定宽召回范围，例如 `cs.RO`、`cs.AI`、`cs.CV`、`cs.LG`、`cs.CL`。
2. 研究话题中的短语和关键词会同时匹配标题与 Abstract，并兼容连字符及常见英文单复数。
3. 缩写需要机器人相关语境，降低 `VLA`、`WAM` 的误报。
4. 可用真实 embedding 时，会对分类范围内的全部标题和 Abstract 做语义召回，最多补回 30 篇；`hash` 模式不会进行语义召回。
5. DeepSeek 只复核排序靠前的一小批候选，避免把当天全部论文直接交给大模型。
6. MMR 负责降低日报中相似论文的重复度，历史反馈会影响排序。

这意味着“抓取到论文但日报新增为 0”不一定是话题过严：它也可能表示论文已出现在该日期的日报中并被默认去重。GUI 会分别显示抓取、命中、已推送和新增数量。

## 隐私与边界

- 不要将 `.env`、API Key、Cookie 或本地数据库提交到 Git。
- 中文短摘要与 LLM 重排只读取标题和摘要；系统不会自动下载或解析全文 PDF。
- 论文标题、摘要与外部链接都视为不可信输入；系统不会自动执行论文附带代码或将本地数据提交到外部阅读站点。
- 本项目不是多用户在线服务。GUI 中的“研究者空间”仅用于本机隔离不同人的话题和阅读记录；部署到公网前必须自行增加身份认证。

## 开发与测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\paperdaily -q
node --check deployments\desktop\static\desktop.js
```

## 上游与许可证

- Fork: [Cat-blizzard/PaperFlow](https://github.com/Cat-blizzard/PaperFlow)
- Upstream: [OpenRaiser/PaperFlow](https://github.com/OpenRaiser/PaperFlow)
- License: [MIT](LICENSE)
