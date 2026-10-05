# LangChain/LangGraph 改造：目录结构设计 + 学习路线

## Context（为什么做这份文档、做成什么样）

**项目现状**（2026-10-05 盘点，来自对工作区和 git 历史的完整勘查 + 官方文档核实）：

- 旧手写 RAG 管线（PyMuPDF 提取 → 章节感知分块 → BGE-M3 + BM25 混合检索 → 本地 Qwen 生成）已从工作区**全部删除**，但**尚未提交**——`git restore` 随时可找回；`text_clean.py`、`chunk.py`、`hybrid_retriever.py` 等域知识完好在 git 历史（`7b7e164`）里。
- 依赖已重置为仅 `langchain-openai>=1.6.7`；全部新代码只有一个 17 行的 `test_vllm.py`。
- vLLM 已验证跑通：`ChatOpenAI(model='/model', base_url='http://localhost:8000/v1', api_key='EMPTY')` → Qwen2.5-7B-AWQ，服务端 `--max-model-len 4096`。
  ⚠️ **4096 token 总上下文是项目最硬的约束**：分块大小、top-k、状态设计都要由它推导（Part 1.4 展开）。
- `src/`、`tests/`、`scripts/`、`configs/`、`outputs/` 均为空目录——白纸一张，正好从命名开始设计。

**用户目标**（本次明确）：

1. 一份 **LangChain / LangGraph 学习路线**——面向求职（JD 高频：LCEL、Retriever、LangGraph、Agent、LangSmith）+ 论文（手册问答 RAG → physics-grounded RUL 预测）。
2. 一份 **目录结构设计**——重视命名，每个名字要能说出理由。
3. **自己动手写代码**——本文档是"地图"不是"代做"；我不实施，按阶段自学的产出由自己验收。

**既定决策**：旧管线删除、全面转 LangChain（git 兜底）；Embedding 选型暂缓但结构上留好可替换边界；边学边在本项目实践。

**文档结构**：Part 1 目录结构设计 → Part 2 学习路线（7 个阶段）→ Part 3 注意事项。

---

# Part 1｜目录结构设计

## 1.1 命名原则（每一条都决定了下面树里的某个名字）

| # | 原则 | 说明 | 反例 |
|---|---|---|---|
| 1 | 包名 snake_case、可导入 | 仓库名用连字符，Python 包用下划线 | — |
| 2 | src/ 布局 | `src/ragnostics/`：导入永远是 `ragnostics.*`，不会误导入 cwd 杂项；与 LangGraph 官方模板一致 | 平铺 `ragnostics/` |
| 3 | 复数 = 同类多个实例；单数 = 一个概念/边界 | `configs/ scripts/ prompts/ graphs/` 复数；`data/ artifacts/ outputs/ src/` 单数 | `utils/`、`helpers/` |
| 4 | 模块是名词，脚本是动词短语 | `chunking.py`、`fusion.py` 是"东西"；`build_index.py`、`ask.py` 是"动作" | `do_stuff.py` |
| 5 | 禁止 `utils.py` / `common/` 垃圾桶 | 用领域名词命名：`io.py`、`text.py`、`budget.py`（旧项目把 `text_clean.py`、`io.py` 写成具体名词，这点做对了） | 旧 `src/rag/common/` 本身就是反例 |
| 6 | Prompt 是数据不是代码 | 长 prompt 放 `prompts/*.md` + 同名 `.meta.yaml`（版本/目标模型/修改说明）；短结构串才留代码 | 40 行 f-string 散落在各个节点里 |
| 7 | `configs/`（复数目录）vs `config.py`（单数模块） | 目录装多个 YAML；加载器是一个单数名词模块 | — |
| 8 | data / artifacts / outputs 三分离 | `data/` = 你没创造的输入（代码只读）；`artifacts/` = 可再生的构建产物（gitignore，可 `rm -rf`）；`outputs/` = 一次运行给人的产物（报告/图，值得提交） | 旧项目把索引和报告都塞 `outputs/`，导致删不敢删、交不想交 |
| 9 | `.env` 在仓库根 | 工具链默认位置（`langgraph.json` 的 `"env": ".env"` 就找这里） | 塞进 `configs/` 会让工具找不到 |
| 10 | 测试按"层"分，不按文件 1:1 镜像 | `tests/unit/`（无 GPU/网络/模型下载）+ `tests/integration/`（打 vLLM，默认跳过） | 逐目录镜像 src/ 属于过度工程 |
| 11 | 图/状态的命名对齐官方模板 | `state.py` `context.py` `prompts.py` `tools.py` `nodes.py`——LangGraph 工具链和官方文档同一套词汇 | 自创 `my_state_v2.py` |
| 12 | 超过 ~300 行或 ~5 个公开单元，才把模块升级为包 | 不要预建 12 个空目录 | 提前造 `nodes/` 包再放 1 个函数 |

## 1.2 推荐目录树（每个条目都给了存在理由）

```
A320-HX-RAG-Prognostics/
├── .env                          # vLLM 地址等本地配置。GITIGNORED，绝不提交
├── .env.example                  # 提交的模板：VLLM_BASE_URL / LANGSMITH_API_KEY / RAGNOSTICS_*
├── .gitignore                    # 先于一切创建：.env、.venv/、artifacts/、__pycache__/
├── .python-version               # "3.14"（uv 读取）
├── pyproject.toml                # 依赖 + hatchling 构建 + ruff/pytest 配置
├── uv.lock                       # 提交：论文的可复现性凭证
├── Makefile                      # 任务别名：make ingest / index / ask / eval
├── README.md                     # 论文摘要 + quickstart + 目录说明
├── langgraph.json                # 阶段 5 才加：LangGraph/LangSmith 部署入口
│
├── configs/                      # 声明式配置（人工维护、提交进 git）
│   ├── default.yaml              # 所有可调参数：chunker / embeddings / retrieval / generation 预算
│   ├── corpus.yaml               # 手册登记表（原 manual_registry.yaml 迁移过来）
│   └── eval.yaml                 # 评估：数据集、评估器、实验标签
│
├── data/                         # 只放"输入"：代码只读，绝不写入
│   └── manuals/                  # 两本 CMM PDF，OEM 交付原样
│       ├── 21-52-11-R.pdf
│       └── 21-52-12-R.pdf
│
├── artifacts/                    # 派生的、可再生的构建状态。GITIGNORED，可随时整目录删掉重建
│   ├── pages/                    # 抽取结果：每本手册一个 JSONL
│   ├── chunks/                   # 分块结果 —— 与 embedding 模型无关（这是关键，见 1.5）
│   └── indexes/
│       └── <embedding-slug>/     # ← embedding 标识写进路径：换模型 = 换目录 = 天然隔离
│           ├── MANIFEST.json     #   embedding id/dim、chunker 与语料哈希、库版本、构建日期
│           ├── bm25.pkl          #   词法索引（与向量同放，保证同一 build 成对出现）
│           └── chroma/           #   向量库 persist_directory
│
├── outputs/                      # 一次运行给人的产物（提交进 git，作为论文记录）
│   ├── eval/                     # 评估报告：eval-<时间戳>.csv + 摘要 JSON
│   ├── figures/                  # 论文插图
│   └── sample_answers/           # 带出处溯源的高质量问答样例
│
├── prompts/                      # Prompt 模板：版本化的"数据"，不是代码
│   ├── rag_answer.md             #   有据可依回答的 system prompt
│   ├── rag_answer.meta.yaml      #   版本 / 目标模型 / 作者 / 修改说明
│   └── query_rewrite.md          #   阶段 5 用（查询改写）
│
├── scripts/                      # 薄入口脚本：动词短语命名，绝不被 src 导入
│   ├── ingest_manuals.py         #   corpus.yaml → artifacts/pages + artifacts/chunks
│   ├── build_index.py            #   chunks → artifacts/indexes/<slug>/
│   ├── ask.py                    #   单次提问，手工体验/调试用
│   └── run_eval.py               #   eval.yaml → outputs/eval/
│
├── src/
│   └── ragnostics/               # 唯一可导入的包（snake_case，无连字符）
│       ├── __init__.py           #   只导出 __version__，保持空
│       ├── config.py             #   pydantic Settings：合并 .env + configs/default.yaml
│       ├── paths.py              #   解析 data/ artifacts/ outputs/ 根路径（无 I/O、无副作用）
│       ├── io.py                 #   JSONL 读写（承接旧 common/io.py 的职责）
│       ├── text.py               #   文本清洗（承接旧 common/text_clean.py 的水印/页眉正则）
│       ├── cli.py                #   Typer 入口：ingest | index | ask | eval
│       │
│       ├── ingestion/            # 边界：PDF 字节 → List[Document]，不知道向量的存在
│       │   ├── pdf.py            #   PyMuPDF 抽页 → Document（保留页码等元数据）
│       │   ├── layout.py         #   ATA 章节/页码 block 识别（旧 chunk.py 的 SECTION 逻辑）
│       │   └── chunking.py       #   章节感知分块（不是无脑 RecursiveCharacterTextSplitter）
│       │
│       ├── indexing/             # 边界：List[Document] → 可检索索引（建库/加载）
│       │   ├── embeddings.py     #   build_embeddings(cfg)：全项目唯一提到具体 embedding 的文件
│       │   ├── vector_store.py   #   Chroma 建/载 + 写 MANIFEST.json
│       │   └── bm25.py           #   rank_bm25 索引（自研：community 包已 sunset）
│       │
│       ├── retrieval/            # 边界：query → 排好序的 List[Document]
│       │   ├── dense.py          #   向量检索器
│       │   ├── lexical.py        #   BM25 检索器
│       │   ├── fusion.py         #   RRF 融合（纯函数，全项目最好测的代码）
│       │   └── hybrid.py         #   HybridRetriever(BaseRetriever)：组合 dense + lexical + fusion
│       │
│       ├── generation/           # 边界：(问题, 上下文) → 答案
│       │   ├── models.py         #   build_chat_model(cfg)：vLLM 的 ChatOpenAI 工厂
│       │   ├── prompts.py        #   加载 prompts/*.md + meta.yaml 并格式化
│       │   ├── budget.py         #   4096 token 预算：塞几个 chunk、超了怎么截（见 1.4）
│       │   └── tokenization.py   #   Qwen tokenizer 包装（不能用 tiktoken，见 1.4）
│       │
│       ├── rag/                  # 边界：组装好的两步 RAG（"被装起来的东西"）
│       │   └── chain.py          #   retrieve → budget → generate
│       │
│       ├── graphs/               # LangGraph 工作流（复数：rag_agent 之后还有 prognostics）
│       │   ├── state.py          #   InputState + State（对齐官方模板的分层写法）
│       │   ├── context.py        #   运行时 Context：model / k / prompt 版本，按次注入
│       │   ├── nodes.py          #   节点函数（>300 行或 >5 个节点再升级为 nodes/ 包）
│       │   └── rag_agent.py      #   构图 + compile，模块级导出 graph
│       │
│       └── evaluation/           # 边界：跑一次 → 出分数
│           ├── datasets.py       #   金标问答集（LangSmith dataset 建/载）
│           └── evaluators.py     #   检索命中率、答案有据性（LLM-judge）、答案正确性
│
├── tests/                        # 按"层"组织，不按文件 1:1 镜像 src/
│   ├── conftest.py               #   共享 fixture：迷你 chunk 集、假模型、临时 artifacts 目录
│   ├── fixtures/
│   │   └── a320_pages.jsonl      #   小样本语料：测试永不依赖真实 PDF
│   ├── unit/                     #   快、无网络、无 GPU、无模型下载
│   │   ├── test_chunking.py
│   │   ├── test_fusion.py
│   │   ├── test_budget.py
│   │   └── test_prompts.py
│   └── integration/              #   打本地 vLLM；默认跳过（CI/慢机器友好）
│       ├── test_retrieval.py
│       └── test_graph.py
│
├── notebooks/                    # 探索用。可以 import ragnostics；反向绝不允许
│   └── 01_chunk_size_exploration.ipynb
│
└── docs/                         # 论文向文档（与代码 README 分开）
    ├── design.md                 #   本文档改写
    └── decisions/                #   ADR：每个不可逆决定一页短记录（embedding、chunker、agentic 与否）
```

## 1.3 几个关键取舍的理由

| 名字 | 为什么是它，不是别的 |
|---|---|
| `ragnostics` | 取自项目自己的标语 "RAGnostics"：短、唯一、可导入；RUL 落地后名字依然成立（RAG + prog**nostics**）。备选 `a320_hx_rag`：更描述性但更长，且等预测模型成为主角时会显得别扭 |
| `graphs/`（复数） | 你会有 `rag_agent` 和后来的 `prognostics` 两个图；单数 `graph/` 会在最难受的时候被迫改名 |
| `graphs/context.py` | 这是官方模板**当前**的名字（旧的 `configuration.py` 是遗留叫法）；它装的是"每次调用注入"的值，"context"也更准确 |
| `generation/budget.py` | "budget" 是 4096 约束的领域词。叫 `utils.py` 会把这个项目最重要的约束藏起来 |
| `artifacts/indexes/<embedding-slug>/` | 索引只对建它的模型有效。把模型名写进路径后，"配置和索引不匹配"从静默错误变成"目录不存在" |
| `prompts/` 放仓库根 | Prompt 会被独立评估、版本化、被论文引用；放在包外便于 grep 和 diff，包里不留长字符串 |
| `configs/corpus.yaml` | 手册登记表是人工维护、入版本控制的——所以是配置；`data/` 回答"字节在哪"，它回答"要收录什么" |
| `tests/unit` + `tests/integration` | 真正重要的切分：单测必须无 GPU、无网络、不下载模型。官方模板叫 `unit_tests/`/`integration_tests/`，我们用短名（目录已在 `tests/` 下，`_tests` 后缀冗余），且短名直接对应 pytest marker `-m unit`。想和模板逐字一致就改回长名，别的设计不受影响 |

## 1.4 4096 上下文的 6 个结构性后果（项目最硬约束）

**预算算式**：4096 = system prompt(~250) + 问题(~60) + 检索上下文 + 生成输出(预留 ~512)。
→ 留给上下文 **≈ 3200 token ≈ 2300 英文词**；k=5 时每个 chunk **≈ 650 token ≈ 2000–2600 字符**。
→ 旧项目 `max_chars=1200` 的起点**正好安全**——但必须写成 `configs/default.yaml` 里**带推导注释**的参数，而不是部落知识。

1. **`budget.py` 是一等模块**：入参（system prompt、排好序的 chunks、问题、输出预留），出参（保留几个、尾部怎么截），放不下就清晰报错。所有生成节点必须走它，否则某个啰嗦 chunk 会在演示中途炸掉。
2. **token 计数必须用 Qwen 自己的 tokenizer，不能用 tiktoken**：`ChatOpenAI` 内部对未知模型回退到 `cl100k_base`，那不是 Qwen2.5 的词表。在满是件号（`21-52-11`、`753`）、扭矩值的技术文本上误差不小。`tokenization.py` 懒加载 `AutoTokenizer`（与服务端同一个），`budget.py` 只信它。
3. **chunk 大小是"推导值"**：把上面的算式当注释写在 `configs/default.yaml` 的 `chunk_size`/`k` 旁边；将来服务端升 `max_model_len`，改一处注释重新推导。
4. **LangGraph state 里不放 chunk 正文**：checkpointer 每步都序列化整个 state，几千 token 的上下文 × 多步 × 多线程 = 把检查点存成 blob 库。state 只放 `question`、`chunk_ids`、`scores`、最终答案；正文在生成节点里现取。（官方模板注释里的 `retrieved_documents: List[Document]` 在这个约束下是陷阱——存引用，不存载荷。）
5. **`max_model_len` 是配置，不是字面量**：Qwen2.5-7B 原生支持 32k，4096 是**我们的部署选择**——答辩必被问，所以配置里要写清楚，升级 = 服务端重启 + 改一个配置值。
6. **流式不是可选项**：每次调用 ~3k token 上下文，不流式体验等于坏了。阶段 1 就把 `stream()` 接进 CLI，之后每一阶段保持可用（这也正是 JD 里的 "streaming"）。

## 1.5 Embedding 延后决策的安全边界（三条规则）

选型推迟到阶段 2–3，但**磁盘布局现在就把边界焊死**：

1. **唯一接缝**：`indexing/embeddings.py` 暴露 `build_embeddings(cfg) -> Embeddings`（返回 `langchain_core` 的接口，下游天然模型无关）。除它之外，任何文件不得出现具体模型名/类名。
2. **索引身份在磁盘上**：`artifacts/indexes/<slug>/MANIFEST.json` 记录 embedding id、维度、归一化、chunker 哈希、语料哈希、库版本。加载时断言与配置一致，不一致就喊"index built for X, configured for Y — re-run build_index"。**这是整个设计里性价比最高的一个决定。**
3. **chunk 与 index 分离**：分块是 chunker 的属性，与 embedding 无关——`artifacts/chunks/` 放在 slug 目录**外**。"试三个 embedding 做对比实验" = 切一次、嵌入三次（这本来就是论文的一组消融实验）。

---

# Part 2｜LangChain / LangGraph 学习路线（7 阶段，约 10–12 周 @ 每周 10–15h）

## 0. 先立参照系：2026-10 生态现状（已对照官方文档核实）

- **主线 = LangChain 1.x + LangGraph 1.x**：`langchain`(1.4.x) / `langgraph`(1.2.x) / `langchain-core` / `langsmith`。Agent 的新标准是 `langchain.agents.create_agent`（`create_react_agent` 已废弃、`AgentExecutor` 是 legacy）。
- ⚠️ **`langchain-community` 已于 2026-05 sunset（仓库归档）、LangServe 已归档**。网上大量教程是 0.x 时代的，`from langchain.retrievers import ...` 那套导入路径已经不对——**照教程敲代码前先看版本**。
- **LCEL 还在**（`Runnable`、`|`、`RunnableParallel`…），但 1.x 官方文档里**已经没有 LCEL 教程页**了，只能从 API reference 和示例代码里学——这不是你搜不到，是官方就那样。
- **没有官方 PDF loader**：官方 RAG 教程自己手写 `pypdf`。你们原来的 PyMuPDF 方案不是落后，是正路。
- **`BM25Retriever` 失去了维护（随 community 一起 sunset）**：正确做法是**自己写**（`rank_bm25` + RRF 融合，共约 40 行）——巧的是，你删掉的 `hybrid_retriever.py` 就是现成参考实现，`git show HEAD:src/rag/retrieval/hybrid_retriever.py` 可看。
- **import 速查表**（学习时对照，避免被旧教程带偏）：

| 能力 | 装哪个包 | 导入 |
|---|---|---|
| vLLM 聊天模型 | `langchain-openai` | `from langchain_openai import ChatOpenAI` |
| Agent 工厂 | `langchain` | `from langchain.agents import create_agent` |
| 工具装饰器 / 消息 | `langchain` | `from langchain.tools import tool` / `from langchain.messages import ...` |
| Document / Runnable / BaseRetriever | `langchain-core` | `from langchain_core.documents import Document` |
| 文本分块 | `langchain-text-splitters` | `from langchain_text_splitters import ...` |
| Embeddings（通用 HF 封装） | `langchain-huggingface` | `from langchain_huggingface import HuggingFaceEmbeddings` |
| Chroma 向量库 | `langchain-chroma` | `from langchain_chroma import Chroma` |
| LangGraph | `langgraph` | `from langgraph.graph import StateGraph, START, END` |
| 追踪/评估 | `langsmith` | `from langsmith import Client, evaluate` |
| BM25 / RRF | `rank-bm25`（自己写 wrapper） | —（不依赖 classic/community） |

## 阶段 0（半天）：清理与基线

- **做**：建 `.gitignore`（先于一切！必须含 `.env`、`artifacts/`）；把当前"删除全部旧管线"的状态**提交**（这是重构的起点，别和新代码混成一个大 diff）；把 `test_vllm.py` 移到 `scripts/ask.py` 的雏形（硬编码的值挪进 `.env` 和 `configs/default.yaml`）。
- **pyproject**：加 `[build-system]`(hatchling) + `[tool.hatch.build.targets.wheel] packages = ["src/ragnostics"]`——src 布局不做 editable install 装不上。
- **验收**：`git status` 干净；`uv run python -c "import ragnostics"` 不报错。

## 阶段 1（~1 周）：模型、消息、Prompt、Runnable

- **学**：`ChatOpenAI` / `init_chat_model`；四类消息（System/Human/AI/Tool）；`invoke` / `stream` / `batch`；Prompt 模板；`|` 管道；`RunnableParallel` / `RunnablePassthrough` / `RunnableLambda`；`with_structured_output()`。
- **读**：[models](https://docs.langchain.com/oss/python/langchain/models) · [messages](https://docs.langchain.com/oss/python/langchain/messages) · [streaming](https://docs.langchain.com/oss/python/langchain/streaming) · [structured-output](https://docs.langchain.com/oss/python/langchain/structured-output) · [Runnable API 参考](https://reference.langchain.com/python/langchain-core/runnables/base/Runnable)
- **建**（对应 Part 1 树）：`generation/models.py`、`generation/prompts.py`、`generation/tokenization.py`、`prompts/rag_answer.md`、`configs/default.yaml`、`scripts/ask.py`（含 `--stream`）。
- **验收**：`ask.py "..."` **完全无检索**也能得到回答；`--stream` 逐 token 打印；LangSmith trace 里能看到完整格式化后的 prompt；`with_structured_output` 返回校验过的 pydantic 对象。
- ⚠️ vLLM + `ChatOpenAI` 官方给的警告：它是照 OpenAI 规范实现的，第三方扩展字段（如 `reasoning_content`）不保证解析。结构化输出用 `with_structured_output(method="function_calling")`，别用 provider 原生 JSON schema。

## 阶段 2（~2 周）：RAG 组件——加载、分块、嵌入、存储、检索

- **学**：`Document` 契约（`page_content` + `metadata`）；`RecursiveCharacterTextSplitter`（以及为什么你**不**用它——章节感知分块要用自定义切分器）；`Embeddings` 接口；`Chroma` + `persist_directory`；`as_retriever()` 的三种 `search_type`；metadata 过滤。
- **读**：[knowledge-base 教程](https://docs.langchain.com/oss/python/langchain/knowledge-base)（主线）· [splitters](https://docs.langchain.com/oss/python/integrations/splitters) · [Chroma](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) · [retrieval](https://docs.langchain.com/oss/python/deepagents/retrieval)（看概念即可，Deep Agents 对 2 本 PDF 是杀鸡用牛刀）
- **建**：`ingestion/pdf.py`（PyMuPDF）、`ingestion/layout.py`（ATA 章节识别——把旧 `chunk.py` 的 `SECTION_KEYWORDS` 逻辑移植成 "Document → Document" 的切分器）、`ingestion/chunking.py`、`indexing/embeddings.py`（先用任意通用方案把管线跑通，选型见阶段 3）、`indexing/vector_store.py`（写 `MANIFEST.json`）、`scripts/ingest_manuals.py`、`scripts/build_index.py`。
- **验收**：两本手册产出的 chunk 数可解释；每个 chunk 的 metadata 带 `manual_id/ata/section/manual_page_start/end`；`similarity_search("heat exchanger leak test procedure", k=5)` 返回的段落在 `TESTING AND FAULT ISOLATION` 章节里；`MANIFEST.json` 的维度 == `len(embed.embed_query("x"))`。
- **排雷（一次性）**：在本阶段确认 Python 3.14 下整条依赖链装得上（torch 已有 cp314 轮子；注意 `langchain-huggingface[full]` 对 sentence-transformers 有版本上界）。风险现在排掉，别拖到第三个月。

## 阶段 3（~1.5 周）：自定义 Retriever、混合检索、RRF

- **学**：子类化 `BaseRetriever`（实现 `_get_relevant_documents` / `_aget_relevant_documents`）；BM25 的 k1/b 参数含义；RRF 为什么对排名融合有效；为什么件号查询靠 BM25、同义改写靠向量。
- **读**：[retrievers 集成页](https://docs.langchain.com/oss/python/integrations/retrievers/) · [BaseRetriever 参考](https://reference.langchain.com/python/langchain-core/retrievers/BaseRetriever) · 参考实现：你 git 里的 `hybrid_retriever.py` + `bm25_index.py`（RRF 就是那个 1/(k+rank) 求和）
- **建**：`indexing/bm25.py`、`retrieval/dense.py`、`retrieval/lexical.py`、`retrieval/fusion.py`（纯 RRF 函数）、`retrieval/hybrid.py`。
- **决定（写进 `docs/decisions/`）**：手写 BM25 + RRF，而不是依赖 `langchain-classic` 的 `EnsembleRetriever` + 已 sunset 的 community `BM25Retriever`。理由：BM25 路径已无人维护、RRF 纯函数最好测、而且"讲清你的 retriever 怎么工作"是你最好的面试答案之一。
- **验收**：手写 10 个测试问题的查询集，出一张 dense vs BM25 vs hybrid 的 top-5 命中表（论文材料）；找到"件号查询 BM25 赢、改写查询 dense 赢"各一例；在配置里换 embedding 重跑 `build_index.py`，确认新索引落在**另一个** slug 目录、旧索引仍能按旧配置加载。
- **顺带（论文消融实验 #1）**：到这里再做 embedding 对比（2–3 个候选，用同一份 chunks），别提前纠结选型。

## 阶段 4（~2 周）：LangGraph 基础——状态、节点、边、持久化、流式

- **学**：`StateGraph`；reducer/注解模型（`Annotated[..., add_messages]`）；`START`/`END`；条件边；`input_schema` vs `State` 分层；`Runtime[Context]` 按次注入；`InMemorySaver` + `thread_id`；`stream()` 的多种 `stream_mode`；`get_graph().draw_mermaid()`。
- **读**：[thinking-in-langgraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph) · [graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api) · [use-graph-api](https://docs.langchain.com/oss/python/langgraph/use-graph-api) · [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) · [streaming](https://docs.langchain.com/oss/python/langgraph/streaming) · [application-structure](https://docs.langchain.com/oss/python/langgraph/application-structure)
- **建**：`graphs/state.py`、`graphs/context.py`、`graphs/nodes.py`、`graphs/rag_agent.py`。节点：`retrieve → budget → generate`，**至少一条条件边**（如"没有 chunk 过阈值 → refuse 节点"）。
- **验收**：`graph.invoke(...)` 返回有出处的回答；`graph.stream(...)` 能逐节点看到更新且你说得出每一步是哪个节点产的；`draw_mermaid()` 和你的白板草图一致；同一 `thread_id` 第二次调用记得上一轮（checkpointer 生效）；LangSmith trace 渲染出**图的形状**而不只是线性链。
- 💡 学习方法提醒：这条路线是**自底向上**（先原语，后抽象），和官方文档"先 `create_agent`、需要时再下沉"的默认相反——对求职面试的理解深度，这个顺序更值。

## 阶段 5（~2 周）：Agent、工具、`create_agent`

- **学**：`@tool` 与"docstring 即 schema"；工具绑定；ReAct 循环；`create_agent(model, tools, system_prompt=...)`；middleware（`before_model`/`after_model`/`wrap_model_call`）；内置 `SummarizationMiddleware` / `HumanInTheLoopMiddleware`；interrupts 人机协同；`langgraph.json` + `langgraph dev` + Studio。
- **读**：[agents](https://docs.langchain.com/oss/python/langchain/agents) · [tools](https://docs.langchain.com/oss/python/langchain/tools) · [middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) · [agentic-rag 教程](https://docs.langchain.com/oss/python/langgraph/agentic-rag) · [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)（读懂"为什么 `create_react_agent` 没了"）
- **建**：把 `retrieval/hybrid.py` 包成 `retrieve_manual` 工具（`graphs/tools.py`）；一个自己决定**何时**检索的 agent；`langgraph.json`；答案步骤加 HITL 中断。
- **验收**：问"什么是热交换器"时 agent **不**调用检索工具（trace 里无 tool call），问"21-52-11 泄漏测试压力是多少"时调用；`langgraph dev` 起得来、Studio 里能看/跑你的图；中断能暂停并批准恢复；`stream(..., stream_mode="messages")` 在 agent 循环里仍逐 token 出答案。
- ⚠️ **提前做**：vLLM 现在没开工具调用。本阶段前需重启服务加 `--enable-auto-tool-choice --tool-call-parser hermes`（Qwen2.5 系列用 hermes 解析器）。7B 模型的工具调用能力有限——工具描述写清楚、参数用简单类型（str/int），别用嵌套结构。

## 阶段 6（~2–3 周）：评估、LangSmith、以及 Prognostics 扩展

- **学**：LangSmith `Client.create_dataset` / `create_example`；`evaluate()` + 自定义 evaluator；LLM-as-judge 评分卡；检索指标（hit-rate、MRR、context precision/recall）vs 生成指标（正确性、有据性、相关性）；数据集版本化；`GenericFakeChatModel` + `InMemorySaver` 做无网单测。
- **读**：[RAG 评估教程](https://docs.langchain.com/langsmith/evaluate-rag-tutorial) · [evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) · [evaluators](https://docs.langchain.com/langsmith/evaluators) · [单元测试](https://docs.langchain.com/oss/python/langchain/test/unit-testing) · [集成测试](https://docs.langchain.com/oss/python/langchain/test/integration-testing)
  ⚠️ 官方 RAG 评估走 **LangSmith evaluators**；RAGAS 与 LangChain 1.x 的兼容性未核实——想用就当实验做，留好退路。
- **建**：`evaluation/datasets.py`（你手写的 30–50 条金标问答，覆盖 190 页语料足够）、`evaluation/evaluators.py`、`scripts/run_eval.py`、`configs/eval.yaml`、`outputs/eval/`。
- **然后是论文主菜**：`graphs/prognostics.py`——把 physics-grounded RUL 做成图里的节点/工具（如 `derive_degradation_features` 读手册限值 → `estimate_rul`）。保持它是**图**：手册依据路径和物理路径同一 trace 里都可见、可分别测。
- **验收**：一条命令产出带版本号的评估报告，之后每次改动都能重跑对比；至少一组 before/after 表（hybrid vs 纯 dense、chunk 800 vs 1200）；一个失败样例的 trace 能指出**哪个节点**错了、检索到了什么。

## 贯穿始终的 4 个习惯（阶段 1 就开始，别断）

1. **LangSmith 追踪从第一个脚本常开**——不然你只能靠 print 猜。
2. **每个纯函数配一个单测**（fusion、budget、prompt 解析、chunking）：无 GPU、无网络、毫秒级。
3. **每阶段一次 commit + 一份评估报告**；用 `scripts/run_eval.py` 的输出来证明"这阶段真的变好了"。
4. **每个不可逆决定写一页 ADR**（`docs/decisions/`：embedding 选型、chunker 策略、2-step vs agentic）——这是论文附录的现成材料，当天写只要 10 分钟，事后补要 2 小时。

---

# Part 3｜注意事项

- **4096 的升级路径**：想开大就把服务端 `--max-model-len` 调大重启（KV cache 显存换上下文），应用侧只改 `configs/default.yaml` 一个值——这正是把它做成配置的意义。
- **别照抄网上教程的 import**：先看教程年份/版本（0.x 的 `langchain.retrievers`、`langchain_community.vectorstores` 那套已过时）；本项目的速查表见 Part 2 开头。
- **旧代码的剩余价值**（只作参考实现，不作基础）：`git show HEAD:src/rag/ingestion/chunk.py`（ATA 章节关键词表）、`.../common/text_clean.py`（水印/页眉正则——这是你们域内最值钱的小资产）、`.../retrieval/hybrid_retriever.py`（RRF 标准写法）。移植进新结构对应文件即可。
- **与论文的映射**：embedding 对比 = 消融实验；hybrid vs dense 表 = 检索章节；评估报告 = 实验章节；ADR = 附录；`outputs/figures/` 直接进论文。
- **与求职的映射**：面试最值得讲的三个点——① 为什么自定义 `BaseRetriever` 而不是用现成（因为 community sunset + 领域需求）；② LangGraph 里 state 为什么只存引用不存正文（4096/checkpoint 权衡）；③ 评估体系怎么设计的。
- **与全栈路线的衔接**：本路线跑通（阶段 6）后接回之前规划的全栈线——FastAPI 包住 LangGraph（`graph.astream` → SSE），前端消费流式输出；目录里的 `cli.py` 和 `graphs/` 那时直接复用，不用推倒重来。
- **交付说明**：本文档即交付物（目录设计 + 学习路线）。按你的要求我不动代码；想让我搭阶段 0 的机械性脚手架（.gitignore / .env.example / pyproject 构建配置 / 空骨架目录），或者哪一步卡住要答疑，随时说。
