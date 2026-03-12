# 从零搭建：RAG + 本地 Llama Python 项目

这个项目是一个**教学友好**的最小可用实现，目标是帮助你理解完整链路：

1. 构建并管理本地知识库
2. 查询知识库
3. 拼接 Prompt 并调用本地模型
4. 返回响应

---

## 1. 项目结构

```text
.
├── pyproject.toml
├── README.md
├── data/
├── src/rag_local_llama/
│   ├── cli.py               # 命令行入口
│   ├── embedding.py         # TF-IDF 向量化与相似度
│   ├── knowledge_base.py    # 知识库（SQLite + chunk）
│   ├── prompting.py         # Prompt 模板拼接
│   ├── llama_client.py      # 本地 Llama 调用
│   └── rag_engine.py        # RAG 主流程
└── tests/
```

---

## 2. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果你想跑测试：

```bash
pip install -e .[dev]
pytest -q
```

---

## 3. 准备知识库数据

把你的资料（`.txt/.md/.json`）放到任意目录，例如：

```bash
rag-cli add ./docs/公司制度.md --db data/knowledge.db --chunk-size 350 --overlap 60
```

### 参数解释（知识库构建）

- `chunk-size`：每个切片最大字符数。大一些保留更多上下文，小一些检索更精细。
- `overlap`：相邻切片重叠字符数，避免信息断裂。
- `db`：SQLite 文件路径，用于持久化知识片段。

---

## 4. 查询知识库

```bash
rag-cli search "报销流程是什么？" --db data/knowledge.db --top-k 3
```

### 参数解释（检索）

- `question`：查询问题。
- `top-k`：返回最相关的前 `k` 个片段。

---

## 5. RAG 问答（Prompt 拼接 + 模型调用）

```bash
rag-cli ask "报销要准备哪些材料？" \
  --db data/knowledge.db \
  --model /path/to/your-model.gguf \
  --top-k 3 \
  --temperature 0.3 \
  --top-p 0.95 \
  --max-tokens 512 \
  --repeat-penalty 1.1 \
  --show-prompt
```

### 5.1 Prompt 中各参数的含义（重点）

`src/rag_local_llama/prompting.py` 的 `PromptConfig`：

- `system_role`
  - 作用：模型的“最高级行为约束”。
  - 例子：你是企业知识库助手，必须基于资料回答。

- `task_instruction`
  - 作用：当前任务的执行要求。
  - 例子：先给结论，再解释依据；不知道就明确说明。

- `response_style`
  - 作用：控制输出风格与格式。
  - 例子：中文、分点、简洁。

- `max_context_chars`
  - 作用：限制注入上下文长度，防止 prompt 过长导致截断或成本增加。
  - 建议：和模型的 `n_ctx` 一起调整。

- `cite_sources`
  - 作用：是否要求模型输出参考来源，提升可追溯性。

### 5.2 生成参数的含义（`GenerationConfig`）

- `temperature`：随机性。越低越稳定，越高越发散。
- `top_p`：核采样范围。越小越保守。
- `max_tokens`：最多生成多少 token。
- `repeat_penalty`：重复惩罚，缓解模型“车轱辘话”。
- `stop`：停止词，命中后停止生成。

### 5.3 RAG 主流程

位于 `src/rag_local_llama/rag_engine.py`：

1. 调 `KnowledgeBase.query()` 检索 top-k 片段
2. 用 `build_prompt()` 组装完整提示词
3. 调 `LocalLlamaClient.generate()` 调本地 Llama
4. 返回响应 + 使用到的上下文

---

## 6. 你可以继续扩展的功能

- 支持 PDF/Word 导入（如 `pypdf`、`python-docx`）
- 接入向量数据库（FAISS / Chroma）
- 增加 rerank 模型，提高检索准确率
- 多轮对话记忆
- FastAPI Web 服务 + 前端页面

---

## 7. 学习建议

先用少量文档跑通，再逐步调以下参数观察效果变化：

1. `chunk-size` / `overlap`
2. `top-k`
3. `max_context_chars`
4. `temperature` / `top_p`

这样你会非常直观地理解 RAG 系统的质量与稳定性是如何被参数影响的。
