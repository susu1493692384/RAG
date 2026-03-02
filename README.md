# 企业 RAG 知识库系统

> 基于 Milvus + LangChain 的企业级知识库问答系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5%2B-orange)](https://milvus.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能特性

- **📚 多源数据支持** - 文本文档、PDF、Word、网页、数据库记录
- **🔍 混合检索** - 向量检索 + BM25 全文检索
- **🎯 重排序优化** - BGE-Reranker 二次排序，提高准确率
- **💬 多轮对话** - 支持对话式问答
- **🌐 RESTful API** - FastAPI 提供 HTTP 接口
- **🖥️ Web 界面** - Streamlit 交互式前端
- **🔧 模块化设计** - 易于扩展和定制

## 系统架构

```
数据源 → 文档处理 → 向量化 → Milvus 存储
                ↓
用户查询 → 检索器 → LLM → 回答
```

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
cd enterprise_rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入 API Key
# DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 构建知识库

```bash
# 准备文档
mkdir -p data/documents
# 将您的文档放入 data/documents 目录

# 构建向量库
python -m enterprise_rag.main --mode build
```

### 4. 启动服务

```bash
# 启动 API 服务
python -m enterprise_rag.main --mode api

# 或启动 Web 界面
python -m enterprise_rag.main --mode frontend
```

## 项目结构

```
enterprise_rag/
├── config/                 # 配置模块
│   ├── embeddings.py       # Embedding 配置
│   ├── llm.py            # LLM 配置
│   └── milvus.py         # Milvus 配置
│
├── processors/            # 数据处理模块
│   ├── document_loader.py # 文档加载
│   ├── ocr_processor.py  # OCR 处理
│   ├── web_scraper.py    # 网页爬取
│   └── text_splitter.py  # 文本分块
│
├── embeddings/           # 向量化模块
│   └── embedding_service.py
│
├── retriever/          # 检索器模块
│   ├── vector_retriever.py  # 向量检索
│   ├── hybrid_retriever.py  # 混合检索
│   └── reranker.py         # 重排序
│
├── chains/             # LangChain 链
│   ├── qa_chain.py    # 问答链
│   └── graph.py       # 流程图
│
├── api/               # API 服务
│   ├── main.py        # FastAPI 主应用
│   ├── routers/       # 路由
│   └── models/        # 数据模型
│
├── frontend/          # 前端
│   └── streamlit/
│       └── app.py    # Streamlit 应用
│
├── utils/            # 工具模块
│   ├── logger.py
│   └── helpers.py
│
└── tests/            # 测试
    ├── test_embeddings.py
    ├── test_retrieval.py
    └── test_chains.py
```

## 使用示例

### Python API

```python
from enterprise_rag import (
    EmbeddingConfig,
    LLMConfig,
    MilvusConfig,
    EmbeddingService,
    VectorRetriever,
    create_qa_chain,
)

# 配置
embedding_config = EmbeddingConfig(model_name="m3e_large")
llm_config = LLMConfig(provider="deepseek")
milvus_config = MilvusConfig(uri="./data/knowledge.db")

# 初始化
embedding_service = EmbeddingService(embedding_config)
retriever = VectorRetriever(milvus_config, embedding_service)

# 添加文档
retriever.add_documents([
    {"content": "文档内容", "metadata": {"source": "doc.txt"}}
])

# 创建问答链
qa_chain = create_qa_chain(retriever, llm_config)

# 查询
result = qa_chain.invoke("您的问题")
print(result.answer)
```

### HTTP API

```bash
# 查询
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "您的问题",
    "top_k": 5
  }'

# 对话
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "您的问题"
  }'
```

## 配置说明

### Embedding 模型

支持以下模型：

| 模型 | 维度 | 特点 |
|------|------|------|
| bge-m3 | 1024 | 多语言，高质量 |
| m3e-large | 1024 | 中文优化 |
| m3e-base | 768 | 中文轻量级 |

### LLM 提供商

支持以下 LLM：

- **DeepSeek**（推荐，性价比高）
- 通义千问
- 智谱 GLM
- OpenAI（需翻墙）

## 开发

### 运行测试

```bash
pytest tests/
```

### 代码风格

```bash
# 格式化代码
black enterprise_rag/

# 检查类型
mypy enterprise_rag/
```

## 性能优化

1. **使用批量处理** - 文档添加、向量化都支持批量操作
2. **启用索引** - HNSW 索引可大幅提升检索速度
3. **调整 chunk_size** - 根据文档类型选择合适的分块大小
4. **使用重排序** - 提高检索准确率

## 常见问题

**Q: 如何选择 Embedding 模型？**
A: 中文场景推荐 m3e-large 或 bge-m3，多语言场景选择 bge-m3。

**Q: 如何提高检索准确率？**
A: 1) 使用更好的 Embedding 模型 2) 启用重排序 3) 调整分块策略

**Q: 支持哪些文档格式？**
A: TXT, MD, PDF, DOCX, XLSX, PPTX，图片（通过 OCR）。

**Q: 可以离线使用吗？**
A: 可以，使用本地 Embedding 模型和本地 LLM（如 Llama）。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Milvus](https://github.com/milvus-io/milvus)
- [BGE Embedding](https://github.com/FlagOpen/FlagEmbedding)
