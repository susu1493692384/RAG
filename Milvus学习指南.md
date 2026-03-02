# Milvus 完整学习指南

> Milvus 是一个开源的向量数据库，专为 AI 应用设计，支持海量向量数据的存储、索引和检索。

---

## 目录

1. [Milvus 简介](#1-milvus-简介)
2. [核心概念](#2-核心概念)
3. [环境安装](#3-环境安装)
4. [快速入门](#4-快速入门)
5. [进阶功能](#5-进阶功能)
6. [生产部署](#6-生产部署)
7. [最佳实践](#7-最佳实践)
8. [常见问题](#8-常见问题)

---

## 1. Milvus 简介

### 1.1 什么是 Milvus？

Milvus 是一个开源的**向量数据库**，专门用于存储、索引和检索海量向量数据。它是构建 AI 应用的核心基础设施，广泛应用于：

- **语义搜索** - 理解查询意图，返回语义相关的结果
- **推荐系统** - 基于用户行为和内容的个性化推荐
- **RAG (检索增强生成)** - 为 LLM 提供外部知识库
- **图像/视频检索** - 以图搜图、相似内容推荐
- **异常检测** - 识别偏离正常模式的数据

### 1.2 为什么选择 Milvus？

| 特性 | 说明 |
|------|------|
| **高性能** | 支持十亿级向量毫秒级检索 |
| **可扩展** | 支持从笔记本到分布式集群的多种部署方式 |
| **易用性** | 提供 Python、Java、Go、C#、Node.js 等多种客户端 SDK |
| **开源免费** | Apache 2.0 许可证，完全开源 |
| **丰富索引** | 支持 FLAT、IVF、HNSW、DISKANN 等多种索引类型 |
| **云原生** | 支持 Kubernetes、Docker 容器化部署 |

### 1.3 Milvus 架构

```
┌─────────────────────────────────────────────────┐
│                  客户端层                      │
│  (Python / Java / Go / C# / Node.js / REST)   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│                  协议层                        │
│           (gRPC / RESTful API)                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│                  服务层                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Proxy    │  │ Query    │  │ Data     │   │
│  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│                  存储层                        │
│     (etcd / MinIO / S3 / Pulsar / RocksDB)     │
└─────────────────────────────────────────────────┘
```

---

## 2. 核心概念

### 2.1 向量 (Vector)

向量是神经网络模型（如 BERT、GPT、CLIP）的输出，用于表示数据的语义信息。

```python
# 示例：文本转换为向量
text = "人工智能改变了世界"
vector = embedding_model.encode(text)  # [0.12, -0.34, 0.56, ..., 0.78]
# 维度：768（取决于模型）
```

### 2.2 Collection（集合）

Collection 类似于 SQL 数据库中的**表**，用于存储：
- **向量字段** - 存储向量数据
- **标量字段** - 存储元数据（文本、数字等）
- **主键字段** - 唯一标识每条记录

### 2.3 Entity（实体）

Entity 是 Collection 中的一条记录，包含：
- 主键 (ID)
- 向量
- 元数据字段

```python
entity = {
    "id": 1,
    "vector": [0.1, 0.2, ..., 0.768],  # 768维向量
    "text": "这是一段文本",
    "category": "科技"
}
```

### 2.4 索引（Index）

索引用于加速向量检索：

| 索引类型 | 特点 | 适用场景 |
|---------|------|----------|
| **FLAT** | 精确搜索，100%召回率 | 数据量 < 100万 |
| **IVF_FLAT** | 平衡速度与精度 | 中等规模数据 |
| **IVF_SQ8** | 内存优化，精度略降 | 大规模数据 |
| **HNSW** | 极快速度，高精度 | 实时性要求高 |
| **DISKANN** | 磁盘索引，节省内存 | 超大规模数据 |

### 2.5 距离度量

衡量向量相似度的方法：

| 度量类型 | 公式 | 适用场景 |
|---------|------|----------|
| **L2 (欧氏距离)** | √Σ(aᵢ - bᵢ)² | 图像、语音特征 |
| **IP (内积)** | Σ(aᵢ × bᵢ) | 归一化向量 |
| **COSINE (余弦)** | cos(θ) = (A·B)/(\|A\|×\|B\|) | 文本语义 |

---

## 3. 环境安装

### 3.1 Python 环境要求

- Python 3.8 或更高版本
- pip 包管理器

### 3.2 安装 Milvus

#### 方式一：Milvus Lite（推荐入门学习）

```bash
pip install -U pymilvus
```

**特点：**
- ✅ 轻量级，无需单独服务
- ✅ 适合本地开发和测试
- ✅ 所有数据存储在单个 `.db` 文件中

#### 方式二：Docker 部署（适合生产环境）

```bash
# 下载 docker-compose.yml
wget https://github.com/milvus-io/milvus/releases/download/v2.5.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动 Milvus 服务
docker-compose up -d
```

#### 方式三：安装模型支持（用于生成向量）

```bash
pip install "pymilvus[model]"
```

---

## 4. 快速入门

### 4.1 连接 Milvus

```python
from pymilvus import MilvusClient

# Milvus Lite - 本地文件存储
client = MilvusClient("milvus_demo.db")

# 或连接远程 Milvus 服务器
# client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
```

### 4.2 创建 Collection

```python
# 删除已存在的 collection（可选）
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")

# 创建新 collection
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # 向量维度（根据 embedding 模型确定）
)

print("✅ Collection 创建成功！")
```

**默认配置说明：**
- 主键字段名：`id`（整数类型）
- 向量字段名：`vector`
- 距离度量：`COSINE`（余弦相似度）
- 不启用自动 ID

### 4.3 准备数据

#### 使用真实 Embedding 模型

```python
from pymilvus import model

# 下载并加载默认 embedding 模型
# 模型：paraphrase-albert-small-v2 (~50MB)
embedding_fn = model.DefaultEmbeddingFunction()

# 准备文本数据
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

# 生成向量
vectors = embedding_fn.encode_documents(docs)

print(f"向量维度: {embedding_fn.dim}")  # 768
print(f"向量数量: {len(vectors)}")      # 3

# 构造数据实体
data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

print(f"✅ 准备了 {len(data)} 条数据")
```

#### 使用随机向量（快速测试）

```python
import random

docs = ["文本1", "文本2", "文本3"]
vectors = [[random.uniform(-1, 1) for _ in range(768)] for _ in docs]

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "test"}
    for i in range(len(vectors))
]
```

### 4.4 插入数据

```python
res = client.insert(collection_name="demo_collection", data=data)

print(f"✅ 成功插入 {res['insert_count']} 条数据")
print(f"   ID 列表: {res['ids']}")
```

**输出示例：**
```
✅ 成功插入 3 条数据
   ID 列表: [0, 1, 2]
```

### 4.5 向量搜索

```python
# 将查询文本转换为向量
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

# 执行搜索
res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    limit=2,                      # 返回前 2 个最相似结果
    output_fields=["text", "subject"],  # 返回的字段
)

# 解析结果
for hit in res[0]:
    print(f"ID: {hit['id']}")
    print(f"距离: {hit['distance']:.4f}")
    print(f"文本: {hit['entity']['text']}")
    print(f"主题: {hit['entity']['subject']}")
    print("-" * 50)
```

**输出示例：**
```
ID: 2
距离: 0.5859
文本: Born in Maida Vale, London, Turing was raised in southern England.
主题: history
--------------------------------------------------
ID: 1
距离: 0.5118
文本: Alan Turing was first person to conduct substantial research in AI.
主题: history
--------------------------------------------------
```

### 4.6 带过滤条件的搜索

```python
# 先插入更多数据
new_docs = [
    "Machine learning has been used for drug design.",
    "Computational synthesis with AI algorithms predicts molecular properties.",
    "DDR1 is involved in cancers and fibrosis.",
]

new_vectors = embedding_fn.encode_documents(new_docs)
new_data = [
    {"id": 3 + i, "vector": new_vectors[i], "text": new_docs[i], "subject": "biology"}
    for i in range(len(new_vectors))
]

client.insert(collection_name="demo_collection", data=new_data)

# 带过滤条件的搜索
res = client.search(
    collection_name="demo_collection",
    data=embedding_fn.encode_queries(["tell me AI related information"]),
    filter="subject == 'biology'",  # 只搜索 subject='biology' 的记录
    limit=2,
    output_fields=["text", "subject"],
)

print("🔍 只检索 'biology' 主题的相关内容：")
for hit in res[0]:
    print(f"  {hit['entity']['text']}")
```

### 4.7 其他操作

#### 查询（Query）

```python
# 根据条件查询
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(f"查询到 {len(res)} 条 history 记录")
```

#### 根据ID获取

```python
# 根据主键获取实体
res = client.query(
    collection_name="demo_collection",
    ids=[0, 2],
    output_fields=["text", "subject"],
)
```

#### 删除数据

```python
# 根据主键删除
res = client.delete(collection_name="demo_collection", ids=[0, 2])
print(f"删除了 ID: {res}")

# 根据条件删除
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'biology'"
)
print(f"删除了 biology 主题的所有记录")
```

#### 删除 Collection

```python
client.drop_collection(collection_name="demo_collection")
print("✅ Collection 已删除")
```

---

## 5. 进阶功能

### 5.1 自定义 Schema

对于生产环境，建议明确定义 Schema：

```python
from pymilvus import MilvusClient, DataType

schema = MilvusClient.create_schema()

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=255)

# 创建 Collection
client.create_collection(
    collection_name="custom_collection",
    schema=schema,
    metric_type="IP",  # 使用内积作为距离度量
)
```

### 5.2 创建索引

```python
# 创建 HNSW 索引（高速度、高精度）
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,        # 图的连接数
        "efConstruction": 256,  # 构建时的搜索范围
    }
)

client.create_index(
    collection_name="demo_collection",
    index_params=index_params,
)
```

### 5.3 向量搜索参数优化

```python
search_params = {
    "metric_type": "COSINE",
    "params": {
        "ef": 64,  # HNSW 搜索参数，越大越精确但越慢
    }
}

res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    search_params=search_params,
    limit=10,
)
```

### 5.4 动态字段（Dynamic Field）

允许插入未在 Schema 中定义的字段：

```python
# 启用动态字段
client.create_collection(
    collection_name="dynamic_collection",
    dimension=768,
    enable_dynamic_field=True,
)

# 插入包含任意字段的数据
data = [{
    "vector": [0.1, 0.2, ...],
    "text": "示例文本",
    "custom_field": "自定义值",  # 未在 Schema 中定义
    "timestamp": 1234567890,
}]

client.insert("dynamic_collection", data=data)
```

---

## 6. 生产部署

### 6.1 Docker Standalone 部署

```yaml
# docker-compose.yml
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    image: milvusdb/milvus:v2.5.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - "etcd"
      - "minio"

volumes:
  etcd:
  minio:
  milvus:
```

启动命令：
```bash
docker-compose up -d
```

### 6.2 连接远程 Milvus

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"  # 默认用户名和密码
)
```

### 6.3 Kubernetes 部署（Milvus Operator）

```bash
# 安装 Milvus Operator
kubectl apply -f https://github.com/milvus-io/milvus-operator/releases/download/v0.9.0/milvus-operator.yaml

# 部署 Milvus 集群
kubectl apply -f milvus_cluster.yaml
```

---

## 7. 最佳实践

### 7.1 数据导入

```python
# 批量插入（推荐）
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    client.insert(collection_name="my_collection", data=batch)
```

### 7.2 索引选择策略

| 数据规模 | 推荐索引 | 内存占用 | 查询速度 | 精度 |
|---------|---------|---------|---------|------|
| < 100万 | FLAT | 高 | 慢 | 100% |
| 100万-1000万 | IVF_FLAT | 高 | 中 | 高 |
| 1000万-1亿 | IVF_SQ8 | 低 | 中 | 中 |
| > 1亿 | HNSW / DISKANN | 中/低 | 快 | 高 |

### 7.3 性能优化

```python
# 1. 使用批量搜索而非单次搜索
query_vectors = embedding_fn.encode_queries(["查询1", "查询2", "查询3"])
res = client.search(collection_name="my_collection", data=query_vectors, limit=10)

# 2. 限制返回字段
res = client.search(
    collection_name="my_collection",
    data=query_vectors,
    output_fields=["id", "title"],  # 只返回需要的字段
    limit=10,
)

# 3. 使用 nprobe 参数调整搜索范围
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16},  # IVF 索引：搜索的聚类数
}
```

### 7.4 数据持久化

```python
# Milvus Lite 数据持久化在文件中
client = MilvusClient("my_data.db")

# 重启程序后重新连接即可恢复数据
client = MilvusClient("my_data.db")  # 数据仍在
```

---

## 8. 常见问题

### Q1: 如何选择向量维度？

**答：** 取决于使用的 embedding 模型：
- `all-MiniLM-L6-v2`: 384 维
- `paraphrase-albert-small-v2`: 768 维
- `text-embedding-ada-002` (OpenAI): 1536 维
- `bert-base-uncased`: 768 维

维度越高，通常语义表示越丰富，但存储和计算成本也越高。

### Q2: 搜索返回的距离值含义？

**答：**
- **COSINE**: 范围 [-1, 1]，越接近 1 越相似
- **L2**: 范围 [0, ∞)，越接近 0 越相似
- **IP**: 范围 (-∞, ∞)，越接近 1 越相似（归一化向量）

### Q3: 如何处理中文文本？

**答：** 使用支持中文的 embedding 模型：
```python
from pymilvus import model

# 使用中文模型
embedding_fn = model.DefaultEmbeddingFunction()
# 或指定其他模型，如 m3e-base、text2vec-base-chinese 等
```

### Q4: Milvus Lite 适合生产环境吗？

**答：** 不推荐。Milvus Lite 仅适合：
- 本地开发测试
- 小规模原型项目（< 10万向量）
- 学习和教学

生产环境建议使用 Docker 或 Kubernetes 部署完整版 Milvus。

### Q5: 如何备份 Milvus Lite 数据？

**答：** 直接复制 `.db` 文件即可：
```bash
cp milvus_demo.db backup/milvus_demo_$(date +%Y%m%d).db
```

### Q6: 如何更新已有向量？

**答：** Milvus 暂不支持直接更新向量，需要：
1. 删除旧记录
2. 插入新记录

```python
# 删除
client.delete(collection_name="my_collection", ids=[1, 2, 3])

# 插入新数据
new_data = [
    {"id": 1, "vector": new_vector1, ...},
    {"id": 2, "vector": new_vector2, ...},
]
client.insert(collection_name="my_collection", data=new_data)
```

---

## 学习路线建议

### 第一阶段：基础掌握（1-2天）
1. ✅ 安装 Milvus Lite 和 pymilvus
2. ✅ 完成快速入门示例
3. ✅ 理解 Collection、Entity、Vector 等核心概念
4. ✅ 练习插入、搜索、查询、删除操作

### 第二阶段：进阶实践（3-5天）
1. ✅ 尝试不同的 embedding 模型
2. ✅ 学习自定义 Schema 和索引
3. ✅ 实现带过滤条件的复杂搜索
4. ✅ 构建一个简单的语义搜索应用

### 第三阶段：生产部署（1周）
1. ✅ 使用 Docker 部署完整 Milvus
2. ✅ 学习性能优化和参数调优
3. ✅ 实现批量导入和大规模数据处理
4. ✅ 集成到实际项目中（如 RAG 系统）

---

## 参考资料

- [Milvus 官方文档](https://milvus.io/docs)
- [PyMilvus API 参考](https://milvus.io/api-reference/pymilvus/v2.5.x/About.md)
- [Milvus GitHub](https://github.com/milvus-io/milvus)
- [向量数据库最佳实践](https://zilliz.com/learn)

---

**最后更新时间：** 2026年2月
**Milvus 版本：** v2.5.x
