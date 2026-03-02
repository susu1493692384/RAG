# Milvus 部署说明

## 问题说明

当前遇到错误：
```
ModuleNotFoundError: No module named 'milvus_lite'
```

这是因为代码默认配置使用 Milvus Lite 本地文件模式，但安装的 pymilvus 可能不完整。

---

## 解决方案

### 方案 1：使用 Docker 运行 Milvus（推荐）

#### 1. 安装 Docker Desktop

下载地址：https://www.docker.com/products/docker-desktop/

#### 2. 启动 Milvus 容器

创建 `docker-compose.yml`：

```yaml
version: '3.5'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
    volumes:
      - etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio:/minio_data
    command: minio server /minio_data

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
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"
```

启动命令：
```powershell
docker-compose up -d
```

#### 3. 验证运行

访问：http://localhost:19530

应该看到 Milvus 运行正常。

---

### 方案 2：修改配置使用内存模式（开发测试）

如果只是想快速测试，可以修改配置使用内存：

```python
# enterprise_rag/config/milvus.py
uri: str = "milvus_demo.db"  # 内存模式
```

然后运行：
```bash
python -m enterprise_rag.main --mode build
```

---

### 方案 3：跳过向量存储，只测试其他模块

如果想先测试文档处理、Embedding 等功能：

创建 `test_simple.py`：

```python
from enterprise_rag.processors import DocumentLoader, TextSplitter
from enterprise_rag.embeddings import EmbeddingService

# 加载文档
loader = DocumentLoader("./data/documents")
docs = loader.load_directory()

# 分割文本
splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents([
    {'content': d.content, 'metadata': d.metadata}
    for d in docs
])

# 测试 Embedding
from enterprise_rag.config import EMBEDDING_CONFIGS
embedding_config = EMBEDDING_CONFIGS['m3e_base']
embedding_service = EmbeddingService(embedding_config)

# 向量化一个测试文档
vector = embedding_service.encode_documents("测试文本")[0]
print(f"向量维度: {len(vector)}")
print(f"前10个值: {vector[:10]}")
```

运行：
```bash
python test_simple.py
```

---

## 快速验证步骤

### 步骤 1：检查 Docker

```powershell
docker --version
docker ps
```

### 步骤 2：启动 Milvus

```powershell
# 创建配置文件
# （使用上面的 docker-compose.yml）

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 步骤 3：测试连接

```python
from pymilvus import MilvusClient

# 连接远程 Milvus
client = MilvusClient(uri="http://localhost:19530")

# 测试
print(f"Milvus 连接成功！")
```

### 步骤 4：构建知识库

```bash
python -m enterprise_rag.main --mode build
```

---

## 常见问题

### Q: Docker 启动失败？

A: 检查端口是否被占用
```powershell
netstat -ano | findstr :19530
```

### Q: pymilvus 连接超时？

A: 确保 Milvus 容器正在运行
```powershell
docker ps | findstr milvus
```

### Q: 构建失败提示连接错误？

A: 修改 Milvus 配置中的 URI，确保正确：
```python
# config/milvus.py
uri: str = "http://localhost:19530"  # 确保地址正确
```

---

## 推荐：使用 Docker Desktop

1. **下载安装** Docker Desktop
2. **启动容器**：`docker-compose up -d`
3. **验证运行**：访问 http://localhost:19530
4. **构建知识库**：`python -m enterprise_rag.main --mode build`
5. **启动服务**：`python -m enterprise_rag.main --mode api`

这是最稳定的方案！
