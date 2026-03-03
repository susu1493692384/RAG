# 知识库管理系统实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 为企业 RAG 系统添加多知识库管理功能，允许用户在前端界面动态创建、切换、删除知识库，并向指定知识库上传文档。

**架构:** 每个知识库使用独立的 Milvus Collection 进行数据隔离。后端提供 REST API 进行 CRUD 操作，前端 Streamlit 页面提供管理界面。所有知识库共享全局检索配置（embedding 模型、chunk_size 等）。

**技术栈:**
- FastAPI - REST API
- Streamlit - Web 前端
- PyMilvus - 向量数据库
- Pytest - 测试框架

---

## 前置准备

### Task 0: 验证环境和依赖

**文件:**
- Check: `requirements.txt`

**Step 1: 确认在正确的 worktree 中**

```bash
cd /e/SOFE/RAG/.worktrees/knowledge-base-management
pwd
```

Expected: `/e/SOFE/RAG/.worktrees/knowledge-base-management`

**Step 2: 检查当前分支**

```bash
git branch --show-current
```

Expected: `feature/knowledge-base-management`

**Step 3: 验证 Python 环境**

```bash
python --version
python -m pytest --version
```

Expected: Python 3.x 和 pytest 9.0.2

---

## 阶段一：数据模型和元数据存储

### Task 1: 创建知识库元数据模型

**文件:**
- Create: `enterprise_rag/services/__init__.py`
- Create: `enterprise_rag/services/models.py`
- Test: `enterprise_rag/tests/test_kb_models.py`

**Step 1: 创建 services 包**

```bash
mkdir -p enterprise_rag/services
touch enterprise_rag/services/__init__.py
```

**Step 2: 创建测试文件**

```python
# enterprise_rag/tests/test_kb_models.py
"""测试知识库数据模型"""
import pytest
from datetime import datetime
from enterprise_rag.services.models import KnowledgeBase, KnowledgeBaseMetadata

def test_create_knowledge_base():
    """测试创建知识库对象"""
    kb = KnowledgeBase(
        id="kb_001",
        name="产品文档",
        description="公司产品相关文档"
    )
    assert kb.id == "kb_001"
    assert kb.name == "产品文档"
    assert kb.collection_name == "kb_001"
    assert kb.doc_count == 0
    assert isinstance(kb.created_at, datetime)

def test_knowledge_base_to_dict():
    """测试知识库转换为字典"""
    kb = KnowledgeBase(
        id="kb_001",
        name="产品文档",
        description="公司产品相关文档"
    )
    data = kb.to_dict()
    assert data["id"] == "kb_001"
    assert data["name"] == "产品文档"
    assert "created_at" in data

def test_create_metadata_storage():
    """测试元数据存储对象"""
    metadata = KnowledgeBaseMetadata()
    assert len(metadata.list_all()) == 0

def test_metadata_add_and_get():
    """测试添加和获取知识库"""
    metadata = KnowledgeBaseMetadata()
    kb = KnowledgeBase(
        id="kb_001",
        name="产品文档",
        description="公司产品相关文档"
    )
    metadata.add(kb)

    retrieved = metadata.get("kb_001")
    assert retrieved is not None
    assert retrieved.id == "kb_001"
    assert retrieved.name == "产品文档"

def test_metadata_list_all():
    """测试列出所有知识库"""
    metadata = KnowledgeBaseMetadata()
    kb1 = KnowledgeBase(id="kb_001", name="KB1")
    kb2 = KnowledgeBase(id="kb_002", name="KB2")

    metadata.add(kb1)
    metadata.add(kb2)

    all_kbs = metadata.list_all()
    assert len(all_kbs) == 2
    assert any(kb.id == "kb_001" for kb in all_kbs)
    assert any(kb.id == "kb_002" for kb in all_kbs)

def test_metadata_delete():
    """测试删除知识库"""
    metadata = KnowledgeBaseMetadata()
    kb = KnowledgeBase(id="kb_001", name="KB1")
    metadata.add(kb)

    metadata.delete("kb_001")
    assert metadata.get("kb_001") is None
    assert len(metadata.list_all()) == 0

def test_metadata_exists():
    """测试知识库是否存在"""
    metadata = KnowledgeBaseMetadata()
    assert not metadata.exists("kb_001")

    kb = KnowledgeBase(id="kb_001", name="KB1")
    metadata.add(kb)
    assert metadata.exists("kb_001")
```

**Step 3: 运行测试确认失败**

```bash
cd /e/SOFE/RAG/.worktrees/knowledge-base-management
python -m pytest enterprise_rag/tests/test_kb_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'enterprise_rag.services'`

**Step 4: 实现数据模型**

```python
# enterprise_rag/services/models.py
"""知识库数据模型"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class KnowledgeBase:
    """知识库模型"""
    id: str
    name: str
    description: str = ""
    collection_name: str = ""
    doc_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """初始化后处理"""
        if not self.collection_name:
            self.collection_name = f"kb_{self.id}"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "collection_name": self.collection_name,
            "doc_count": self.doc_count,
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeBase':
        """从字典创建"""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            collection_name=data.get("collection_name", f"kb_{data['id']}"),
            doc_count=data.get("doc_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )


class KnowledgeBaseMetadata:
    """知识库元数据存储"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化元数据存储

        Args:
            storage_path: JSON 存储文件路径
        """
        if storage_path is None:
            # 默认存储在项目 data 目录下
            self.storage_path = Path(__file__).parent.parent.parent / "data" / "knowledge_bases.json"
        else:
            self.storage_path = Path(storage_path)

        # 确保目录存在
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载现有数据
        self._data = self._load()

    def _load(self) -> dict:
        """从文件加载数据"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("knowledge_bases", [])
        return {}

    def _save(self):
        """保存数据到文件"""
        data = {
            "knowledge_bases": [
                kb.to_dict() for kb in self._data.values()
            ]
        }
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, kb: KnowledgeBase):
        """添加知识库"""
        self._data[kb.id] = kb
        self._save()

    def get(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self._data.get(kb_id)

    def list_all(self) -> List[KnowledgeBase]:
        """列出所有知识库"""
        return list(self._data.values())

    def delete(self, kb_id: str):
        """删除知识库"""
        if kb_id in self._data:
            del self._data[kb_id]
            self._save()

    def exists(self, kb_id: str) -> bool:
        """检查知识库是否存在"""
        return kb_id in self._data

    def update_doc_count(self, kb_id: str, count: int):
        """更新文档数量"""
        if kb_id in self._data:
            self._data[kb_id].doc_count = count
            self._save()
```

**Step 5: 运行测试确认通过**

```bash
python -m pytest enterprise_rag/tests/test_kb_models.py -v
```

Expected: `10 passed`

**Step 6: 提交**

```bash
git add enterprise_rag/services/ enterprise_rag/tests/test_kb_models.py
git commit -m "feat: add knowledge base data models and metadata storage

- Add KnowledgeBase dataclass with validation
- Add KnowledgeBaseMetadata for JSON persistence
- Add comprehensive tests for data models
```

---

### Task 2: 添加唯一 ID 生成器

**文件:**
- Modify: `enterprise_rag/services/models.py`
- Test: `enterprise_rag/tests/test_kb_models.py`

**Step 1: 添加测试**

```python
# 在 enterprise_rag/tests/test_kb_models.py 末尾添加

def test_generate_kb_id():
    """测试生成知识库 ID"""
    from enterprise_rag.services.models import generate_kb_id

    id1 = generate_kb_id()
    id2 = generate_kb_id()

    assert id1.startswith("kb_")
    assert id2.startswith("kb_")
    assert id1 != id2  # 应该是唯一的

def test_generate_kb_id_with_prefix():
    """测试带前缀的 ID 生成"""
    from enterprise_rag.services.models import generate_kb_id

    # 添加几个知识库后
    metadata = KnowledgeBaseMetadata()
    metadata.add(KnowledgeBase(id="kb_001", name="KB1"))

    new_id = generate_kb_id(metadata)
    assert new_id != "kb_001"  # 不应该重复
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest enterprise_rag/tests/test_kb_models.py::test_generate_kb_id -v
```

Expected: `ImportError: cannot import name 'generate_kb_id'`

**Step 3: 实现 ID 生成器**

```python
# 在 enterprise_rag/services/models.py 中添加

import uuid

def generate_kb_id(metadata: Optional[KnowledgeBaseMetadata] = None) -> str:
    """
    生成唯一的知识库 ID

    Args:
        metadata: 元数据存储对象，用于检查唯一性

    Returns:
        知识库 ID (格式: kb_<uuid>)
    """
    # 生成候选 ID
    if metadata:
        # 如果提供了 metadata，生成唯一 ID
        existing_ids = set(kb.id for kb in metadata.list_all())
        while True:
            kb_id = f"kb_{uuid.uuid4().hex[:8]}"
            if kb_id not in existing_ids:
                return kb_id
    else:
        # 简单生成
        return f"kb_{uuid.uuid4().hex[:8]}"
```

**Step 4: 运行测试确认通过**

```bash
python -m pytest enterprise_rag/tests/test_kb_models.py::test_generate_kb_id -v
```

Expected: `2 passed`

**Step 5: 提交**

```bash
git add enterprise_rag/services/models.py enterprise_rag/tests/test_kb_models.py
git commit -m "feat: add unique ID generator for knowledge bases"
```

---

## 阶段二：知识库核心服务

### Task 3: 创建知识库服务基础

**文件:**
- Create: `enterprise_rag/services/knowledge_base_service.py`
- Test: `enterprise_rag/tests/test_kb_service.py`

**Step 1: 创建测试文件**

```python
# enterprise_rag/tests/test_kb_service.py
"""测试知识库服务"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from enterprise_rag.services import KnowledgeBaseService
from enterprise_rag.services.models import KnowledgeBase


@pytest.fixture
def mock_embedding_service():
    """模拟 embedding 服务"""
    mock = Mock()
    mock.get_dimension.return_value = 1024
    return mock


@pytest.fixture
def mock_milvus_client():
    """模拟 Milvus 客户端"""
    with patch('enterprise_rag.services.knowledge_base_service.MilvusClient') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


def test_create_service(mock_embedding_service):
    """测试创建服务"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)
    assert service is not None
    assert service.embedding_service == mock_embedding_service


def test_create_knowledge_base(mock_embedding_service, mock_milvus_client):
    """测试创建知识库"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    kb = service.create_knowledge_base(
        name="产品文档",
        description="公司产品相关文档"
    )

    assert kb is not None
    assert kb.name == "产品文档"
    assert kb.description == "公司产品相关文档"
    assert kb.collection_name.startswith("kb_")

    # 验证 Milvus collection 被创建
    mock_milvus_client.create_collection.assert_called_once()


def test_create_knowledge_base_duplicate_name(mock_embedding_service):
    """测试创建重名知识库"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    # 创建第一个
    service.create_knowledge_base(name="产品文档")

    # 尝试创建同名
    with pytest.raises(ValueError, match="已存在"):
        service.create_knowledge_base(name="产品文档")


def test_list_knowledge_bases(mock_embedding_service):
    """测试列出知识库"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    # 创建几个知识库
    service.create_knowledge_base(name="KB1")
    service.create_knowledge_base(name="KB2")

    kbs = service.list_knowledge_bases()
    assert len(kbs) == 2
    assert any(kb.name == "KB1" for kb in kbs)
    assert any(kb.name == "KB2" for kb in kbs)


def test_get_knowledge_base(mock_embedding_service):
    """测试获取知识库"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    created = service.create_knowledge_base(name="测试KB")
    retrieved = service.get_knowledge_base(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "测试KB"


def test_delete_knowledge_base(mock_embedding_service, mock_milvus_client):
    """测试删除知识库"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    kb = service.create_knowledge_base(name="待删除KB")
    assert service.get_knowledge_base(kb.id) is not None

    service.delete_knowledge_base(kb.id)

    # 验证已删除
    assert service.get_knowledge_base(kb.id) is None

    # 验证 Milvus collection 被删除
    mock_milvus_client.drop_collection.assert_called_once()


def test_knowledge_base_exists(mock_embedding_service):
    """测试知识库是否存在"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    assert not service.knowledge_base_exists("kb_001")

    kb = service.create_knowledge_base(name="测试KB")
    assert service.knowledge_base_exists(kb.id)
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest enterprise_rag/tests/test_kb_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'enterprise_rag.services.knowledge_base_service'`

**Step 3: 实现服务**

```python
# enterprise_rag/services/knowledge_base_service.py
"""知识库服务"""
from typing import List, Optional
from pathlib import Path

from .models import KnowledgeBase, KnowledgeBaseMetadata, generate_kb_id


class KnowledgeBaseService:
    """知识库管理服务"""

    def __init__(self, embedding_service, milvus_config=None):
        """
        初始化知识库服务

        Args:
            embedding_service: Embedding 服务实例
            milvus_config: Milvus 配置
        """
        self.embedding_service = embedding_service
        self.metadata = KnowledgeBaseMetadata()
        self.milvus_config = milvus_config

    def create_knowledge_base(
        self,
        name: str,
        description: str = ""
    ) -> KnowledgeBase:
        """
        创建知识库

        Args:
            name: 知识库名称
            description: 知识库描述

        Returns:
            创建的知识库对象

        Raises:
            ValueError: 知识库名称已存在
        """
        # 检查名称是否已存在
        for kb in self.metadata.list_all():
            if kb.name == name:
                raise ValueError(f"知识库名称 '{name}' 已存在")

        # 生成唯一 ID
        kb_id = generate_kb_id(self.metadata)

        # 创建知识库对象
        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description
        )

        # 创建 Milvus Collection
        self._create_milvus_collection(kb.collection_name)

        # 保存元数据
        self.metadata.add(kb)

        return kb

    def _create_milvus_collection(self, collection_name: str):
        """创建 Milvus Collection"""
        from pymilvus import MilvusClient

        # 获取向量维度
        dimension = self.embedding_service.get_dimension()

        # 使用配置中的 URI 或默认值
        uri = self.milvus_config.uri if self.milvus_config else "./data/milvus_test.db"

        client = MilvusClient(uri=uri)

        # 创建 collection
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension
        )

    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        """列出所有知识库"""
        return self.metadata.list_all()

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self.metadata.get(kb_id)

    def delete_knowledge_base(self, kb_id: str):
        """
        删除知识库

        Args:
            kb_id: 知识库 ID
        """
        kb = self.metadata.get(kb_id)
        if not kb:
            raise ValueError(f"知识库 {kb_id} 不存在")

        # 删除 Milvus Collection
        self._drop_milvus_collection(kb.collection_name)

        # 删除元数据
        self.metadata.delete(kb_id)

    def _drop_milvus_collection(self, collection_name: str):
        """删除 Milvus Collection"""
        from pymilvus import MilvusClient

        uri = self.milvus_config.uri if self.milvus_config else "./data/milvus_test.db"
        client = MilvusClient(uri=uri)

        # 检查 collection 是否存在
        if client.has_collection(collection_name=collection_name):
            client.drop_collection(collection_name=collection_name)

    def knowledge_base_exists(self, kb_id: str) -> bool:
        """检查知识库是否存在"""
        return self.metadata.exists(kb_id)
```

**Step 4: 更新 services/__init__.py**

```python
# enterprise_rag/services/__init__.py
"""服务模块"""
from .knowledge_base_service import KnowledgeBaseService

__all__ = ['KnowledgeBaseService']
```

**Step 5: 运行测试确认通过**

```bash
python -m pytest enterprise_rag/tests/test_kb_service.py -v
```

Expected: `8 passed`

**Step 6: 提交**

```bash
git add enterprise_rag/services/ enterprise_rag/tests/test_kb_service.py
git commit -m "feat: add knowledge base service with CRUD operations

- Add KnowledgeBaseService for managing knowledge bases
- Support create, list, get, delete operations
- Auto-create Milvus collections for each knowledge base
- Add comprehensive tests
```

---

## 阶段三：API 路由

### Task 4: 创建 API 请求/响应模型

**文件:**
- Create: `enterprise_rag/api/models/knowledge_base.py`
- Test: `enterprise_rag/tests/test_api_kb_models.py`

**Step 1: 创建测试**

```python
# enterprise_rag/tests/test_api_kb_models.py
"""测试知识库 API 模型"""
import pytest
from datetime import datetime
from enterprise_rag.api.models.knowledge_base import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseResponse,
    UploadDocumentsRequest
)


def test_create_request_valid():
    """测试创建请求验证"""
    request = CreateKnowledgeBaseRequest(
        name="产品文档",
        description="公司产品相关文档"
    )
    assert request.name == "产品文档"
    assert request.description == "公司产品相关文档"


def test_create_request_name_too_short():
    """测试名称过短"""
    with pytest.raises(ValueError):
        CreateKnowledgeBaseRequest(name="")


def test_kb_response_from_model():
    """测试从数据模型创建响应"""
    from enterprise_rag.services.models import KnowledgeBase

    kb_model = KnowledgeBase(
        id="kb_001",
        name="产品文档",
        description="测试"
    )

    response = KnowledgeBaseResponse.from_model(kb_model)

    assert response.id == "kb_001"
    assert response.name == "产品文档"
    assert response.description == "测试"
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest enterprise_rag/tests/test_api_kb_models.py -v
```

Expected: `ModuleNotFoundError`

**Step 3: 实现 API 模型**

```python
# enterprise_rag/api/models/knowledge_base.py
"""知识库 API 模型"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求"""
    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: str = Field(default="", max_length=500, description="知识库描述")

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('名称不能为空')
        return v.strip()


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    id: str
    name: str
    description: str
    collection_name: str
    doc_count: int
    created_at: str

    @classmethod
    def from_model(cls, kb) -> 'KnowledgeBaseResponse':
        """从数据模型创建响应"""
        return cls(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            collection_name=kb.collection_name,
            doc_count=kb.doc_count,
            created_at=kb.created_at.isoformat()
        )


class UploadDocumentsRequest(BaseModel):
    """上传文档请求（用于表单数据）"""
    pass  # 实际使用 multipart/form-data
```

**Step 4: 更新 api/models/__init__.py**

```python
# enterprise_rag/api/models/__init__.py
"""API 模型"""
from .schemas import QueryRequest, QueryResponse, SourceInfo
from .knowledge_base import (
    CreateKnowledgeBaseRequest,
    KnowledgeBaseResponse
)

__all__ = [
    'QueryRequest',
    'QueryResponse',
    'SourceInfo',
    'CreateKnowledgeBaseRequest',
    'KnowledgeBaseResponse',
]
```

**Step 5: 运行测试确认通过**

```bash
python -m pytest enterprise_rag/tests/test_api_kb_models.py -v
```

Expected: `3 passed`

**Step 6: 提交**

```bash
git add enterprise_rag/api/models/ enterprise_rag/tests/test_api_kb_models.py
git commit -m "feat: add API request/response models for knowledge base"
```

---

### Task 5: 创建知识库 API 路由

**文件:**
- Create: `enterprise_rag/api/routers/knowledge_base.py`
- Test: `enterprise_rag/tests/test_api_kb_routes.py`
- Modify: `enterprise_rag/api/main.py`

**Step 1: 创建路由测试**

```python
# enterprise_rag/tests/test_api_kb_routes.py
"""测试知识库 API 路由"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from enterprise_rag.api.main import app


@pytest.fixture
def client():
    """测试客户端"""
    return TestClient(app)


@pytest.fixture
def mock_kb_service():
    """模拟知识库服务"""
    with patch('enterprise_rag.api.routers.knowledge_base.get_kb_service') as mock:
        service = Mock()
        mock.return_value = service
        yield service


def test_create_knowledge_base(client, mock_kb_service):
    """测试创建知识库 API"""
    from enterprise_rag.services.models import KnowledgeBase

    # 模拟返回
    mock_kb = KnowledgeBase(id="kb_001", name="产品文档", description="测试")
    mock_kb_service.create_knowledge_base.return_value = mock_kb

    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "产品文档", "description": "测试"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "kb_001"
    assert data["name"] == "产品文档"
    mock_kb_service.create_knowledge_base.assert_called_once()


def test_list_knowledge_bases(client, mock_kb_service):
    """测试列出知识库 API"""
    from enterprise_rag.services.models import KnowledgeBase

    mock_kb_service.list_knowledge_bases.return_value = [
        KnowledgeBase(id="kb_001", name="KB1"),
        KnowledgeBase(id="kb_002", name="KB2")
    ]

    response = client.get("/api/v1/knowledge-bases")

    assert response.status_code == 200
    data = response.json()
    assert len(data["knowledge_bases"]) == 2


def test_get_knowledge_base(client, mock_kb_service):
    """测试获取知识库 API"""
    from enterprise_rag.services.models import KnowledgeBase

    mock_kb = KnowledgeBase(id="kb_001", name="KB1")
    mock_kb_service.get_knowledge_base.return_value = mock_kb

    response = client.get("/api/v1/knowledge-bases/kb_001")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "kb_001"


def test_get_knowledge_base_not_found(client, mock_kb_service):
    """测试获取不存在的知识库"""
    mock_kb_service.get_knowledge_base.return_value = None

    response = client.get("/api/v1/knowledge-bases/kb_999")

    assert response.status_code == 404


def test_delete_knowledge_base(client, mock_kb_service):
    """测试删除知识库 API"""
    response = client.delete("/api/v1/knowledge-bases/kb_001")

    assert response.status_code == 200
    mock_kb_service.delete_knowledge_base.assert_called_once_with("kb_001")
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest enterprise_rag/tests/test_api_kb_routes.py -v
```

Expected: `ModuleNotFoundError`

**Step 3: 实现路由**

```python
# enterprise_rag/api/routers/knowledge_base.py
"""知识库管理路由"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from ..models import CreateKnowledgeBaseRequest, KnowledgeBaseResponse
from ...services import KnowledgeBaseService


# 全局服务实例（实际应使用依赖注入）
_kb_service: List[KnowledgeBaseService] = []


kb_router = APIRouter(
    prefix="/api/v1/knowledge-bases",
    tags=["知识库管理"],
)


def get_kb_service() -> KnowledgeBaseService:
    """获取知识库服务实例"""
    if not _kb_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="知识库服务未初始化"
        )
    return _kb_service[0]


@kb_router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    创建知识库

    Args:
        request: 创建请求
        service: 知识库服务

    Returns:
        创建的知识库
    """
    try:
        kb = service.create_knowledge_base(
            name=request.name,
            description=request.description
        )
        return KnowledgeBaseResponse.from_model(kb)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@kb_router.get("", response_model=dict)
async def list_knowledge_bases(
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    列出所有知识库

    Returns:
        知识库列表
    """
    kbs = service.list_knowledge_bases()
    return {
        "knowledge_bases": [
            KnowledgeBaseResponse.from_model(kb)
            for kb in kbs
        ]
    }


@kb_router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    获取知识库详情

    Args:
        kb_id: 知识库 ID
        service: 知识库服务

    Returns:
        知识库详情
    """
    kb = service.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库 {kb_id} 不存在"
        )
    return KnowledgeBaseResponse.from_model(kb)


@kb_router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    删除知识库

    Args:
        kb_id: 知识库 ID
        service: 知识库服务

    Returns:
        删除确认
    """
    try:
        service.delete_knowledge_base(kb_id)
        return {"message": f"知识库 {kb_id} 已删除"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


def initialize_kb_service(service: KnowledgeBaseService):
    """初始化知识库服务"""
    global _kb_service
    _kb_service.clear()
    _kb_service.append(service)
```

**Step 4: 注册路由**

```python
# 修改 enterprise_rag/api/main.py

# 在导入部分添加
from .routers import knowledge_base

# 在 initialize_service 函数中添加
def initialize_service(
    retriever,
    llm_config,
    use_reranker=True,
    kb_service=None,  # 新增参数
):
    # ... 现有代码 ...

    # 在文件末尾添加
    if kb_service:
        from .routers.knowledge_base import initialize_kb_service
        initialize_kb_service(kb_service)

# 在 run_server 函数中注册路由
def run_server(host: str = "0.0.0.0", port: int = 8000):
    app.include_router(qa_router)
    app.include_router(knowledge_base.kb_router)  # 新增

    # ... 现有代码 ...
```

**Step 5: 运行测试确认通过**

```bash
python -m pytest enterprise_rag/tests/test_api_kb_routes.py -v
```

Expected: `5 passed`

**Step 6: 提交**

```bash
git add enterprise_rag/api/routers/ enterprise_rag/api/main.py enterprise_rag/tests/test_api_kb_routes.py
git commit -m "feat: add knowledge base API routes

- Add POST /api/v1/knowledge-bases - create knowledge base
- Add GET /api/v1/knowledge-bases - list all knowledge bases
- Add GET /api/v1/knowledge-bases/{id} - get knowledge base details
- Add DELETE /api/v1/knowledge-bases/{id} - delete knowledge base
- Register routes in main API app
```

---

## 阶段四：前端管理页面

### Task 6: 创建知识库管理页面

**文件:**
- Create: `enterprise_rag/frontend/streamlit/pages/1_Knowledge_Management.py`

**Step 1: 创建页面**

```python
# enterprise_rag/frontend/streamlit/pages/1_Knowledge_Management.py
"""知识库管理页面"""
import streamlit as st
import requests
from typing import List, Dict


# 配置页面
st.set_page_config(
    page_title="知识库管理",
    page_icon="📚",
    layout="wide",
)

# API 配置
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")


def init_session_state():
    """初始化会话状态"""
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"


def list_knowledge_bases() -> List[Dict]:
    """获取知识库列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/knowledge-bases")
        if response.status_code == 200:
            data = response.json()
            return data.get("knowledge_bases", [])
    except Exception as e:
        st.error(f"获取知识库列表失败: {e}")
    return []


def create_knowledge_base(name: str, description: str) -> bool:
    """创建知识库"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/knowledge-bases",
            json={"name": name, "description": description}
        )
        if response.status_code == 200:
            st.success("知识库创建成功！")
            return True
        else:
            error = response.json().get("detail", "未知错误")
            st.error(f"创建失败: {error}")
    except Exception as e:
        st.error(f"创建失败: {e}")
    return False


def delete_knowledge_base(kb_id: str) -> bool:
    """删除知识库"""
    try:
        response = requests.delete(f"{API_BASE_URL}/api/v1/knowledge-bases/{kb_id}")
        if response.status_code == 200:
            st.success("知识库已删除")
            return True
        else:
            st.error("删除失败")
    except Exception as e:
        st.error(f"删除失败: {e}")
    return False


def show_create_dialog():
    """显示创建对话框"""
    with st.expander("➕ 创建新知识库", expanded=False):
        name = st.text_input("知识库名称", key="new_kb_name")
        description = st.text_area("描述", key="new_kb_desc")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("创建", type="primary"):
                if name and create_knowledge_base(name, description):
                    st.rerun()
        with col2:
            if st.button("取消"):
                st.rerun()


def show_kb_list(knowledge_bases: List[Dict]):
    """显示知识库列表"""
    if not knowledge_bases:
        st.info("暂无知识库，请创建一个")
        return

    st.subheader("📚 知识库列表")

    for kb in knowledge_bases:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"### 📁 {kb['name']}")
                st.caption(f"{kb.get('description', '无描述')}")
                st.caption(f"ID: {kb['id']}")

            with col2:
                st.metric("文档数量", kb.get('doc_count', 0))

            with col3:
                if st.button("🗑️ 删除", key=f"delete_{kb['id']}"):
                    if delete_knowledge_base(kb['id']):
                        st.rerun()

            st.divider()


def main():
    """主函数"""
    init_session_state()

    st.title("📚 知识库管理")

    # API 配置
    with st.sidebar:
        st.title("⚙️ 配置")
        api_url = st.text_input("API 地址", value=API_BASE_URL)
        st.session_state.api_base_url = api_url

    # 创建知识库
    show_create_dialog()

    # 显示知识库列表
    knowledge_bases = list_knowledge_bases()
    show_kb_list(knowledge_bases)


if __name__ == "__main__":
    main()
```

**Step 2: 手动测试页面**

```bash
# 启动 API 服务器（需要先完成 Task 7）
cd /e/SOFE/RAG/.worktrees/knowledge-base-management
python -m enterprise_rag.main --mode api &

# 在另一个终端启动前端
streamlit run enterprise_rag/frontend/streamlit/pages/1_Knowledge_Management.py
```

Expected: 页面显示知识库管理界面

**Step 3: 提交**

```bash
git add enterprise_rag/frontend/streamlit/pages/
git commit -m "feat: add knowledge base management page

- Add Streamlit page for managing knowledge bases
- Support create, list, delete operations
- Clean and intuitive UI
```

---

### Task 7: 集成到主应用

**文件:**
- Modify: `enterprise_rag/main.py`
- Modify: `enterprise_rag/frontend/streamlit/app.py`

**Step 1: 更新 main.py 以初始化知识库服务**

```python
# 在 enterprise_rag/main.py 中修改

def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """启动 API 服务器"""
    import os
    from enterprise_rag.api import initialize_service, run_server
    from enterprise_rag.services import KnowledgeBaseService  # 新增

    logger = get_logger(__name__)

    # 加载配置
    env_file = Path(__file__).parent / ".env"
    try:
        load_env_config(env_file=str(env_file), required_keys=['ZHIPUAI_API_KEY'])
    except ValueError as e:
        logger.warning(f"环境变量配置缺失: {e}")

    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "zhipuai")
    milvus_config = MILVUS_CONFIGS['standalone']
    embedding_config = EMBEDDING_CONFIGS.get(embedding_backend, EMBEDDING_CONFIGS['zhipuai'])
    llm_config = LLM_CONFIGS['zhipu']
    llm_config.api_key = os.environ.get("ZHIPUAI_API_KEY", None)

    # 初始化服务
    embedding_service = EmbeddingService(embedding_config)
    retriever = VectorRetriever(
        config=milvus_config,
        embedding_service=embedding_service,
    )

    # 初始化知识库服务（新增）
    kb_service = KnowledgeBaseService(
        embedding_service=embedding_service,
        milvus_config=milvus_config
    )

    initialize_service(
        retriever=retriever,
        llm_config=llm_config,
        use_reranker=True,
        kb_service=kb_service,  # 新增参数
    )

    logger.info(f"启动 API 服务器: http://{host}:{port}")
    run_server(host=host, port=port)
```

**Step 2: 更新前端主页面添加知识库选择器**

```python
# 在 enterprise_rag/frontend/streamlit/app.py 中修改

def init_session_state():
    """初始化会话状态"""
    # ... 现有代码 ...

    # 新增：知识库选择
    if "current_kb_id" not in st.session_state:
        st.session_state.current_kb_id = None

    if "knowledge_bases" not in st.session_state:
        st.session_state.knowledge_bases = []


def load_knowledge_bases():
    """加载知识库列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/knowledge-bases")
        if response.status_code == 200:
            data = response.json()
            st.session_state.knowledge_bases = data.get("knowledge_bases", [])

            # 如果没有选中的知识库，默认选中第一个
            if not st.session_state.current_kb_id and st.session_state.knowledge_bases:
                st.session_state.current_kb_id = st.session_state.knowledge_bases[0]['id']
    except Exception as e:
        st.error(f"加载知识库列表失败: {e}")


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.title("⚙️ 配置")

        # API 地址
        api_url = st.text_input("API 地址", value=API_BASE_URL, key="api_base_url_input")
        st.session_state.api_base_url = api_url

        st.divider()

        # 新增：知识库选择
        st.subheader("📚 知识库")

        # 刷新按钮
        if st.button("🔄 刷新知识库", use_container_width=True):
            load_knowledge_bases()

        # 加载知识库列表
        if not st.session_state.knowledge_bases:
            load_knowledge_bases()

        # 知识库选择器
        if st.session_state.knowledge_bases:
            kb_options = {kb['name']: kb['id'] for kb in st.session_state.knowledge_bases}
            selected_name = st.selectbox(
                "选择知识库",
                options=list(kb_options.keys()),
                index=list(kb_options.values()).index(st.session_state.current_kb_id) if st.session_state.current_kb_id in kb_options.values() else 0
            )
            st.session_state.current_kb_id = kb_options[selected_name]

            st.caption(f"当前: {selected_name}")
        else:
            st.info("暂无知识库，请先创建")
            if st.button("创建知识库"):
                st.switch_page("pages/1_Knowledge_Management.py")

        st.divider()

        # ... 现有设置 ...
```

**Step 3: 提交**

```bash
git add enterprise_rag/main.py enterprise_rag/frontend/streamlit/app.py
git commit -m "feat: integrate knowledge base service into main application

- Initialize KnowledgeBaseService in API server
- Add knowledge base selector to main chat interface
- Auto-load knowledge bases on startup
```

---

## 阶段五：文档上传功能

### Task 8: 添加文档上传 API

**文件:**
- Modify: `enterprise_rag/services/knowledge_base_service.py`
- Modify: `enterprise_rag/api/routers/knowledge_base.py`
- Test: `enterprise_rag/tests/test_kb_service.py`

**Step 1: 添加服务测试**

```python
# 在 enterprise_rag/tests/test_kb_service.py 添加

def test_add_documents_to_kb(mock_embedding_service, mock_milvus_client):
    """测试向知识库添加文档"""
    service = KnowledgeBaseService(embedding_service=mock_embedding_service)

    kb = service.create_knowledge_base(name="测试KB")

    # 模拟文档
    documents = [
        {"content": "测试文档1", "metadata": {"source": "test1.txt"}},
        {"content": "测试文档2", "metadata": {"source": "test2.txt"}}
    ]

    result = service.add_documents(kb.id, documents)

    assert result["total"] == 2
    assert result["succeeded"] == 2
```

**Step 2: 实现文档添加服务**

```python
# 在 enterprise_rag/services/knowledge_base_service.py 添加

from .processors import DocumentLoader, TextSplitter

class KnowledgeBaseService:
    # ... 现有代码 ...

    def add_documents(
        self,
        kb_id: str,
        documents: List[dict]
    ) -> dict:
        """
        向知识库添加文档

        Args:
            kb_id: 知识库 ID
            documents: 文档列表 [{"content": "...", "metadata": {...}}]

        Returns:
            添加结果 {"total": N, "succeeded": M, "failed": L}
        """
        kb = self.metadata.get(kb_id)
        if not kb:
            raise ValueError(f"知识库 {kb_id} 不存在")

        # 获取向量检索器
        from pymilvus import MilvusClient

        uri = self.milvus_config.uri if self.milvus_config else "./data/milvus_test.db"
        client = MilvusClient(uri=uri)

        # 分割文本
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)

        all_chunks = []
        for doc in documents:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)

        # 向量化并存储
        succeeded = 0
        for chunk in all_chunks:
            try:
                # 生成向量
                vector = self.embedding_service.encode_documents(chunk.content)[0]

                # 插入数据
                client.insert(
                    collection_name=kb.collection_name,
                    data=[{
                        "id": str(uuid.uuid4()),
                        "vector": vector,
                        "content": chunk.content,
                        **chunk.metadata
                    }]
                )
                succeeded += 1
            except Exception as e:
                print(f"插入失败: {e}")

        # 更新文档数量
        kb.doc_count += len(documents)
        self.metadata.add(kb)  # 保存更新

        return {
            "total": len(all_chunks),
            "succeeded": succeeded,
            "failed": len(all_chunks) - succeeded
        }
```

**Step 3: 添加 API 路由**

```python
# 在 enterprise_rag/api/routers/knowledge_base.py 添加

from fastapi import UploadFile, File, Form

@kb_router.post("/{kb_id}/documents")
async def upload_documents(
    kb_id: str,
    files: List[UploadFile] = File(...),
    service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    上传文档到知识库

    Args:
        kb_id: 知识库 ID
        files: 上传的文件列表
        service: 知识库服务

    Returns:
        上传结果
    """
    kb = service.get_knowledge_base(kb_id)
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库 {kb_id} 不存在"
        )

    # 临时保存文件
    import tempfile
    import os

    documents = []
    uploaded = 0
    failed = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            try:
                # 保存文件
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, 'wb') as f:
                    f.write(await file.read())

                # 加载文档
                loader = DocumentLoader(base_path=temp_dir)
                docs = loader.load_file(file_path)

                documents.extend([
                    {"content": d.content, "metadata": d.metadata}
                    for d in docs
                ])
                uploaded += 1

            except Exception as e:
                print(f"加载文件失败 {file.filename}: {e}")
                failed += 1

        # 添加到知识库
        result = service.add_documents(kb_id, documents)

    return {
        "uploaded_files": uploaded,
        "failed_files": failed,
        "total_chunks": result["total"],
        "succeeded_chunks": result["succeeded"]
    }
```

**Step 4: 提交**

```bash
git add enterprise_rag/services/knowledge_base_service.py enterprise_rag/api/routers/knowledge_base.py
git commit -m "feat: add document upload functionality

- Support uploading multiple files to knowledge base
- Auto-process documents with existing loaders
- Vectorize and store in Milvus collection
- Update document count in metadata
```

---

## 阶段六：问答支持指定知识库

### Task 9: 支持指定知识库查询

**文件:**
- Modify: `enterprise_rag/api/routers/qa.py`
- Test: `enterprise_rag/tests/test_qa_with_kb.py`

**Step 1: 添加测试**

```python
# enterprise_rag/tests/test_qa_with_kb.py
"""测试问答支持指定知识库"""
import pytest
from unittest.mock import Mock, patch


def test_query_with_kb_id():
    """测试带知识库 ID 的查询"""
    # 实现测试
    pass


def test_query_without_kb_id_uses_default():
    """测试不指定知识库时使用默认值"""
    # 实现测试
    pass
```

**Step 2: 修改 QA 路由**

```python
# 修改 enterprise_rag/api/routers/qa.py

@qa_router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    qa_chain: QAChain = Depends(get_qa_chain),
):
    """执行问答查询"""
    try:
        # 支持 kb_id 参数
        kb_id = getattr(request, 'kb_id', None)

        result = qa_chain.invoke(
            request.question,
            top_k=request.top_k,
            use_rerank=request.use_rerank,
            kb_id=kb_id  # 新增参数
        )

        # ... 现有代码 ...
```

**Step 3: 修改 QueryRequest 模型**

```python
# 修改 enterprise_rag/api/models/schemas.py

class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., min_length=1, description="问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数")
    use_rerank: bool = Field(default=True, description="是否使用重排序")
    kb_id: Optional[str] = Field(default=None, description="知识库 ID")  # 新增
```

**Step 4: 提交**

```bash
git add enterprise_rag/api/routers/qa.py enterprise_rag/api/models/schemas.py
git commit -m "feat: support querying specific knowledge base

- Add kb_id parameter to query request
- Pass kb_id to QA chain for collection filtering
- Maintain backward compatibility (kb_id is optional)
```

---

## 完成和测试

### Task 10: 端到端测试

**Step 1: 启动完整系统**

```bash
# 终端 1: 启动 API 服务器
cd /e/SOFE/RAG/.worktrees/knowledge-base-management
python -m enterprise_rag.main --mode api

# 终端 2: 启动前端
streamlit run enterprise_rag/frontend/streamlit/app.py

# 终端 3: 启动知识库管理页面
streamlit run enterprise_rag/frontend/streamlit/pages/1_Knowledge_Management.py
```

**Step 2: 测试流程**

1. 创建知识库
2. 上传文档
3. 在问答页面切换知识库
4. 执行查询
5. 验证结果来自正确的知识库

**Step 3: 运行所有测试**

```bash
python -m pytest enterprise_rag/tests/ -v
```

**Step 4: 最终提交**

```bash
git add .
git commit -m "feat: complete knowledge base management system

Phase 1: Data models and metadata storage
Phase 2: Knowledge base core service
Phase 3: API routes and models
Phase 4: Frontend management page
Phase 5: Document upload functionality
Phase 6: Query-specific knowledge base support

All tests passing. Ready for review.
```

---

## 实施检查清单

- [ ] Task 0: 验证环境和依赖
- [ ] Task 1: 创建知识库元数据模型
- [ ] Task 2: 添加唯一 ID 生成器
- [ ] Task 3: 创建知识库服务基础
- [ ] Task 4: 创建 API 请求/响应模型
- [ ] Task 5: 创建知识库 API 路由
- [ ] Task 6: 创建知识库管理页面
- [ ] Task 7: 集成到主应用
- [ ] Task 8: 添加文档上传 API
- [ ] Task 9: 支持指定知识库查询
- [ ] Task 10: 端到端测试

---

## 技术说明

### 文档上传处理

文档上传后：
1. 保存到临时目录
2. 使用 `DocumentLoader` 加载
3. 使用 `TextSplitter` 分块
4. 使用 `EmbeddingService` 向量化
5. 插入到指定知识库的 Milvus Collection

### Collection 隔离

每个知识库使用独立的 Milvus Collection：
- Collection 名称: `kb_{kb_id}`
- 查询时根据 `kb_id` 指定 Collection
- 删除知识库时删除对应 Collection

### 元数据持久化

知识库元数据存储在 `data/knowledge_bases.json`：
- 包含所有知识库的信息
- 启动时自动加载
- 操作后自动保存

---

## 参考资料

- 设计文档: `docs/plans/2026-03-03-knowledge-base-management-design.md`
- Milvus 文档: https://milvus.io/docs/
- FastAPI 文档: https://fastapi.tiangolo.com/
- Streamlit 文档: https://docs.streamlit.io/
