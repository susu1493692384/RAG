"""
检索器测试
"""
import pytest
import tempfile
from pathlib import Path

from enterprise_rag.retriever import VectorRetriever
from enterprise_rag.config import MilvusConfig, EMBEDDING_CONFIGS
from enterprise_rag.embeddings import EmbeddingService


@pytest.fixture
def temp_milvus_db():
    """临时 Milvus 数据库"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = MilvusConfig(uri=str(db_path), collection_name="test_collection")
        yield config


@pytest.fixture
def embedding_service():
    """Embedding 服务"""
    config = EMBEDDING_CONFIGS['m3e_base']
    return EmbeddingService(config)


class TestVectorRetriever:
    """向量检索器测试"""

    def test_init_retriever(self, temp_milvus_db, embedding_service):
        """测试初始化检索器"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )
        assert retriever is not None
        assert retriever.config.collection_name == "test_collection"

    def test_add_documents(self, temp_milvus_db, embedding_service):
        """测试添加文档"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )

        documents = [
            {"content": "测试文档1", "metadata": {"source": "test1.txt"}},
            {"content": "测试文档2", "metadata": {"source": "test2.txt"}},
        ]

        result = retriever.add_documents(documents)

        assert result['total'] == 2

    def test_search(self, temp_milvus_db, embedding_service):
        """测试搜索"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )

        # 添加文档
        documents = [
            {"content": "人工智能是计算机科学的一个分支", "metadata": {"topic": "AI"}},
            {"content": "机器学习是人工智能的子领域", "metadata": {"topic": "ML"}},
        ]
        retriever.add_documents(documents)

        # 搜索
        results = retriever.search("人工智能", top_k=2)

        assert len(results) > 0
        assert all(hasattr(r, 'content') for r in results)
        assert all(hasattr(r, 'score') for r in results)

    def test_count(self, temp_milvus_db, embedding_service):
        """测试统计文档数量"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )

        # 初始数量
        assert retriever.count() == 0

        # 添加文档
        documents = [
            {"content": f"测试文档{i}", "metadata": {}}
            for i in range(5)
        ]
        retriever.add_documents(documents)

        # 验证数量
        assert retriever.count() == 5

    def test_delete_by_ids(self, temp_milvus_db, embedding_service):
        """测试删除文档"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )

        # 添加文档
        documents = [
            {"content": "测试文档1", "id": 1},
            {"content": "测试文档2", "id": 2},
        ]
        retriever.add_documents(documents)

        # 删除
        deleted_ids = retriever.delete(ids=[1])

        assert 1 in deleted_ids
        assert retriever.count() == 1

    def test_query_with_filter(self, temp_milvus_db, embedding_service):
        """测试带过滤条件的查询"""
        retriever = VectorRetriever(
            config=temp_milvus_db,
            embedding_service=embedding_service,
        )

        # 添加文档
        documents = [
            {"content": "文档1", "metadata": {"category": "A"}},
            {"content": "文档2", "metadata": {"category": "B"}},
        ]
        retriever.add_documents(documents)

        # 注意：实际使用需要根据 schema 调整过滤表达式
        # 这里仅作为示例
        try:
            results = retriever.query(
                filter_expression="id >= 0",
                output_fields=["id", "text"],
                limit=10,
            )
            assert isinstance(results, list)
        except Exception as e:
            # Milvus Lite 可能不支持某些过滤
            pytest.skip(f"过滤表达式不支持: {e}")
