"""
Embedding 服务测试
"""
import pytest
from pathlib import Path

from enterprise_rag.embeddings import EmbeddingService, create_embedding_service
from enterprise_rag.config import EmbeddingConfig, EMBEDDING_CONFIGS


class TestEmbeddingService:
    """Embedding 服务测试类"""

    def test_create_embedding_service(self):
        """测试创建 Embedding 服务"""
        # 使用 pymilvus 默认模型
        service = create_embedding_service(model_name='bge-m3')
        assert service is not None
        assert service.config.model_name == 'bge-m3'

    def test_encode_documents(self):
        """测试编码文档"""
        service = create_embedding_service()

        documents = ["测试文档1", "测试文档2"]
        vectors = service.encode_documents(documents)

        assert len(vectors) == 2
        assert all(isinstance(v, list) for v in vectors)
        assert all(isinstance(x, float) for v in vectors for x in v)

    def test_encode_queries(self):
        """测试编码查询"""
        service = create_embedding_service()

        queries = ["测试查询"]
        vectors = service.encode_queries(queries)

        assert len(vectors) == 1
        assert len(vectors[0]) == service.get_dimension()

    def test_get_dimension(self):
        """测试获取向量维度"""
        config = EMBEDDING_CONFIGS['bge_m3']
        service = EmbeddingService(config)

        dimension = service.get_dimension()
        assert dimension == 1024


@pytest.mark.skipif(
    "not FlagEmbedding",
    reason="需要安装 FlagEmbedding"
)
class TestFlagEmbedding:
    """FlagEmbedding 测试"""

    def test_bge_model_loading(self):
        """测试 BGE 模型加载"""
        from enterprise_rag.config import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="bge-m3",
            model_path="BAAI/bge-m3",
        )

        service = EmbeddingService(config)
        assert service.model is not None

    def test_encode_with_bge(self):
        """测试使用 BGE 编码"""
        from enterprise_rag.config import EmbeddingConfig

        config = EmbeddingConfig(
            model_name="bge-m3",
            model_path="BAAI/bge-m3",
        )

        service = EmbeddingService(config)
        vectors = service.encode_documents(["测试文本"])

        assert len(vectors) == 1
        assert len(vectors[0]) == 1024
