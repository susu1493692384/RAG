"""
问答链测试
"""
import pytest
import tempfile
from pathlib import Path

from enterprise_rag.chains import QAChain, create_qa_chain
from enterprise_rag.retriever import VectorRetriever
from enterprise_rag.config import MilvusConfig, LLMConfig, EMBEDDING_CONFIGS
from enterprise_rag.embeddings import EmbeddingService


@pytest.fixture
def temp_retriever():
    """临时检索器"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        config = MilvusConfig(
            uri=str(db_path),
            collection_name="test_collection"
        )
        embedding_config = EMBEDDING_CONFIGS['m3e_base']
        embedding_service = EmbeddingService(embedding_config)

        retriever = VectorRetriever(
            config=config,
            embedding_service=embedding_service,
        )

        # 添加测试文档
        documents = [
            {
                "content": "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。",
                "metadata": {"source": "test1.txt"}
            },
            {
                "content": "JavaScript 是一种脚本语言，主要用于 Web 开发。",
                "metadata": {"source": "test2.txt"}
            },
            {
                "content": "Java 是一种面向对象的编程语言，由 Sun Microsystems 开发。",
                "metadata": {"source": "test3.txt"}
            },
        ]
        retriever.add_documents(documents)

        yield retriever


@pytest.fixture
def llm_config():
    """LLM 配置（使用模拟）"""
    return LLMConfig(
        provider="deepseek",
        base_url="http://localhost:8000",  # Mock URL
        model_name="deepseek-chat",
        temperature=0.1,
    )


class TestQAChain:
    """问答链测试"""

    def test_create_qa_chain(self, temp_retriever, llm_config):
        """测试创建问答链"""
        chain = create_qa_chain(
            retriever=temp_retriever,
            llm_config=llm_config,
            conversational=False,
        )

        assert chain is not None
        assert isinstance(chain, QAChain)

    @pytest.mark.skipif(
        "not os.environ.get('DEEPSEEK_API_KEY')",
        reason="需要 DEEPSEEK_API_KEY 环境变量"
    )
    def test_qa_invoke(self, temp_retriever):
        """测试问答调用（需要真实 API）"""
        from enterprise_rag.config import LLM_CONFIGS

        chain = create_qa_chain(
            retriever=temp_retriever,
            llm_config=LLM_CONFIGS['deepseek'],
        )

        result = chain.invoke("什么是 Python？")

        assert result.answer is not None
        assert len(result.sources) > 0
        assert result.query == "什么是 Python？"

    def test_qa_with_mock_llm(self, temp_retriever, llm_config):
        """测试问答调用（模拟 LLM）"""
        # 这里可以 mock LLM 响应
        chain = QAChain(
            retriever=temp_retriever,
            llm_config=llm_config,
        )

        # 由于 LLM 不可用，这里仅测试检索部分
        search_results = temp_retriever.search("Python", top_k=3)

        assert len(search_results) > 0
        assert all(hasattr(r, 'content') for r in search_results)
