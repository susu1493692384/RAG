"""
重排序器
对检索结果进行二次排序，提高相关性
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RerankResult:
    """重排序结果"""
    content: str
    score: float
    rank: int
    metadata: Dict[str, Any]


class Reranker:
    """重排序器类"""

    def __init__(
        self,
        model_name: str = 'bge-reranker-v2-m3',
        device: str = 'cpu',
        batch_size: int = 32,
    ):
        """
        初始化重排序器

        Args:
            model_name: 重排序模型名称
            device: 运行设备
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载重排序模型"""
        # 使用 FlagEmbedding 的 Reranker
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                device=self.device,
            )
            print(f"加载重排序模型: {self.model_name}")
        except ImportError:
            print(
                "FlagEmbedding 不可用，重排序功能将被禁用。"
                "安装命令: pip install -U FlagEmbedding"
            )
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表，每个文档包含:
                - content: 文本内容
                - score: 初始分数
                - metadata: 元数据
            top_k: 返回前 k 个结果

        Returns:
            RerankResult 列表
        """
        if not documents:
            return []

        if self.model is None:
            # 模型不可用，返回原始结果
            return [
                RerankResult(
                    content=doc.get('content', ''),
                    score=doc.get('score', 0),
                    rank=i,
                    metadata=doc.get('metadata', {}),
                )
                for i, doc in enumerate(documents)
            ]

        # 准备查询-文档对
        pairs = [[query, doc.get('content', '')] for doc in documents]

        # 计算重排序分数
        scores = self.model.compute_score(pairs)

        # 按分数排序
        sorted_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # 截取 top_k
        if top_k:
            sorted_docs = sorted_docs[:top_k]

        # 构建结果
        results = []
        for rank, (doc, score) in enumerate(sorted_docs):
            results.append(RerankResult(
                content=doc.get('content', ''),
                score=float(score),
                rank=rank,
                metadata=doc.get('metadata', {}),
            ))

        return results

    def rerank_search_results(
        self,
        query: str,
        search_results: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """
        重排序搜索结果（兼容多种搜索结果格式）

        Args:
            query: 查询文本
            search_results: 搜索结果列表
            top_k: 返回前 k 个结果

        Returns:
            重排序后的搜索结果列表
        """
        if not search_results:
            return []

        # 转换为统一格式
        documents = []
        for result in search_results:
            if hasattr(result, 'content'):
                # SearchResult 对象
                documents.append({
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata,
                    '_original': result,
                })
            elif isinstance(result, dict):
                # 字典格式
                documents.append({
                    'content': result.get('text', result.get('content', '')),
                    'score': result.get('score', 0),
                    'metadata': result,
                    '_original': result,
                })

        # 重排序
        reranked = self.rerank(query, documents, top_k)

        # 返回原始格式
        return [r.metadata.get('_original', r) for r in reranked]


class SimpleReranker(Reranker):
    """简单重排序器（基于关键词匹配）"""

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """基于关键词匹配的重排序"""
        try:
            import jieba
        except ImportError:
            # jieba 不可用，返回原始顺序
            return [
                RerankResult(
                    content=doc.get('content', ''),
                    score=doc.get('score', 0),
                    rank=i,
                    metadata=doc.get('metadata', {}),
                )
                for i, doc in enumerate(documents)
            ]

        # 分词
        query_tokens = set(jieba.cut(query))

        # 计算关键词匹配分数
        scored_docs = []
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            content_tokens = set(jieba.cut(content))

            # 计算重叠度
            overlap = len(query_tokens & content_tokens)
            score = overlap / len(query_tokens) if query_tokens else 0

            # 结合原始分数
            original_score = doc.get('score', 0)
            combined_score = 0.7 * original_score + 0.3 * score

            scored_docs.append((doc, combined_score, i))

        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 截取 top_k
        if top_k:
            scored_docs = scored_docs[:top_k]

        # 构建结果
        results = []
        for doc, score, rank in scored_docs:
            results.append(RerankResult(
                content=doc.get('content', ''),
                score=score,
                rank=rank,
                metadata=doc.get('metadata', {}),
            ))

        return results
