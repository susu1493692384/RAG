"""
混合检索器
结合向量检索和全文检索
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class HybridSearchResult:
    """混合搜索结果"""
    id: Any
    score: float
    content: str
    metadata: Dict[str, Any]
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None


class HybridRetriever:
    """混合检索器类"""

    def __init__(
        self,
        vector_retriever,
        use_rerank: bool = True,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """
        初始化混合检索器

        Args:
            vector_retriever: 向量检索器
            use_rerank: 是否使用重排序
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_retriever = vector_retriever
        self.use_rerank = use_rerank
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        # BM25 索引（延迟初始化）
        self._bm25_index = None
        self._corpus = []

    def _build_bm25_index(self, documents: List[Dict[str, Any]]):
        """构建 BM25 索引"""
        try:
            from rank_bm25 import BM25Okapi
            import jieba
        except ImportError:
            raise ImportError(
                "需要安装 rank_bm25 和 jieba: "
                "pip install rank-bm25 jieba"
            )

        self._corpus = [doc['content'] for doc in documents]

        # 使用 jieba 分词
        tokenized_corpus = [
            list(jieba.cut(doc))
            for doc in self._corpus
        ]

        self._bm25_index = BM25Okapi(tokenized_corpus)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        添加文档并更新索引

        Args:
            documents: 文档列表
        """
        # 添加到向量检索器
        self.vector_retriever.add_documents(documents)

        # 重建 BM25 索引
        existing_docs = self._get_existing_documents()
        all_docs = existing_docs + documents
        self._build_bm25_index(all_docs)

    def _get_existing_documents(self) -> List[Dict[str, Any]]:
        """获取已存在的文档"""
        try:
            results = self.vector_retriever.query(
                filter_expression="id >= 0",
                output_fields=['id', 'text'],
            )
            return [
                {'id': r['id'], 'content': r.get('text', '')}
                for r in results
            ]
        except Exception:
            return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expression: Optional[str] = None,
        rerank_top_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """
        混合搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_expression: 过滤表达式
            rerank_top_k: 重排序后返回的数量

        Returns:
            HybridSearchResult 列表
        """
        # 向量检索
        vector_results = self.vector_retriever.search(
            query=query,
            top_k=top_k * 2,  # 获取更多候选
            filter_expression=filter_expression,
        )

        # BM25 关键词检索
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        # 融合结果（RRF 算法）
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60,
        )

        # 截取 top_k
        results = fused_results[:top_k]

        # 重排序
        if self.use_rerank and rerank_top_k:
            results = self._rerank(query, results, rerank_top_k)

        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """BM25 关键词检索"""
        if self._bm25_index is None:
            return []

        try:
            import jieba
        except ImportError:
            return []

        # 分词
        tokenized_query = list(jieba.cut(query))

        # BM25 检索
        scores = self._bm25_index.get_scores(tokenized_query)

        # 获取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'id': idx,
                    'score': float(scores[idx]),
                    'content': self._corpus[idx],
                })

        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: List,
        keyword_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[HybridSearchResult]:
        """
        倒数排名融合（RRF）

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            k: RRF 常数

        Returns:
            融合后的结果列表
        """
        # 创建 ID 到结果的映射
        keyword_map = {r['id']: r for r in keyword_results}

        # 计算每个文档的 RRF 分数
        scores = {}

        # 向量检索排名
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.id
            scores[doc_id] = scores.get(doc_id, 0) + (
                self.vector_weight / (k + rank)
            )

        # 关键词检索排名
        for rank, result in enumerate(keyword_results, 1):
            doc_id = result['id']
            scores[doc_id] = scores.get(doc_id, 0) + (
                self.keyword_weight / (k + rank)
            )

        # 排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建结果
        results = []
        for doc_id in sorted_ids:
            # 从向量结果获取基本信息
            vec_result = next(
                (r for r in vector_results if r.id == doc_id),
                None
            )

            if vec_result:
                results.append(HybridSearchResult(
                    id=doc_id,
                    score=scores[doc_id],
                    content=vec_result.content,
                    metadata=vec_result.metadata,
                    vector_score=vec_result.score,
                    keyword_score=keyword_map.get(doc_id, {}).get('score'),
                ))

        return results

    def _rerank(
        self,
        query: str,
        results: List[HybridSearchResult],
        top_k: int,
    ) -> List[HybridSearchResult]:
        """
        重排序

        Args:
            query: 查询文本
            results: 候选结果
            top_k: 返回数量

        Returns:
            重排序后的结果
        """
        # TODO: 可以集成 CrossEncoder 等重排序模型
        # 这里简化实现：根据 metadata 排序
        return results[:top_k]

    def count(self) -> int:
        """获取文档总数"""
        return self.vector_retriever.count()

    def delete(
        self,
        ids: Optional[List[Any]] = None,
        filter_expression: Optional[str] = None,
    ):
        """删除文档"""
        self.vector_retriever.delete(ids, filter_expression)
        # 需要重建 BM25 索引
        self._build_bm25_index(self._get_existing_documents())
