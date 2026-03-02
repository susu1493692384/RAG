"""
向量检索器
基于 Milvus 的向量相似度检索
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pymilvus import MilvusClient

from ..config import MilvusConfig
from ..embeddings import EmbeddingService


@dataclass
class SearchResult:
    """搜索结果数据类"""
    id: Any
    score: float
    distance: float
    content: str
    metadata: Dict[str, Any]


class VectorRetriever:
    """向量检索器类"""

    def __init__(
        self,
        config: Optional[MilvusConfig] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        初始化向量检索器

        Args:
            config: Milvus 配置
            embedding_service: Embedding 服务
        """
        self.config = config or MilvusConfig()
        self.embedding_service = embedding_service

        # 连接 Milvus
        self.client = MilvusClient(
            uri=self.config.uri,
            token=self.config.token,
        )

        # 确保 collection 存在
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 collection 存在"""
        if not self.client.has_collection(self.config.collection_name):
            self.client.create_collection(
                collection_name=self.config.collection_name,
                dimension=self.config.dimension,
                enable_dynamic_field=self.config.enable_dynamic_field,
            )
            print(f"创建 collection: {self.config.collection_name}")
        else:
            print(f"连接到 collection: {self.config.collection_name}")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        添加文档到向量库

        Args:
            documents: 文档列表，每个文档包含:
                - content: 文本内容
                - metadata: 元数据字典
            batch_size: 批处理大小

        Returns:
            插入结果
        """
        batch_size = batch_size or self.config.batch_size

        # 准备数据
        data = []
        for i, doc in enumerate(documents):
            # 获取或生成向量
            if 'vector' in doc:
                vector = doc['vector']
            elif self.embedding_service:
                vector = self.embedding_service.encode_documents(doc['content'])[0]
            else:
                raise ValueError("需要提供向量或 embedding_service")

            # 构建记录
            record = {
                'id': doc.get('id', i),
                'vector': vector,
                'text': doc['content'],
            }

            # 添加元数据
            if 'metadata' in doc:
                record.update(doc['metadata'])

            data.append(record)

        # 批量插入
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = self.client.insert(
                collection_name=self.config.collection_name,
                data=batch,
            )
            results.append(result)

        # 刷新数据到磁盘（重要！否则数据可能不会被查询到）
        self.client.flush(self.config.collection_name)

        # MutationResult 是 OmitZeroDict，用键访问
        total_inserted = sum(r.get('insert_count', 0) for r in results)
        return {
            'total': total_inserted,
            'details': results,
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expression: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        向量搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_expression: 过滤表达式
            output_fields: 返回字段列表

        Returns:
            SearchResult 列表
        """
        if not self.embedding_service:
            raise ValueError("需要提供 embedding_service")

        # 编码查询
        query_vectors = self.embedding_service.encode_queries(query)

        # 默认返回字段
        if output_fields is None:
            output_fields = ['text', 'source', 'page']

        # 执行搜索
        results = self.client.search(
            collection_name=self.config.collection_name,
            data=query_vectors,
            limit=top_k,
            filter=filter_expression,
            output_fields=output_fields,
            search_params=self.config.search_params,
        )

        # 转换结果
        search_results = []
        for hit in results[0]:
            search_results.append(SearchResult(
                id=hit['id'],
                score=1.0 - hit['distance'],  # 转换为相似度分数
                distance=hit['distance'],
                content=hit['entity'].get('text', ''),
                metadata=hit['entity'],
            ))

        return search_results

    def delete(
        self,
        ids: Optional[List[Any]] = None,
        filter_expression: Optional[str] = None,
    ) -> List[Any]:
        """
        删除文档

        Args:
            ids: 主键列表
            filter_expression: 过滤表达式

        Returns:
            删除的 ID 列表
        """
        if ids:
            result = self.client.delete(
                collection_name=self.config.collection_name,
                ids=ids,
            )
        elif filter_expression:
            result = self.client.delete(
                collection_name=self.config.collection_name,
                filter=filter_expression,
            )
        else:
            raise ValueError("必须提供 ids 或 filter_expression")

        return result

    def query(
        self,
        filter_expression: str,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        标量查询（不使用向量）

        Args:
            filter_expression: 过滤表达式
            output_fields: 返回字段列表
            limit: 返回结果数量

        Returns:
            结果列表
        """
        results = self.client.query(
            collection_name=self.config.collection_name,
            filter=filter_expression,
            output_fields=output_fields,
            limit=limit,
        )

        return results

    def count(self) -> int:
        """获取文档总数"""
        # 先尝试从 stats 获取
        try:
            stats = self.client.get_collection_stats(
                collection_name=self.config.collection_name
            )
            return stats.get('row_count', 0)
        except Exception:
            # 备用方法：查询全部
            results = self.client.query(
                collection_name=self.config.collection_name,
                output_fields=['id'],
            )
            return len(results)

    def drop_collection(self):
        """删除 collection"""
        self.client.drop_collection(self.config.collection_name)
        print(f"已删除 collection: {self.config.collection_name}")

    def create_index(self):
        """创建索引"""
        from pymilvus import MilvusException

        index_params = self.client.prepare_index_params()

        # 添加向量索引
        index_params.add_index(
            field_name="vector",
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
            params=self.config.index_params,
        )

        try:
            self.client.create_index(
                collection_name=self.config.collection_name,
                index_params=index_params,
            )
            print(f"已创建索引: {self.config.index_type}")
        except MilvusException as e:
            print(f"创建索引失败: {e}")

    def flush(self):
        """强制刷新数据到磁盘"""
        self.client.flush(self.config.collection_name)
