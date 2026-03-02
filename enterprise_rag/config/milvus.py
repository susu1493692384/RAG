"""
Milvus 配置
"""
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class MilvusConfig:
    """Milvus 向量数据库配置类"""

    # 连接 URI
    # 选项 1：本地文件模式（需要 pymilvus[milvus_lite]）
    # uri: str = "./data/milvus_data.db"

    # 选项 2：远程服务器模式（需要运行 Milvus 服务器）
    # uri: str = "http://localhost:19530"

    # 当前默认使用远程模式（更稳定）
    uri: str = "http://localhost:19530"

    # 认证令牌（远程连接）
    token: Optional[str] = None

    # Collection 名称
    collection_name: str = "knowledge_base"

    # 向量维度（需要与 embedding 模型匹配）
    dimension: int = 1024

    # 索引类型
    index_type: str = "HNSW"  # FLAT, IVF_FLAT, IVF_SQ8, HNSW, DISKANN

    # 距离度量类型
    metric_type: str = "COSINE"  # L2, IP, COSINE

    # 索引参数
    index_params: dict = None

    # 搜索参数
    search_params: dict = None

    # 是否启用动态字段
    enable_dynamic_field: bool = True

    # 一致性级别
    consistency_level: str = "Session"  # Strong, Session, Bounded, Eventually, Customized

    # 超时时间（秒）
    timeout: int = 30

    # 批处理大小
    batch_size: int = 100

    def __post_init__(self):
        """初始化后设置默认索引和搜索参数"""
        if self.index_params is None:
            self.index_params = self._get_default_index_params()

        if self.search_params is None:
            self.search_params = self._get_default_search_params()

    def _get_default_index_params(self) -> dict:
        """获取默认索引参数"""
        params_map = {
            "FLAT": {},
            "IVF_FLAT": {"nlist": 128},
            "IVF_SQ8": {"nlist": 128},
            "HNSW": {
                "M": 16,
                "efConstruction": 256,
            },
            "DISKANN": {},
        }
        return params_map.get(self.index_type, {"M": 16})

    def _get_default_search_params(self) -> dict:
        """获取默认搜索参数"""
        params_map = {
            "FLAT": {},
            "IVF_FLAT": {"nprobe": 16},
            "IVF_SQ8": {"nprobe": 16},
            "HNSW": {"ef": 64},
            "DISKANN": {"search_list_size": 100},
        }
        return params_map.get(self.index_type, {})

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MilvusConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "uri": self.uri,
            "token": self.token,
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "enable_dynamic_field": self.enable_dynamic_field,
            "consistency_level": self.consistency_level,
            "timeout": self.timeout,
            "batch_size": self.batch_size,
        }


# 预定义配置
MILVUS_CONFIGS = {
    "lite": MilvusConfig(
        uri=str(Path(__file__).parent.parent / "data" / "milvus_data.db"),
        collection_name="knowledge_base",
        dimension=2048,  # 匹配智谱AI embedding-3 的维度
        index_type="HNSW",
        metric_type="COSINE",
    ),
    "standalone": MilvusConfig(
        uri="http://localhost:19530",
        token="root:Milvus",
        collection_name="knowledge_base",
        dimension=2048,  # 匹配智谱AI embedding-3 的维度
        index_type="HNSW",
        metric_type="COSINE",
    ),
    "distributed": MilvusConfig(
        uri="http://milvus-server:19530",
        collection_name="knowledge_base",
        dimension=2048,  # 匹配智谱AI embedding-3 的维度
        index_type="IVF_FLAT",
        metric_type="COSINE",
    ),
}
