"""
Embedding 模型配置
"""
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置类"""

    # 模型选择
    model_name: str = "zhipuai"  # 可选: zhipuai, bge-m3, m3e-large, m3e-base

    # 模型路径（本地或 HuggingFace）
    model_path: str = "BAAI/bge-m3"

    # 设备配置
    device: str = "cpu"  # cpu 或 cuda

    # 批处理大小
    batch_size: int = 32

    # 向量维度（根据模型自动设置）
    dimension: Optional[int] = None  # bge-m3: 1024, m3e-large: 1024

    # 是否使用 FP16
    use_fp16: bool = False

    # 最大序列长度
    max_length: int = 512

    # 缓存配置
    cache_dir: str = "./data/cache/embeddings"

    # 归一化向量（提高相似度计算）
    normalize_embeddings: bool = True

    def __post_init__(self):
        """初始化后设置默认维度和缓存目录"""
        # 设置默认缓存目录为绝对路径
        if self.cache_dir == "./data/cache/embeddings":
            self.cache_dir = str(Path(__file__).parent.parent / "data" / "cache" / "embeddings")

        if self.dimension is None:
            # 根据模型设置默认维度
            dimension_map = {
                "bge-m3": 1024,
                "m3e-large": 1024,
                "m3e-base": 768,
                "bge-large-zh": 1024,
                "text2vec-base-chinese": 768,
            }
            self.dimension = dimension_map.get(
                self.model_name,
                768  # 默认维度
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EmbeddingConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
            "dimension": self.dimension,
            "use_fp16": self.use_fp16,
            "max_length": self.max_length,
            "cache_dir": self.cache_dir,
            "normalize_embeddings": self.normalize_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EmbeddingConfig':
        """从字典创建配置"""
        return cls(**config_dict)

# 预定义配置
EMBEDDING_CONFIGS = {
    "zhipuai": EmbeddingConfig(
        model_name="zhipuai",
        model_path="embedding-3",
        dimension=2048,  # 智谱AI embedding-3 模型返回 2048 维向量
        normalize_embeddings=True,
    ),
    "bge_m3": EmbeddingConfig(
        model_name="bge-m3",
        model_path="BAAI/bge-m3",
        dimension=1024,
        normalize_embeddings=True,
    ),
    "m3e_large": EmbeddingConfig(
        model_name="m3e-large",
        model_path="moka-ai/m3e-large",
        dimension=1024,
        normalize_embeddings=True,
    ),
    "m3e_base": EmbeddingConfig(
        model_name="m3e-base",
        model_path="moka-ai/m3e-base",
        dimension=768,
        normalize_embeddings=True,
    ),
}

__all__ = [
    'EmbeddingConfig',
    'EMBEDDING_CONFIGS',
]
