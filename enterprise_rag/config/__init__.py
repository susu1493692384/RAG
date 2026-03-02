"""
配置模块
"""
from .embeddings import EmbeddingConfig, EMBEDDING_CONFIGS
from .llm import LLMConfig, LLM_CONFIGS
from .milvus import MilvusConfig, MILVUS_CONFIGS

# 智谱AI（可选，需要安装 zhipuai）
try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZHIPUAI_AVAILABLE = False

__all__ = [
    'EmbeddingConfig',
    'LLMConfig',
    'MilvusConfig',
    'EMBEDDING_CONFIGS',
    'LLM_CONFIGS',
    'MILVUS_CONFIGS',
    'ZhipuAI',
    'ZHIPUAI_AVAILABLE',
]
