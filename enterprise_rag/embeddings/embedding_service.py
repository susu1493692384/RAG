"""
Embedding 服务
支持多种 Embedding 方式（本地模型和API）
"""
from typing import List, Union, Optional
from pathlib import Path
import hashlib
import pickle
import requests

from ..config import EmbeddingConfig, EMBEDDING_CONFIGS


class ZhipuEmbeddingService:
    """智谱AI Embedding API 服务"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        """
        初始化智谱AI客户端

        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        编码文档

        Args:
            texts: 文本列表

        Returns:
            向量列表
        """
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }

        response = requests.post(
            f"{self.base_url}",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            embeddings = result.get("data", [])
            # 提取向量（API返回格式：{"data": [{"embedding": [...], "index": 0}, ...]}）
            return [item["embedding"] for item in embeddings]
        else:
            raise Exception(f"智谱AI API错误: {response.status_code}, {response.text}")

    def encode_queries(self, queries: List[str]) -> List[List[float]]:
        """
        编码查询（与文档编码相同）

        Args:
            queries: 查询列表

        Returns:
            向量列表
        """
        return self.encode_documents(queries)


class EmbeddingService:
    """Embedding 服务类（支持本地模型和API）"""

    def __init__(self, config: Optional[EmbeddingConfig] = None, use_api: bool = False):
        """
        初始化 Embedding 服务

        Args:
            config: Embedding配置
            use_api: 是否使用API而非本地模型
        """
        import os

        self.config = config or EmbeddingConfig()
        self._model = None
        self._cache = {}
        self._use_api = use_api  # 兼容手动指定
        self._api_service = None

        # 创建缓存目录
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # 🔥 新增：从环境变量读取 backend 选择
        backend = os.environ.get("EMBEDDING_BACKEND", "").lower()

        # 根据 backend 决定是否使用 API
        if backend == "zhipuai":
            print("检测到 EMBEDDING_BACKEND=zhipuai，使用智谱AI API")
            self._use_api = True
            self._init_api_service()
        elif backend:
            print(f"检测到 EMBEDDING_BACKEND={backend}")
            # 其他 backend 可以在这里添加
        else:
            print("未设置 EMBEDDING_BACKEND，使用默认本地模型")

        # 兼容手动指定 use_api（优先级低于环境变量）
        if use_api and not self._use_api:
            print("手动指定 use_api=True，覆盖环境变量设置")
            self._use_api = True
            self._init_api_service()

    def _init_api_service(self):
        """初始化API服务"""
        try:
            import os
            api_key = os.environ.get("ZHIPUAI_API_KEY")
            if not api_key:
                raise ValueError("使用API模式需要设置 ZHIPUAI_API_KEY 环境变量")

            self._api_service = ZhipuEmbeddingService(api_key)
            print("使用智谱AI Embedding API")
        except ImportError:
            print("requests 未安装")
            raise ImportError("需要安装 requests: pip install requests")

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """加载模型"""
        # 如果使用API，不需要加载本地模型
        if self._use_api:
            return self._api_service

        model_name = self.config.model_name.lower()

        # 使用智谱AI
        if model_name == 'zhipu':
            return self._api_service

        # 使用pymilvus内置模型
        if model_name in ['bge-m3', 'default']:
            try:
                from pymilvus import model as milvus_model
                print("加载pymilvus默认模型: paraphrase-albert-small-v2")
                return milvus_model.DefaultEmbeddingFunction()
            except ImportError:
                print("pymilvus不可用")
                raise ImportError("需要安装pymilvus")

        # 使用sentence-transformers
        if model_name in ['m3e-large', 'm3e-base']:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"加载模型: {self.config.model_path}")
                model = SentenceTransformer(
                    self.config.model_path,
                    device=self.config.device,
                )
                return model
            except ImportError:
                raise ImportError(
                    "需要安装 sentence-transformers: "
                    "pip install sentence-transformers"
                )

        # 使用FlagEmbedding
        if 'bge' in model_name:
            try:
                from FlagEmbedding import FlagEmbedding
                print(f"加载模型: {self.config.model_path}")
                model = FlagEmbedding(
                    model_name_or_path=self.config.model_path,
                    normalization=self.config.normalize_embeddings,
                    device=self.config.device,
                )
                return model
            except ImportError:
                raise ImportError("需要安装 FlagEmbedding: pip install FlagEmbedding")

        raise ValueError(f"不支持的模型: {model_name}")

    def encode_documents(
        self,
        documents: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        编码文档

        Args:
            documents: 文档或文档列表
            batch_size: 批处理大小

        Returns:
            向量列表
        """
        batch_size = batch_size or self.config.batch_size

        # 转换为列表
        if isinstance(documents, str):
            documents = [documents]

        # 如果使用API，调用API服务
        if self._use_api:
            return self._api_service.encode_documents(documents)

        # 检查缓存
        vectors = []
        uncached_docs = []
        uncached_indices = []

        for i, doc in enumerate(documents):
            cache_key = self._get_cache_key(doc)
            if cache_key in self._cache:
                vectors.append((i, self._cache[cache_key]))
            else:
                uncached_docs.append(doc)
                uncached_indices.append(i)

        # 编码未缓存的文档
        if uncached_docs:
            new_vectors = self._encode_batch(uncached_docs, batch_size)

            for idx, vector in enumerate(new_vectors):
                orig_idx = uncached_indices[idx]
                cache_key = self._get_cache_key(uncached_docs[idx])
                self._cache[cache_key] = vector
                vectors.append((orig_idx, vector))

        # 按原始顺序排序
        vectors.sort(key=lambda x: x[0])
        return [v[1] for v in vectors]

    def encode_queries(
        self,
        queries: Union[str, List[str]],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        编码查询

        Args:
            queries: 查询或查询列表
            batch_size: 批处理大小

        Returns:
            向量列表
        """
        # 如果使用API
        if self._use_api:
            return self._api_service.encode_queries(
                queries if isinstance(queries, list) else [queries]
            )

        # 对于大多数本地模型，文档和查询使用相同编码方式
        return self.encode_documents(
            queries if isinstance(queries, list) else [queries],
            batch_size
        )

    def _encode_batch(
        self,
        texts: List[str],
        batch_size: int,
    ) -> List[List[float]]:
        """批量编码"""
        # 根据模型类型选择编码方式
        if hasattr(self._model, 'encode'):
            # sentence-transformers 或 FlagEmbedding
            vectors = self._model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=len(texts) > 100,
            )
            return vectors.tolist() if hasattr(vectors, 'tolist') else list(vectors)
        elif hasattr(self._model, 'encode_documents'):
            # pymilvus model
            return self._model.encode_documents(texts)
        else:
            raise ValueError(f"不支持的模型类型: {type(self._model)}")

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        return hashlib.md5(text.encode()).hexdigest()

    def save_cache(self, path: Optional[str] = None):
        """保存缓存到文件"""
        cache_path = path or str(
            Path(self.config.cache_dir) / "embedding_cache.pkl"
        )
        with open(cache_path, 'wb') as f:
            pickle.dump(self._cache, f)

    def load_cache(self, path: Optional[str] = None):
        """从文件加载缓存"""
        cache_path = path or str(
            Path(self.config.cache_dir) / "embedding_cache.pkl"
        )
        if Path(cache_path).exists():
            with open(cache_path, 'rb') as f:
                self._cache = pickle.load(f)

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def get_dimension(self) -> int:
        """获取向量维度"""
        if self.config.dimension:
            return self.config.dimension

        # 测试编码获取维度
        test_vector = self.encode_documents("测试")[0]
        return len(test_vector)


class CachedEmbeddingService(EmbeddingService):
    """带文件缓存的 Embedding 服务"""

    def __init__(self, config: Optional[EmbeddingConfig] = None, cache_file: str = None, use_api: bool = False):
        super().__init__(config, use_api)
        self.cache_file = cache_file or str(
            Path(self.config.cache_dir) / "embedding_cache.pkl"
        )
        self.load_cache()

    def encode_documents(self, documents, batch_size=None):
        """编码文档并自动保存缓存"""
        vectors = super().encode_documents(documents, batch_size)
        self.save_cache()
        return vectors


# 便捷函数service = create_embedding_service(use_api=True, api_key='your_key')
def create_embedding_service(
    model_name: str = 'm3e_large',
    device: str = 'cpu',
    normalize: bool = True,
    use_api: bool = False,
    api_key: Optional[str] = None,
) -> EmbeddingService:
    """
    创建 Embedding 服务

    Args:
        model_name: 模型名称
        device: 设备
        normalize: 是否归一化
        use_api: 是否使用API模式
        api_key: API密钥（使用API时必需）

    Returns:
        EmbeddingService实例
    """
    from ..config import EMBEDDING_CONFIGS

    # 使用API模式
    if use_api:
        if not api_key:
            raise ValueError("使用API模式需要提供api_key")

        config = EmbeddingConfig(
            model_name='zhipu',
            device=device,
            normalize_embeddings=normalize,
        )
        return EmbeddingService(config, use_api=True)

    # 使用预定义配置
    if model_name in EMBEDDING_CONFIGS:
        config = EMBEDDING_CONFIGS[model_name]
        return EmbeddingService(config)

    # 自定义配置
    config = EmbeddingConfig(
        model_name=model_name,
        device=device,
        normalize_embeddings=normalize,
    )
    return EmbeddingService(config)


# LangChain 集成
try:
    from langchain_core.embeddings import Embeddings as LangchainEmbeddings

    class LangchainEmbeddingWrapper(LangchainEmbeddings):
        """LangChain Embedding 包装器"""

        def __init__(self, service: EmbeddingService):
            self.service = service

        def embed_documents(self, texts):
            vectors = self.service.encode_documents(texts)
            # 转换为langchain格式
            return [v for v in vectors]

        def embed_query(self, text):
            vectors = self.service.encode_queries(text)
            return vectors[0] if vectors else []

except ImportError:
    LangchainEmbeddingsWrapper = None
