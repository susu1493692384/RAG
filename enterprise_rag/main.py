"""
企业 RAG 系统 - 主入口

使用示例:
    # 启动 API 服务
    python -m enterprise_rag.main --mode api

    # 启动 Streamlit 前端
    python -m enterprise_rag.main --mode frontend

    # 构建知识库
    python -m enterprise_rag.main --mode build
"""
import argparse
import sys
from pathlib import Path

from enterprise_rag.utils import load_env_config, setup_logging, get_logger
from enterprise_rag.config import (
    EmbeddingConfig,
    LLMConfig,
    MilvusConfig,
    EMBEDDING_CONFIGS,
    LLM_CONFIGS,
    MILVUS_CONFIGS,
)
from enterprise_rag.embeddings import EmbeddingService
from enterprise_rag.retriever import VectorRetriever
from enterprise_rag.processors import DocumentLoader, TextSplitter


def build_knowledge_base(
    data_dir: str = None,
    config_name: str = "standalone",
):
    """
    构建知识库

    Args:
        data_dir: 文档目录，默认为项目数据目录
        config_name: Milvus 配置名称（lite/standalone/distributed）
    """
    import os

    # 设置默认数据目录
    if data_dir is None:
        data_dir = str(Path(__file__).parent / "data" / "documents")

    logger = get_logger(__name__)
    logger.info("开始构建知识库...")

    # 先加载 .env 文件（使用绝对路径）
    env_file = Path(__file__).parent / ".env"
    load_env_config(env_file=str(env_file))

    # 从环境变量读取 embedding backend 选择
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "zhipuai")

    # 加载配置
    milvus_config = MILVUS_CONFIGS.get(config_name, MilvusConfig())
    embedding_config = EMBEDDING_CONFIGS.get(embedding_backend, EMBEDDING_CONFIGS['zhipuai'])

    # 初始化服务
    logger.info("初始化 Embedding 服务...")
    embedding_service = EmbeddingService(embedding_config)

    logger.info("初始化检索器...")
    retriever = VectorRetriever(
        config=milvus_config,
        embedding_service=embedding_service,
    )

    # 加载文档
    logger.info(f"加载文档从: {data_dir}")
    loader = DocumentLoader(base_path=data_dir)
    documents = loader.load_directory()

    logger.info(f"加载了 {len(documents)} 个文档")

    # 分割文本
    logger.info("分割文本...")
    splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([
        {'page_content': d.content, 'metadata': d.metadata}
        for d in documents
    ])

    logger.info(f"分割成 {len(chunks)} 个文本块")

    # 添加到向量库
    logger.info("向量化并存储...")
    result = retriever.add_documents([
        {'content': c.content, 'metadata': c.metadata}
        for c in chunks
    ])

    logger.info(f"成功添加 {result['total']} 个文档块")
    logger.info("知识库构建完成！")


def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    启动 API 服务器

    Args:
        host: 主机地址
        port: 端口
    """
    import os
    from enterprise_rag.api import initialize_service, run_server

    logger = get_logger(__name__)

    # 加载配置（使用绝对路径）
    env_file = Path(__file__).parent / ".env"
    try:
        load_env_config(env_file=str(env_file), required_keys=['ZHIPUAI_API_KEY'])
    except ValueError as e:
        logger.warning(f"环境变量配置缺失: {e}")
        logger.warning("API 将使用模拟模式")

    # 从环境变量读取 embedding backend 选择
    embedding_backend = os.environ.get("EMBEDDING_BACKEND", "zhipuai")

    # 初始化服务
    milvus_config = MILVUS_CONFIGS['standalone']
    embedding_config = EMBEDDING_CONFIGS.get(embedding_backend, EMBEDDING_CONFIGS['zhipuai'])

    # 加载 LLM 配置并设置 API key（使用智谱AI）
    llm_config = LLM_CONFIGS['zhipu']
    llm_config.api_key = os.environ.get("ZHIPUAI_API_KEY", None)

    embedding_service = EmbeddingService(embedding_config)
    retriever = VectorRetriever(
        config=milvus_config,
        embedding_service=embedding_service,
    )

    initialize_service(
        retriever=retriever,
        llm_config=llm_config,
        use_reranker=True,
    )

    # 启动服务器
    logger.info(f"启动 API 服务器: http://{host}:{port}")
    run_server(host=host, port=port)


def start_frontend():
    """启动 Streamlit 前端"""
    import subprocess

    logger = get_logger(__name__)
    logger.info("启动 Streamlit 前端...")

    frontend_path = Path(__file__).parent / "frontend" / "streamlit" / "app.py"

    subprocess.run([
        "streamlit",
        "run",
        str(frontend_path),
    ])


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="企业 RAG 知识库系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["api", "frontend", "build"],
        default="api",
        help="运行模式",
    )

    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent / "data" / "documents"),
        help="文档目录（用于 build 模式）",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API 服务器主机地址",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 服务器端口",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(level=getattr(__import__("logging"), args.log_level))

    # 执行对应模式
    if args.mode == "build":
        build_knowledge_base(args.data_dir)
    elif args.mode == "api":
        start_api_server(args.host, args.port)
    elif args.mode == "frontend":
        start_frontend()


if __name__ == "__main__":
    main()
