"""
FastAPI 主应用
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routers import qa_router
from .models import HealthResponse, ErrorResponse


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 全局状态
_service_components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("服务启动中...")
    _service_components['status'] = 'starting'

    yield

    # 关闭
    logger.info("服务关闭中...")
    _service_components['status'] = 'stopping'


# 创建应用
app = FastAPI(
    title="Enterprise RAG API",
    description="企业级 RAG 知识库问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册路由
app.include_router(qa_router)


@app.get("/", response_model=HealthResponse)
async def root():
    """根路径"""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        components={
            "api": "running",
            "qa_chain": "ready" if _service_components.get("qa_chain") else "not_initialized",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "api": "running",
            "qa_chain": "ready" if _service_components.get("qa_chain") else "not_initialized",
        },
    )


# 异常处理器
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """值错误处理"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="value_error",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail=str(exc),
        ).model_dump(),
    )


def initialize_service(
    retriever,
    llm_config,
    use_reranker: bool = True,
):
    """
    初始化服务

    Args:
        retriever: 检索器
        llm_config: LLM 配置
        use_reranker: 是否使用重排序
    """
    from .routers.qa import initialize_qa_chain

    initialize_qa_chain(retriever, llm_config, use_reranker)

    _service_components['qa_chain'] = 'ready'
    logger.info("服务初始化完成")


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """
    运行服务器

    Args:
        host: 主机地址
        port: 端口
        reload: 是否自动重载
    """
    import uvicorn

    uvicorn.run(
        "enterprise_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    # 开发环境直接运行
    run_server(reload=True)
