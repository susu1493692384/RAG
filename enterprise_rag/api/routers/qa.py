"""
问答路由
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List

from ..models import QueryRequest, QueryResponse, SourceInfo
from ...chains import QAChain, ConversationalQAChain
from ...retriever import VectorRetriever
from ...config import LLMConfig


# 全局依赖（在实际应用中应使用依赖注入）
_qa_chain: List[QAChain] = []
_conversational_chain: List[ConversationalQAChain] = []


qa_router = APIRouter(
    prefix="/api/v1",
    tags=["问答"],
)


def get_qa_chain() -> QAChain:
    """获取 QA 链实例"""
    if not _qa_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA 链未初始化，请先调用初始化接口"
        )
    return _qa_chain[0]


def get_conversational_chain() -> ConversationalQAChain:
    """获取对话链实例"""
    if not _conversational_chain:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="对话链未初始化，请先调用初始化接口"
        )
    return _conversational_chain[0]


@qa_router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    qa_chain: QAChain = Depends(get_qa_chain),
):
    """
    执行问答查询

    Args:
        request: 查询请求

    Returns:
        QueryResponse: 查询响应
    """
    try:
        # 执行问答（传递 top_k 和 use_rerank 参数）
        result = qa_chain.invoke(
            request.question,
            top_k=request.top_k,
            use_rerank=request.use_rerank
        )

        # 转换为响应格式
        sources = [
            SourceInfo(
                content=source['content'],
                source=source.get('source', '未知来源'),
                score=source.get('score', 0),
            )
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            query=result.query,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询失败: {str(e)}"
        )


@qa_router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    qa_chain: QAChain = Depends(get_qa_chain),
):
    """
    流式问答查询（SSE）

    Args:
        request: 查询请求

    Returns:
        StreamingResponse: 流式响应
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            for chunk in qa_chain.stream(request.question):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@qa_router.post("/chat")
async def chat(
    request: QueryRequest,
    conv_chain: ConversationalQAChain = Depends(get_conversational_chain),
):
    """
    对话式问答

    Args:
        request: 查询请求

    Returns:
        QueryResponse: 查询响应
    """
    try:
        # 执行问答（传递 top_k 和 use_rerank 参数）
        result = conv_chain.invoke(
            request.question,
            top_k=request.top_k,
            use_rerank=request.use_rerank
        )

        # 转换为响应格式
        sources = [
            SourceInfo(
                content=source['content'],
                source=source.get('source', '未知来源'),
                score=source.get('score', 0),
            )
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            query=result.query,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"对话失败: {str(e)}"
        )


@qa_router.post("/chat/clear")
async def clear_chat_history(
    conv_chain: ConversationalQAChain = Depends(get_conversational_chain),
):
    """
    清空对话历史

    Returns:
        成功消息
    """
    conv_chain.clear_history()
    return {"message": "对话历史已清空"}


def initialize_qa_chain(
    retriever: VectorRetriever,
    llm_config: LLMConfig,
    use_reranker: bool = True,
):
    """
    初始化 QA 链

    Args:
        retriever: 检索器
        llm_config: LLM 配置
        use_reranker: 是否使用重排序
    """
    global _qa_chain, _conversational_chain

    # 创建重排序器
    reranker = None
    if use_reranker:
        try:
            from ...retriever import Reranker
            reranker = Reranker()
        except Exception as e:
            print(f"重排序器初始化失败: {e}")

    # 创建 QA 链
    from ...chains import create_qa_chain

    _qa_chain.clear()
    _qa_chain.append(
        create_qa_chain(
            retriever=retriever,
            llm_config=llm_config,
            reranker=reranker,
            conversational=False,
        )
    )

    _conversational_chain.clear()
    _conversational_chain.append(
        create_qa_chain(
            retriever=retriever,
            llm_config=llm_config,
            reranker=reranker,
            conversational=True,
        )
    )

    print("QA 链初始化完成")
