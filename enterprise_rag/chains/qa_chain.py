"""
问答链
实现 RAG 问答的核心逻辑
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..retriever import VectorRetriever, HybridRetriever, Reranker
from ..config import LLMConfig


@dataclass
class QAResult:
    """问答结果"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context: List[str]


class QAChain:
    """问答链类"""

    # 默认提示模板
    DEFAULT_TEMPLATE = """你是一个专业的企业知识库助手。请根据以下上下文信息回答用户问题。

上下文信息：
{context}

用户问题：
{question}

回答要求：
1. 基于上下文信息回答，不要编造答案
2. 如果上下文中没有相关信息，请明确告知用户
3. 回答要准确、清晰、有条理
4. 必要时可以引用上下文中的具体内容

回答："""

    CONVERSATIONAL_TEMPLATE = """你是一个专业的企业知识库助手。请根据上下文信息和对话历史回答用户问题。

上下文信息：
{context}

对话历史：
{chat_history}

用户问题：
{question}

回答要求：
1. 优先使用最新的上下文信息
2. 结合对话历史理解问题
3. 如果信息不足，可以询问用户更多细节
4. 回答要自然流畅，符合对话习惯

回答："""

    def __init__(
        self,
        retriever,
        llm_config: LLMConfig,
        reranker: Optional[Reranker] = None,
        top_k: int = 5,
        conversational: bool = False,
        custom_template: Optional[str] = None,
    ):
        """
        初始化问答链

        Args:
            retriever: 检索器（VectorRetriever 或 HybridRetriever）
            llm_config: LLM 配置
            reranker: 重排序器
            top_k: 检索结果数量
            conversational: 是否为对话模式
            custom_template: 自定义提示模板
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 LangChain: pip install langchain langchain-openai")

        self.retriever = retriever
        self.llm_config = llm_config
        self.reranker = reranker
        self.top_k = top_k
        self.conversational = conversational

        # 初始化 LLM
        self.llm = ChatOpenAI(**llm_config.get_llm_kwargs())

        # 构建提示模板
        template = custom_template or (
            self.CONVERSATIONAL_TEMPLATE if conversational
            else self.DEFAULT_TEMPLATE
        )
        self.prompt = ChatPromptTemplate.from_template(template)

        # 构建链
        self.chain = self._build_chain()

    def _build_chain(self):
        """构建问答链"""
        def retrieve_context(query: str) -> List[Dict[str, Any]]:
            """检索上下文"""
            results = self.retriever.search(query, top_k=self.top_k)

            # 重排序
            if self.reranker:
                results = self.reranker.rerank_search_results(query, results)

            return results

        def format_context(results: List[Dict[str, Any]]) -> str:
            """格式化上下文"""
            if not results:
                return "未找到相关上下文信息。"

            contexts = []
            for i, result in enumerate(results, 1):
                content = result.content if hasattr(result, 'content') else result.get('text', '')
                source = result.metadata.get('source', '未知来源') if hasattr(result, 'metadata') else '未知来源'
                contexts.append(f"[来源 {i}] {source}\n{content}")

            return "\n\n---\n\n".join(contexts)

        # 构建完整链
        chain = (
            {
                "context": RunnableLambda(retrieve_context) | RunnableLambda(format_context),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def invoke(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None
    ) -> QAResult:
        """
        执行问答

        Args:
            query: 用户问题
            top_k: 返回结果数量（可选，默认使用初始化时的值）
            use_rerank: 是否使用重排序（可选，默认使用初始化时的值）

        Returns:
            QAResult 对象
        """
        # 使用传入的参数或默认值
        top_k = top_k if top_k is not None else self.top_k
        use_rerank = use_rerank if use_rerank is not None else (self.reranker is not None)

        # 检索上下文
        search_results = self.retriever.search(query, top_k=top_k)

        # 重排序
        if use_rerank and self.reranker:
            search_results = self.reranker.rerank_search_results(
                query,
                search_results
            )

        # 格式化上下文
        contexts = [
            result.content if hasattr(result, 'content') else result.get('text', '')
            for result in search_results
        ]

        # 生成回答
        answer = self.chain.invoke(query)

        # 提取来源
        sources = []
        for result in search_results:
            if hasattr(result, 'metadata'):
                metadata = result.metadata
            else:
                metadata = result

            sources.append({
                'content': metadata.get('text', ''),
                'source': metadata.get('source', '未知来源'),
                'score': getattr(result, 'score', 0),
            })

        return QAResult(
            answer=answer,
            sources=sources,
            query=query,
            context=contexts,
        )

    def stream(self, query: str):
        """
        流式输出

        Args:
            query: 用户问题

        Yields:
            回答文本片段
        """
        # 检索上下文
        search_results = self.retriever.search(query, top_k=self.top_k)

        if self.reranker:
            search_results = self.reranker.rerank_search_results(
                query,
                search_results
            )

        # 格式化上下文
        contexts = [
            result.content if hasattr(result, 'content') else result.get('text', '')
            for result in search_results
        ]
        context_str = "\n\n---\n\n".join(contexts)

        # 构建输入
        chain_input = {
            "context": context_str,
            "question": query,
        }

        # 流式输出
        for chunk in self.chain.stream(chain_input):
            yield chunk


class ConversationalQAChain(QAChain):
    """对话式问答链（支持多轮对话）"""

    # 查询重写模板
    QUERY_REWRITE_TEMPLATE = """你是一个查询优化助手。根据对话历史，将用户的当前问题重写为一个独立、完整的问题。

对话历史：
{chat_history}

当前问题：
{current_query}

请重写问题，使其：
1. 包含所有必要的上下文信息
2. 是一个独立、完整的问题
3. 适合用于检索相关信息

只输出重写后的问题，不要有其他内容。"""

    def __init__(self, *args, memory_limit=5, **kwargs):
        """
        初始化对话式问答链

        Args:
            memory_limit: 历史记录保留轮数
        """
        super().__init__(*args, conversational=True, **kwargs)
        self.memory_limit = memory_limit
        self.chat_history = []

    def _rewrite_query_with_history(self, query: str) -> str:
        """
        使用对话历史重写查询，以便更好地检索

        Args:
            query: 当前用户问题

        Returns:
            重写后的查询
        """
        if not self.chat_history:
            # 没有历史，直接返回原问题
            return query

        # 格式化历史
        history_str = self._format_history()

        # 构建提示
        prompt = self.QUERY_REWRITE_TEMPLATE.format(
            chat_history=history_str,
            current_query=query
        )

        try:
            # 使用 LLM 重写查询
            rewritten = self.llm.invoke(prompt).content.strip()
            # 移除可能的引号
            rewritten = rewritten.strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as e:
            # 重写失败，使用原问题
            print(f"查询重写失败: {e}")
            return query

    def invoke(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None
    ) -> QAResult:
        """
        执行问答（带对话历史）

        Args:
            query: 用户问题
            top_k: 返回结果数量（可选，默认使用初始化时的值）
            use_rerank: 是否使用重排序（可选，默认使用初始化时的值）
        """
        # 使用传入的参数或默认值
        top_k = top_k if top_k is not None else self.top_k
        use_rerank = use_rerank if use_rerank is not None else (self.reranker is not None)

        # 添加到历史
        self.chat_history.append(("user", query))

        # 重写查询以支持上下文检索
        search_query = self._rewrite_query_with_history(query)

        # 使用重写后的查询检索上下文
        search_results = self.retriever.search(search_query, top_k=top_k)

        # 重排序
        if use_rerank and self.reranker:
            search_results = self.reranker.rerank_search_results(
                search_query,
                search_results
            )

        # 格式化上下文
        contexts = [
            result.content if hasattr(result, 'content') else result.get('text', '')
            for result in search_results
        ]
        context_str = "\n\n---\n\n".join(contexts)

        # 格式化历史
        history_str = self._format_history()

        # 构建输入
        chain_input = {
            "context": context_str,
            "question": query,
            "chat_history": history_str,
        }

        # 生成回答
        answer = self.llm.invoke(
            self.CONVERSATIONAL_TEMPLATE.format(**chain_input)
        ).content

        # 添加到历史
        self.chat_history.append(("assistant", answer))

        # 限制历史长度
        if len(self.chat_history) > self.memory_limit * 2:
            self.chat_history = self.chat_history[-self.memory_limit * 2:]

        return QAResult(
            answer=answer,
            sources=[{
                'content': r.content if hasattr(r, 'content') else r.get('text', ''),
                'source': r.metadata.get('source', '') if hasattr(r, 'metadata') else '',
            } for r in search_results],
            query=query,
            context=contexts,
        )

    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.chat_history:
            return "无历史对话"

        formatted = []
        for role, message in self.chat_history:
            if role == "user":
                formatted.append(f"用户: {message}")
            else:
                formatted.append(f"助手: {message}")

        return "\n".join(formatted)

    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()


def create_qa_chain(
    retriever,
    llm_config: LLMConfig,
    reranker: Optional[Reranker] = None,
    conversational: bool = False,
) -> QAChain:
    """
    创建问答链

    Args:
        retriever: 检索器
        llm_config: LLM 配置
        reranker: 重排序器
        conversational: 是否为对话模式

    Returns:
        QAChain 实例
    """
    if conversational:
        return ConversationalQAChain(
            retriever=retriever,
            llm_config=llm_config,
            reranker=reranker,
        )
    else:
        return QAChain(
            retriever=retriever,
            llm_config=llm_config,
            reranker=reranker,
        )
