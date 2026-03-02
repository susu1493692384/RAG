"""
复杂流程图
使用 LangGraph 构建复杂的 RAG 流程
"""
from typing import TypedDict, Annotated, Sequence, List
from operator import itemgetter

try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from ..retriever import VectorRetriever, HybridRetriever
from ..config import LLMConfig


class RAGState(TypedDict):
    """RAG 流程状态"""
    question: str
    retrieved_documents: List[dict]
    generation: str
    steps: List[str]


class RAGGraph:
    """RAG 流程图类"""

    def __init__(
        self,
        retriever: VectorRetriever,
        llm_config: LLMConfig,
        use_query_rewriting: bool = True,
        use_rerank: bool = True,
        max_retrieval_rounds: int = 2,
    ):
        """
        初始化 RAG 流程图

        Args:
            retriever: 检索器
            llm_config: LLM 配置
            use_query_rewriting: 是否使用查询改写
            use_rerank: 是否使用重排序
            max_retrieval_rounds: 最大检索轮数
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("需要安装 LangGraph: pip install langgraph")

        self.retriever = retriever
        self.llm_config = llm_config
        self.use_query_rewriting = use_query_rewriting
        self.use_rerank = use_rerank
        self.max_retrieval_rounds = max_retrieval_rounds

        # 初始化 LLM
        try:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(**llm_config.get_llm_kwargs())
        except ImportError:
            raise ImportError("需要安装 langchain-openai")

        # 构建图
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建流程图"""
        workflow = StateGraph(RAGState)

        # 添加节点
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)

        # 设置入口
        workflow.set_entry_point("retrieve")

        # 添加边
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # 条件边：是否需要查询改写
        if self.use_query_rewriting:
            workflow.add_conditional_edges(
                "retrieve",
                self._should_rewrite_query,
                {
                    "rewrite": "rewrite_query",
                    "continue": "generate",
                }
            )
            workflow.add_edge("rewrite_query", "retrieve")

        return workflow.compile()

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """检索节点"""
        question = state["question"]

        # 执行检索
        results = self.retriever.search(question, top_k=5)

        state["retrieved_documents"] = [
            {
                "content": r.content if hasattr(r, 'content') else r.get('text', ''),
                "score": r.score if hasattr(r, 'score') else r.get('score', 0),
                "metadata": r.metadata if hasattr(r, 'metadata') else r,
            }
            for r in results
        ]
        state["steps"] = state.get("steps", []) + ["retrieve"]

        return state

    def _generate_node(self, state: RAGState) -> RAGState:
        """生成节点"""
        question = state["question"]
        documents = state["retrieved_documents"]

        # 格式化上下文
        context = "\n\n---\n\n".join([
            f"[来源 {i+1}] {doc['content']}"
            for i, doc in enumerate(documents)
        ])

        # 构建提示
        prompt = ChatPromptTemplate.from_template(
            "你是一个专业的企业知识库助手。请根据以下上下文信息回答用户问题。\n\n"
            "上下文信息：\n{context}\n\n"
            "用户问题：\n{question}\n\n"
            "回答要求：\n"
            "1. 基于上下文信息回答，不要编造答案\n"
            "2. 如果上下文中没有相关信息，请明确告知用户\n"
            "3. 回答要准确、清晰、有条理\n\n"
            "回答："
        )

        # 生成回答
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        state["generation"] = response.content
        state["steps"] = state.get("steps", []) + ["generate"]

        return state

    def _rewrite_query_node(self, state: RAGState) -> RAGState:
        """查询改写节点"""
        question = state["question"]

        # 使用 LLM 改写查询
        prompt = ChatPromptTemplate.from_template(
            "将用户查询改写为更适合检索的形式，保持原意不变。\n\n"
            "原查询：{question}\n\n"
            "改写后的查询："
        )

        chain = prompt | self.llm
        response = chain.invoke({"question": question})

        state["question"] = response.content
        state["steps"] = state.get("steps", []) + ["rewrite_query"]

        return state

    def _should_rewrite_query(self, state: RAGState) -> str:
        """判断是否需要改写查询"""
        # 检查检索结果质量
        documents = state.get("retrieved_documents", [])

        if not documents:
            return "rewrite"

        # 如果最高分太低，改写查询重新检索
        max_score = max([doc.get("score", 0) for doc in documents])
        if max_score < 0.5:
            # 检查是否超过最大轮数
            rewrite_count = state.get("steps", []).count("rewrite_query")
            if rewrite_count < self.max_retrieval_rounds:
                return "rewrite"

        return "continue"

    def invoke(self, question: str) -> dict:
        """
        执行流程

        Args:
            question: 用户问题

        Returns:
            结果字典
        """
        initial_state = {
            "question": question,
            "retrieved_documents": [],
            "generation": "",
            "steps": [],
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "question": question,
            "answer": final_state["generation"],
            "sources": final_state["retrieved_documents"],
            "steps": final_state["steps"],
        }

    def stream(self, question: str):
        """流式执行流程"""
        initial_state = {
            "question": question,
            "retrieved_documents": [],
            "generation": "",
            "steps": [],
        }

        for event in self.graph.stream(initial_state):
            yield event


def create_rag_graph(
    retriever: VectorRetriever,
    llm_config: LLMConfig,
    **kwargs
) -> RAGGraph:
    """
    创建 RAG 流程图

    Args:
        retriever: 检索器
        llm_config: LLM 配置

    Returns:
        RAGGraph 实例
    """
    return RAGGraph(
        retriever=retriever,
        llm_config=llm_config,
        **kwargs
    )
