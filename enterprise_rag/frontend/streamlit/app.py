"""
Streamlit 前端应用
"""
import streamlit as st
import requests
from typing import List, Dict

# 配置页面
st.set_page_config(
    page_title="企业 RAG 知识库",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 配置
API_BASE_URL = st.session_state.get("api_base_url", "http://localhost:8000")


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = "http://localhost:8000"

    if "conversational" not in st.session_state:
        st.session_state.conversational = False

    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    if "use_rerank" not in st.session_state:
        st.session_state.use_rerank = True


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.title("⚙️ 配置")

        # API 地址
        api_url = st.text_input(
            "API 地址",
            value=API_BASE_URL,
            key="api_base_url_input"
        )
        st.session_state.api_base_url = api_url

        st.divider()

        # 设置
        st.subheader("检索设置")
        st.session_state.top_k = st.slider(
            "返回结果数",
            min_value=1,
            max_value=20,
            value=st.session_state.get("top_k", 5)
        )
        st.session_state.use_rerank = st.checkbox(
            "使用重排序",
            value=st.session_state.get("use_rerank", True)
        )
        st.session_state.conversational = st.checkbox(
            "对话模式",
            value=st.session_state.get("conversational", False)
        )

        st.divider()

        # 操作
        st.subheader("操作")
        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.rerun()

        if st.button("清空聊天记录"):
            try:
                response = requests.post(f"{API_BASE_URL}/api/v1/chat/clear")
                if response.status_code == 200:
                    st.success("对话历史已清空")
                else:
                    st.error("清空失败")
            except Exception as e:
                st.error(f"清空失败: {e}")

        st.divider()

        # 关于
        st.subheader("关于")
        st.info(
            """
            **企业 RAG 知识库**

            基于 Milvus + LangChain 的
            企业级知识库问答系统

            版本: 1.0.0
            """
        )


def chat_interface():
    """聊天界面"""
    st.title("🤖 企业知识库问答")

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # 显示来源
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 查看来源"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**来源 {i}** (相似度: {source['score']:.2f})")
                        st.text(source['content'])
                        st.caption(f"文件: {source['source']}")
                        st.divider()


def main():
    """主函数"""
    init_session_state()
    sidebar()
    chat_interface()

    # 用户输入
    if prompt := st.chat_input("输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 API
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # 判断使用哪个接口
                    if st.session_state.get("conversational", False):
                        endpoint = "/api/v1/chat"
                    else:
                        endpoint = "/api/v1/query"

                    # 发送请求
                    response = requests.post(
                        f"{st.session_state.api_base_url}{endpoint}",
                        json={
                            "question": prompt,
                            "top_k": st.session_state.get("top_k", 5),
                            "use_rerank": st.session_state.get("use_rerank", True),
                        },
                        timeout=30,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        # 显示回答
                        st.markdown(answer)

                        # 保存到历史
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    else:
                        error_msg = response.json().get("detail", "未知错误")
                        st.error(f"查询失败: {error_msg}")

                except requests.exceptions.Timeout:
                    st.error("请求超时，请稍后重试")
                except Exception as e:
                    st.error(f"发生错误: {e}")


if __name__ == "__main__":
    main()
