"""中文页面：配置和展示保存在 Streamlit，业务逻辑见 blog_rag.py。"""

import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from blog_rag import AppConfig, add_documents_to_qdrant, ask_question, initialize_components

BASE_DIR = Path(__file__).resolve().parent


def show_error(message, error, config=None):
    """保留诊断信息，同时隐藏配置中的密钥。"""
    detail = str(error)
    if config is not None:
        for key in (config.qwen_api_key, config.deepseek_api_key, config.qdrant_api_key):
            if key:
                detail = detail.replace(key, "<密钥已隐藏>")
    st.error(message)
    with st.expander("查看错误详情"):
        st.code(detail, language="text")


def set_sidebar():
    """先填写配置，点击连接时才访问外部服务。每个会话单独持有连接。"""
    with st.sidebar:
        st.subheader("模型与数据库配置")
        st.caption("自动读取项目 .env，也可以在此修改。密钥仅保留在当前会话，不写回文件。")
        with st.form("connection_config"):
            st.markdown("**千问：文本嵌入**")
            qwen_model = st.text_input("千问嵌入模型 ID", value=os.getenv("QWEN_MODEL", ""))
            qwen_api_key = st.text_input("千问 API 密钥", value=os.getenv("QWEN_API_KEY", ""), type="password")
            qwen_base_url = st.text_input("千问 API 地址", value=os.getenv("QWEN_BASE_URL", ""))
            st.markdown("**DeepSeek：问答、评分与改写**")
            deepseek_model = st.text_input("DeepSeek 模型 ID", value=os.getenv("DEEPSEEK_MODEL", ""))
            deepseek_api_key = st.text_input("DeepSeek API 密钥", value=os.getenv("DEEPSEEK_API_KEY", ""),
                                             type="password")
            deepseek_base_url = st.text_input("DeepSeek API 地址", value=os.getenv("DEEPSEEK_BASE_URL", ""))
            st.markdown("**Qdrant：向量数据库**")
            qdrant_host = st.text_input("Qdrant 服务地址", value=os.getenv("QDRANT_HOST", ""))
            qdrant_api_key = st.text_input("Qdrant API 密钥（本地无认证服务可留空）",
                                           value=os.getenv("QDRANT_API_KEY", ""), type="password")
            submitted = st.form_submit_button("保存并连接")
        if submitted:
            config = AppConfig(
                qwen_model=qwen_model.strip(), qwen_api_key=qwen_api_key.strip(),
                qwen_base_url=qwen_base_url.strip(), deepseek_model=deepseek_model.strip(),
                deepseek_api_key=deepseek_api_key.strip(), deepseek_base_url=deepseek_base_url.strip(),
                qdrant_host=qdrant_host.strip(), qdrant_api_key=qdrant_api_key.strip(),
            )
            try:
                with st.spinner("正在检查嵌入模型并连接数据库……"):
                    resources = initialize_components(config)
            except Exception as error:
                show_error("连接失败，配置未更新。", error, config)
            else:
                previous = st.session_state.get("resources")
                st.session_state.resources = resources
                st.session_state.config = config
                st.session_state.pop("last_result", None)
                if previous is not None:
                    previous.client.close()
                st.success("嵌入服务与数据库已连接；DeepSeek 将在提问时调用。")


def main():
    load_dotenv(BASE_DIR / ".env")
    st.set_page_config(page_title="AI 博客问答", page_icon=":mag_right:")
    st.header(":blue[AI 博客问答]")
    st.caption("千问生成文本向量，DeepSeek 检索资料并用中文回答。")
    set_sidebar()
    resources = st.session_state.get("resources")
    if resources is None:
        st.info("请先在左侧确认配置，再点击“保存并连接”。")
        return
    config = st.session_state.config
    st.caption(f"当前知识库：{resources.db.collection_name}")
    st.info("首次使用本版本，或更换嵌入模型后，请重新导入博客。旧知识库中的数据会保留。")

    with st.form("import_blog"):
        url = st.text_input(":link: 博客文章链接", placeholder="请输入公开可访问的文章 URL")
        import_clicked = st.form_submit_button("导入博客")
    # 每次重跑都保留同一个提示位置，避免提示出现或消失时移动下方表单。
    import_feedback = st.empty()

    with st.form("ask_blog"):
        query = st.text_area(":bulb: 你想了解什么？", placeholder="例如：这篇文章如何区分短期记忆和长期记忆？")
        ask_clicked = st.form_submit_button("提交问题")
    question_feedback = st.empty()
    answer_area = st.empty()
    st.markdown("---")

    # 先完成固定布局，再执行耗时任务；加载期间也只保留一份提问表单。
    if import_clicked:
        with import_feedback.container():
            if not url.strip():
                st.warning("请输入博客文章链接。")
            else:
                try:
                    with st.spinner("正在读取正文、切片并写入数据库……"):
                        count = add_documents_to_qdrant(url, resources.db)
                    st.success(f"博客已导入，共 {count} 个片段，可以开始提问。")
                except Exception as error:
                    show_error("导入博客失败。", error, config)

    if ask_clicked:
        st.session_state.pop("last_result", None)
        with question_feedback.container():
            if not query.strip():
                st.warning("请输入你想询问的问题。")
            else:
                try:
                    with st.spinner("正在检索资料并生成回答（最多改写两次）……"):
                        st.session_state.last_result = ask_question(resources.graph, query)
                except Exception as error:
                    show_error("生成回答失败。", error, config)
    result = st.session_state.get("last_result")
    if result is not None:
        with answer_area.container():
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("查看检索资料"):
                    for index, doc in enumerate(result["documents"], start=1):
                        st.write(f"[{index}] {doc.metadata.get('source', '未知来源')}")
                        st.text(doc.page_content)
            else:
                st.caption("此回答未附带博客证据。")


if __name__ == "__main__":
    main()
