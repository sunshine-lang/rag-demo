import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="AI智能助手",
    page_icon="🤖",
    layout="wide"
)

page = st.sidebar.radio("选择页面", ["💬 AI对话", "🤖 模型配置", "📚 知识库管理"])

if page == "💬 AI对话":
    exec(open(os.path.join(os.path.dirname(__file__), "chat.py"), encoding='utf-8').read())
elif page == "🤖 模型配置":
    exec(open(os.path.join(os.path.dirname(__file__), "model_config_page.py"), encoding='utf-8').read())
elif page == "📚 知识库管理":
    exec(open(os.path.join(os.path.dirname(__file__), "knowledge_management.py"), encoding='utf-8').read())
