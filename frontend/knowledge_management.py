import os
import requests
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import time
from typing import Dict, List, Optional
import json

load_dotenv(override=True)

st.set_page_config(
    page_title="知识库管理",
    page_icon="📚",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
EMBEDDING_CONFIG_FILE = "embedding_config.json"

EMBEDDING_MODEL_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ],
        "requires_base_url": False
    },
    "AzureOpenAI": {
        "name": "Azure OpenAI",
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "endpoint_env": "AZURE_OPENAI_ENDPOINT",
        "deployment_env": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "api_version_env": "AZURE_OPENAI_API_VERSION",
        "default_api_version": "2024-02-15-preview",
        "models": [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002"
        ],
        "requires_endpoint": True
    },
    "硅基流动": {
        "name": "硅基流动",
        "api_key_env": "SILICONFLOW_API_KEY",
        "base_url_env": "SILICONFLOW_BASE_URL",
        "default_base_url": "https://api.siliconflow.cn/v1",
        "models": [
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-base-zh-v1.5",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-m3",
            "sentence-transformers/all-MiniLM-L6-v2"
        ],
        "requires_base_url": False
    },
    "阿里百炼": {
        "name": "阿里百炼",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            "text-embedding-v3",
            "text-embedding-v2",
            "text-embedding-async-v2"
        ],
        "requires_base_url": False
    },
    "DeepSeek": {
        "name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-embeddings"
        ],
        "requires_base_url": False
    }
}

def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def format_upload_time(upload_time):
    try:
        dt = datetime.fromisoformat(upload_time)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return upload_time

def validate_embedding_config(provider: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, str]:
    try:
        provider_config = EMBEDDING_MODEL_PROVIDERS[provider]
        
        if not api_key:
            return False, f"{provider_config['name']} API Key 不能为空"
        
        if provider == "AzureOpenAI":
            if not endpoint:
                return False, "Azure OpenAI Endpoint 不能为空"
            if not api_version:
                return False, "Azure OpenAI API Version 不能为空"
        else:
            if provider_config.get("requires_base_url", False) and not base_url:
                return False, f"{provider_config['name']} Base URL 不能为空"
        
        return True, "配置验证通过"
    except Exception as e:
        return False, f"验证失败: {str(e)}"

def test_embedding_connection(provider: str, model: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, str]:
    try:
        if provider == "AzureOpenAI":
            url = f"{endpoint.rstrip('/')}/openai/deployments/{model}/embeddings?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "input": "test"
            }
        else:
            url = f"{base_url.rstrip('/')}/embeddings"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "input": "test"
            }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return True, "连接测试成功"
        elif response.status_code == 401:
            return False, "API Key 无效或已过期"
        elif response.status_code == 404:
            return False, "模型不存在或未部署"
        elif response.status_code == 429:
            return False, "请求频率超限，请稍后重试"
        else:
            error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            return False, f"连接失败: {error_msg}"
    except requests.exceptions.Timeout:
        return False, "连接超时，请检查网络或服务地址"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到服务器，请检查网络或服务地址"
    except Exception as e:
        return False, f"测试连接时出错: {str(e)}"

def fetch_embedding_models(provider: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, List[str], str]:
    try:
        if provider == "AzureOpenAI":
            url = f"{endpoint.rstrip('/')}/openai/deployments?api-version={api_version}"
            headers = {"api-key": api_key}
        else:
            url = f"{base_url.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if provider == "AzureOpenAI":
                models = [deployment.get("model", deployment.get("id", "")) for deployment in data.get("data", [])]
            else:
                models = [model.get("id", "") for model in data.get("data", [])]
            
            models = [m for m in models if m]
            
            if models:
                return True, models, f"成功获取 {len(models)} 个模型"
            else:
                return False, [], "未找到可用模型"
        else:
            return False, [], f"获取模型列表失败: HTTP {response.status_code}"
    except Exception as e:
        return False, [], f"获取模型列表时出错: {str(e)}"

def save_embedding_config(config: Dict):
    try:
        with open(EMBEDDING_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True, "配置已保存"
    except Exception as e:
        return False, f"保存配置失败: {str(e)}"

def load_embedding_config() -> Optional[Dict]:
    try:
        if os.path.exists(EMBEDDING_CONFIG_FILE):
            with open(EMBEDDING_CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception:
        return None

@st.cache_resource(ttl=60)
def get_documents():
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def delete_document(document_id):
    try:
        response = requests.delete(f"{API_URL}/documents/{document_id}", timeout=10)
        if response.status_code == 200:
            get_documents.clear()
            return True
        return False
    except:
        return False

def upload_document(file, chunk_size=500, overlap=50, embedding_config=None):
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"chunk_size": chunk_size, "overlap": overlap}
        if embedding_config:
            data["embedding_config"] = embedding_config
        response = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=60)
        if response.status_code == 200:
            get_documents.clear()
            return response.json()
        return {"success": False, "error": "上传失败"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def render_embedding_config_sidebar():
    st.sidebar.markdown("### 🔤 嵌入模型配置")
    
    if "embedding_config" not in st.session_state:
        saved_config = load_embedding_config()
        if saved_config:
            st.session_state.embedding_config = saved_config
        else:
            st.session_state.embedding_config = {
                "provider": "OpenAI",
                "model": "text-embedding-3-small",
                "api_key": "",
                "base_url": "",
                "endpoint": "",
                "api_version": ""
            }
    
    provider_options = list(EMBEDDING_MODEL_PROVIDERS.keys())
    provider_display = [EMBEDDING_MODEL_PROVIDERS[p]["name"] for p in provider_options]
    
    selected_provider_display = st.sidebar.selectbox(
        "选择嵌入供应商",
        options=provider_display,
        index=provider_options.index(st.session_state.embedding_config["provider"]),
        help="选择要使用的嵌入模型供应商"
    )
    
    selected_provider = provider_options[provider_display.index(selected_provider_display)]
    
    if st.session_state.embedding_config["provider"] != selected_provider:
        st.session_state.embedding_config["provider"] = selected_provider
        provider_config = EMBEDDING_MODEL_PROVIDERS[selected_provider]
        api_key_env_value = os.getenv(provider_config["api_key_env"], "")
        st.session_state.embedding_config["api_key"] = api_key_env_value
        if selected_provider == "AzureOpenAI":
            endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
            api_version_env_value = os.getenv(provider_config["api_version_env"], "")
            deployment_env_value = os.getenv(provider_config["deployment_env"], "")
            st.session_state.embedding_config["endpoint"] = endpoint_env_value
            st.session_state.embedding_config["api_version"] = api_version_env_value or provider_config["default_api_version"]
            st.session_state.embedding_config["deployment"] = deployment_env_value
            st.session_state.embedding_config["base_url"] = ""
        else:
            base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
            st.session_state.embedding_config["base_url"] = base_url_env_value or provider_config.get("default_base_url", "")
            st.session_state.embedding_config["endpoint"] = ""
            st.session_state.embedding_config["api_version"] = ""
            st.session_state.embedding_config["deployment"] = ""
        st.session_state.embedding_config["model"] = provider_config["models"][0]
        st.rerun()
    
    provider_config = EMBEDDING_MODEL_PROVIDERS[selected_provider]
    
    st.sidebar.caption(f"当前供应商: {provider_config['name']}")
    
    api_key_env_value = os.getenv(provider_config["api_key_env"], "")
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.embedding_config.get("api_key") or api_key_env_value,
        type="password",
        help=f"输入 {provider_config['name']} 的 API Key（也可通过环境变量 {provider_config['api_key_env']} 配置）"
    )
    st.session_state.embedding_config["api_key"] = api_key
    
    if selected_provider == "AzureOpenAI":
        endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
        endpoint = st.sidebar.text_input(
            "Endpoint",
            value=st.session_state.embedding_config.get("endpoint") or endpoint_env_value,
            help="Azure OpenAI 服务端点，例如: https://your-resource.openai.azure.com"
        )
        st.session_state.embedding_config["endpoint"] = endpoint
        
        api_version_env_value = os.getenv(provider_config["api_version_env"], "")
        api_version = st.sidebar.text_input(
            "API Version",
            value=st.session_state.embedding_config.get("api_version") or api_version_env_value or provider_config["default_api_version"],
            help="Azure OpenAI API 版本"
        )
        st.session_state.embedding_config["api_version"] = api_version
        
        deployment_env_value = os.getenv(provider_config["deployment_env"], "")
        deployment = st.sidebar.text_input(
            "部署名称 (可选)",
            value=st.session_state.embedding_config.get("deployment") or deployment_env_value,
            help="Azure OpenAI 部署名称（如果与模型名称不同）"
        )
        st.session_state.embedding_config["deployment"] = deployment
    else:
        base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
        base_url = st.sidebar.text_input(
            "Base URL (可选)",
            value=st.session_state.embedding_config.get("base_url") or base_url_env_value or provider_config.get("default_base_url", ""),
            help=f"{provider_config['name']} API 的 Base URL（留空使用默认值）"
        )
        st.session_state.embedding_config["base_url"] = base_url
    
    available_models = provider_config["models"]
    
    # if st.sidebar.button("🔄 刷新模型列表", key="refresh_embedding_models"):
    #     with st.spinner("正在获取嵌入模型列表..."):
    #         if selected_provider == "AzureOpenAI":
    #             success, models, message = fetch_embedding_models(
    #                 selected_provider, api_key,
    #                 endpoint=st.session_state.embedding_config.get("endpoint"),
    #                 api_version=st.session_state.embedding_config.get("api_version")
    #             )
    #         else:
    #             success, models, message = fetch_embedding_models(
    #                 selected_provider, api_key,
    #                 base_url=st.session_state.embedding_config.get("base_url")
    #             )
            
    #         if success:
    #             st.sidebar.success(f"✓ {message}")
    #             if models:
    #                 EMBEDDING_MODEL_PROVIDERS[selected_provider]["models"] = models
    #                 st.session_state.embedding_config["model"] = models[0]
    #                 st.rerun()
    #         else:
    #             st.sidebar.error(f"✗ {message}")
    
    if st.session_state.embedding_config["model"] not in available_models:
        st.session_state.embedding_config["model"] = available_models[0]
    
    selected_model = st.sidebar.selectbox(
        "选择嵌入模型",
        options=available_models,
        index=available_models.index(st.session_state.embedding_config["model"]),
        help=f"选择 {provider_config['name']} 的具体嵌入模型"
    )
    st.session_state.embedding_config["model"] = selected_model
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.sidebar.button("✅ 验证", use_container_width=True, key="validate_embedding"):
            is_valid, message = validate_embedding_config(
                selected_provider,
                api_key,
                base_url=st.session_state.embedding_config.get("base_url"),
                endpoint=st.session_state.embedding_config.get("endpoint"),
                api_version=st.session_state.embedding_config.get("api_version")
            )
            if is_valid:
                st.sidebar.success(f"✓ {message}")
            else:
                st.sidebar.error(f"✗ {message}")
    
    with col2:
        if st.sidebar.button("🔗 测试", use_container_width=True, key="test_embedding"):
            is_valid, _ = validate_embedding_config(
                selected_provider,
                api_key,
                base_url=st.session_state.embedding_config.get("base_url"),
                endpoint=st.session_state.embedding_config.get("endpoint"),
                api_version=st.session_state.embedding_config.get("api_version")
            )
            if not is_valid:
                st.sidebar.error("✗ 请先完成配置验证")
            else:
                with st.spinner("正在测试嵌入连接..."):
                    if selected_provider == "AzureOpenAI":
                        success, message = test_embedding_connection(
                            selected_provider, selected_model, api_key,
                            endpoint=st.session_state.embedding_config.get("endpoint"),
                            api_version=st.session_state.embedding_config.get("api_version")
                        )
                    else:
                        success, message = test_embedding_connection(
                            selected_provider, selected_model, api_key,
                            base_url=st.session_state.embedding_config.get("base_url")
                        )
                    if success:
                        st.sidebar.success(f"✓ {message}")
                    else:
                        st.sidebar.error(f"✗ {message}")
    
    with col3:
        if st.sidebar.button("💾 保存", use_container_width=True, key="save_embedding"):
            success, message = save_embedding_config(st.session_state.embedding_config)
            if success:
                st.sidebar.success(f"✓ {message}")
            else:
                st.sidebar.error(f"✗ {message}")
    
    st.sidebar.divider()
    
    return st.session_state.embedding_config

st.title("📚 知识库管理")

st.sidebar.title("⚙️ 设置")

embedding_config = render_embedding_config_sidebar()

is_embedding_config_valid, _ = validate_embedding_config(
    embedding_config["provider"],
    embedding_config.get("api_key", ""),
    base_url=embedding_config.get("base_url"),
    endpoint=embedding_config.get("endpoint"),
    api_version=embedding_config.get("api_version")
)

if not is_embedding_config_valid:
    st.warning("⚠️ 嵌入模型配置不完整，请在侧边栏完成配置后上传文档")

st.markdown("""
<div style="padding: 10px; background-color: #e8f5e9; border-radius: 5px; margin-bottom: 20px;">
    <strong>📖 知识库说明：</strong>在此页面管理您的文档，上传的文档将被处理并存储到知识库中，供AI对话时检索使用。
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

documents_placeholder = st.empty()
documents = get_documents()

with col1:
    st.subheader("📁 上传文档")
    
    st.markdown("#### ⚙️ 文本分割设置")
    col_chunk, col_overlap = st.columns(2)
    with col_chunk:
        chunk_size = st.slider(
            "文本块大小",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
            help="每个文本块包含的字符数"
        )
    with col_overlap:
        overlap = st.slider(
            "重叠块大小",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            help="相邻文本块之间的重叠字符数"
        )
    st.caption("💡 较小的文本块可以提高检索精度，较大的文本块可以提供更多上下文")
    st.divider()
    
    if "uploaded_files_cache" not in st.session_state:
        st.session_state.uploaded_files_cache = []
    
    uploaded_files = st.file_uploader(
        "选择文档",
        type=["pdf", "txt", "docx", "xlsx"],
        accept_multiple_files=True,
        help="支持 PDF、TXT、DOCX、XLSX 格式的文档",
        key="file_uploader"
    )
    
    if st.button("🔄 清除上传缓存", key="clear_cache"):
        st.session_state.uploaded_files_cache = []
        st.success("缓存已清除，可以重新上传文件")
        st.rerun()
    
    if uploaded_files:
        new_files = []
        for uploaded_file in uploaded_files:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_key not in st.session_state.uploaded_files_cache:
                new_files.append(uploaded_file)
        
        if new_files:
            for uploaded_file in new_files:
                with st.spinner(f"正在处理 {uploaded_file.name}..."):
                    result = upload_document(uploaded_file, chunk_size, overlap, embedding_config)
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    st.session_state.uploaded_files_cache.append(file_key)
                    if result.get("success"):
                        st.success(f"✅ {uploaded_file.name} 上传成功！生成 {result.get('chunks_count', 0)} 个知识块")
                    else:
                        error_msg = result.get('error', '未知错误')
                        if "提取的文本内容为空" in error_msg:
                            error_msg += "\n💡 提示：该文件可能是图片型PDF，请使用可提取文本的PDF或转换为TXT格式"
                        st.error(f"❌ {uploaded_file.name} 上传失败: {error_msg}")
            st.rerun()
    
    st.divider()
    
    st.subheader("📊 统计信息")
    documents = get_documents()
    
    total_docs = len(documents)
    completed_docs = sum(1 for doc in documents if doc.get('status') == 'completed')
    total_chunks = sum(doc.get('chunks_count', 0) for doc in documents if doc.get('status') == 'completed')
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("总文档数", total_docs)
    with col_b:
        st.metric("已处理文档", completed_docs)
    with col_c:
        st.metric("知识块总数", total_chunks)

with documents_placeholder.container():
    with col2:
        st.subheader(f"📄 已上传文档 ({len(documents)})")
        
        if documents:
            for doc in documents:
                status_emoji = "✅" if doc['status'] == 'completed' else "⏳" if doc['status'] == 'processing' else "❌"
                status_text = "已处理" if doc['status'] == 'completed' else "处理中" if doc['status'] == 'processing' else "失败"
                
                with st.expander(f"{status_emoji} {doc['filename']}", expanded=False):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.write(f"**上传时间:** {format_upload_time(doc['upload_time'])}")
                        st.write(f"**文件大小:** {format_file_size(doc['file_size'])}")
                        st.write(f"**文件类型:** {doc['file_type']}")
                        if doc.get('status') == 'completed':
                            st.write(f"**知识块数量:** {doc.get('chunks_count', 0)}")
                        st.write(f"**状态:** {status_text}")
                    
                    with col_b:
                        if st.button("🗑️ 删除", key=f"delete_{doc['id']}"):
                            st.session_state[f"confirm_delete_{doc['id']}"] = True
                    
                    if st.session_state.get(f"confirm_delete_{doc['id']}", False):
                        st.warning(f"⚠️ 确认要删除文档 '{doc['filename']}' 吗？此操作不可恢复。")
                        col_c, col_d = st.columns([1, 1])
                        with col_c:
                            if st.button("✅ 确认删除", key=f"confirm_{doc['id']}", type="primary"):
                                with st.spinner("正在删除..."):
                                    time.sleep(0.3)
                                    if delete_document(doc['id']):
                                        st.success("文档删除成功")
                                        del st.session_state[f"confirm_delete_{doc['id']}"]
                                        st.rerun()
                                    else:
                                        st.error("文档删除失败")
                        with col_d:
                            if st.button("❌ 取消", key=f"cancel_{doc['id']}"):
                                del st.session_state[f"confirm_delete_{doc['id']}"]
                                st.rerun()
        else:
            st.info("📭 暂无文档，请上传文档开始使用知识库")

st.sidebar.divider()
st.sidebar.markdown("### 📖 使用说明")
st.sidebar.markdown("""
1. **嵌入模型配置**: 
   - 选择嵌入模型供应商（OpenAI、AzureOpenAI、硅基流动、阿里百炼、DeepSeek）
   - 配置API Key和相关参数
   - 验证配置并测试连接
   - 保存配置以供后续使用
2. **上传文档**: 支持上传 PDF、TXT、DOCX 格式的文档
3. **文本分割设置**: 
   - 文本块大小：每个知识块包含的字符数（100-2000）
   - 重叠块大小：相邻块之间的重叠字符数（0-500）
   - 较小的文本块可提高检索精度，较大的文本块提供更多上下文
4. **文档处理**: 系统会使用配置的嵌入模型将文档内容向量化并存储到知识库
5. **知识块生成**: 文档会被分割成多个知识块，便于检索
6. **删除文档**: 可以删除不需要的文档，同时清除相关知识块
7. **对话使用**: 在AI对话页面提问时，系统会检索此知识库中的内容
""")
