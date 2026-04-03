import json
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, List, Optional
import sseclient

load_dotenv(override=True)

# Note: set_page_config is already called in app.py

API_URL = os.getenv("API_URL", "http://localhost:8001")
CONFIG_FILE = "model_config.json"

MODEL_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-5.1",
            "gpt-5.2",
        ],
        "requires_base_url": False
    },
    "AzureOpenAI": {
        "name": "Azure OpenAI",
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "endpoint_env": "AZURE_OPENAI_ENDPOINT",
        "deployment_env": "AZURE_OPENAI_DEPLOYMENT",
        "api_version_env": "AZURE_OPENAI_API_VERSION",
        "default_api_version": "2024-02-15-preview",
        "models": [
            "gpt-5.1",
            "gpt-5.2",
        ],
        "requires_endpoint": True
    },
    "硅基流动": {
        "name": "硅基流动",
        "api_key_env": "SILICONFLOW_API_KEY",
        "base_url_env": "SILICONFLOW_BASE_URL",
        "default_base_url": "https://api.siliconflow.cn/v1",
        "models": [
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Qwen/Qwen2.5-72B-Instruct",
            "THUDM/glm-4-9b-chat",
            "meta-llama/Llama-3.3-70B-Instruct",
            "01-ai/Yi-1.5-34B-Chat"
        ],
        "requires_base_url": False
    },
    "阿里百炼": {
        "name": "阿里百炼",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": [
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen-long",
            "qwen-vl-max",
            "qwen-vl-plus"
        ],
        "requires_base_url": False
    },
    "DeepSeek": {
        "name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner"
        ],
        "requires_base_url": False
    }
}

def validate_model_config(provider: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, str]:
    try:
        provider_config = MODEL_PROVIDERS[provider]
        
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

def test_model_connection(provider: str, model: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, str]:
    try:
        if provider == "AzureOpenAI":
            url = f"{endpoint.rstrip('/')}/openai/deployments/{model}/chat/completions?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {
                "messages": [{"role": "user", "content": "test"}]
            }
        else:
            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}]
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

def fetch_available_models(provider: str, api_key: str, base_url: Optional[str] = None, endpoint: Optional[str] = None, api_version: Optional[str] = None) -> tuple[bool, List[str], str]:
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

def send_chat_message(messages, top_k=5, use_hybrid=True, use_query_expansion=True, model_config=None, embedding_config=None, web_search_enabled=True):
    try:
        payload = {
            "messages": messages,
            "system_prompt": "你是一个专业的智能助手。请优先基于知识库内容回答用户问题。如果知识库中没有相关信息，请基于联网搜索结果回答。回答时请清晰标注信息来源。",
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_query_expansion": use_query_expansion,
            "web_search_enabled": web_search_enabled
        }

        if model_config:
            payload["llm_config"] = model_config
        if embedding_config:
            payload["embedding_config"] = embedding_config
        
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def send_chat_message_stream(messages, top_k=5, use_hybrid=True, use_query_expansion=True, model_config=None, embedding_config=None, web_search_enabled=True):
    try:
        payload = {
            "messages": messages,
            "system_prompt": "你是一个专业的智能助手。请优先基于知识库内容回答用户问题。如果知识库中没有相关信息，请基于联网搜索结果回答。回答时请清晰标注信息来源。",
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_query_expansion": use_query_expansion,
            "web_search_enabled": web_search_enabled
        }

        if model_config:
            payload["llm_config"] = model_config
        if embedding_config:
            payload["embedding_config"] = embedding_config
        
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=300
        )
        
        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            
            metadata = None
            error = None
            
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    
                    if data.get("type") == "metadata":
                        metadata = data.get("data")
                    elif data.get("type") == "content":
                        yield data.get("content", ""), metadata
                    elif data.get("type") == "error":
                        error = data.get("error")
                        break
                    elif data.get("type") == "done":
                        break
                        
                except json.JSONDecodeError:
                    continue
            
            if error:
                yield None, {"error": error}
        else:
            yield None, {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        yield None, {"error": str(e)}

def save_model_config(config: Dict):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True, "配置已保存"
    except Exception as e:
        return False, f"保存配置失败: {str(e)}"

def load_model_config() -> Optional[Dict]:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    except Exception:
        return None

def build_default_model_config(provider: str = "OpenAI") -> Dict:
    provider_config = MODEL_PROVIDERS[provider]
    config = {
        "provider": provider,
        "model": provider_config["models"][0],
        "api_key": os.getenv(provider_config["api_key_env"], ""),
        "base_url": "",
        "endpoint": "",
        "api_version": "",
        "deployment": "",
        "temperature": 0.7
    }

    if provider == "AzureOpenAI":
        config["endpoint"] = os.getenv(provider_config["endpoint_env"], "")
        config["api_version"] = os.getenv(provider_config["api_version_env"], "") or provider_config["default_api_version"]
        config["deployment"] = os.getenv(provider_config["deployment_env"], "")
    else:
        config["base_url"] = os.getenv(provider_config.get("base_url_env", ""), "") or provider_config.get("default_base_url", "")

    return config

def get_active_model_config() -> Dict:
    saved_config = load_model_config()
    if saved_config and saved_config.get("provider") in MODEL_PROVIDERS:
        base = build_default_model_config(saved_config["provider"])
        base.update(saved_config)
        return base
    return build_default_model_config()

def render_model_config_sidebar():
    st.sidebar.markdown("### 🤖 模型配置")
    
    if "model_config" not in st.session_state:
        saved_config = load_model_config()
        if saved_config:
            st.session_state.model_config = saved_config
        else:
            st.session_state.model_config = {
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "api_key": "",
                "base_url": "",
                "endpoint": "",
                "api_version": "",
                "temperature": 0.7
            }
    
    provider_options = list(MODEL_PROVIDERS.keys())
    provider_display = [MODEL_PROVIDERS[p]["name"] for p in provider_options]
    
    selected_provider_display = st.sidebar.selectbox(
        "选择供应商",
        options=provider_display,
        index=provider_options.index(st.session_state.model_config["provider"]),
        help="选择要使用的AI模型供应商",
        key="provider_selectbox"
    )
    
    selected_provider = provider_options[provider_display.index(selected_provider_display)]
    
    if st.session_state.model_config["provider"] != selected_provider:
        st.session_state.model_config["provider"] = selected_provider
        provider_config = MODEL_PROVIDERS[selected_provider]
        api_key_env_value = os.getenv(provider_config["api_key_env"], "")
        st.session_state.model_config["api_key"] = api_key_env_value
        if selected_provider == "AzureOpenAI":
            endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
            api_version_env_value = os.getenv(provider_config["api_version_env"], "")
            deployment_env_value = os.getenv(provider_config["deployment_env"], "")
            st.session_state.model_config["endpoint"] = endpoint_env_value
            st.session_state.model_config["api_version"] = api_version_env_value or provider_config["default_api_version"]
            st.session_state.model_config["deployment"] = deployment_env_value
            st.session_state.model_config["base_url"] = ""
        else:
            base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
            st.session_state.model_config["base_url"] = base_url_env_value or provider_config.get("default_base_url", "")
            st.session_state.model_config["endpoint"] = ""
            st.session_state.model_config["api_version"] = ""
            st.session_state.model_config["deployment"] = ""
        st.session_state.model_config["model"] = provider_config["models"][0]
        st.rerun()
    
    provider_config = MODEL_PROVIDERS[selected_provider]
    
    st.sidebar.caption(f"当前供应商: {provider_config['name']}")
    
    api_key_env_value = os.getenv(provider_config["api_key_env"], "")
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.model_config.get("api_key") or api_key_env_value,
        type="password",
        help=f"输入 {provider_config['name']} 的 API Key（也可通过环境变量 {provider_config['api_key_env']} 配置）"
    )
    st.session_state.model_config["api_key"] = api_key
    
    if selected_provider == "AzureOpenAI":
        endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
        endpoint = st.sidebar.text_input(
            "Endpoint",
            value=st.session_state.model_config.get("endpoint") or endpoint_env_value,
            help="Azure OpenAI 服务端点，例如: https://your-resource.openai.azure.com"
        )
        st.session_state.model_config["endpoint"] = endpoint
        
        api_version_env_value = os.getenv(provider_config["api_version_env"], "")
        api_version = st.sidebar.text_input(
            "API Version",
            value=st.session_state.model_config.get("api_version") or api_version_env_value or provider_config["default_api_version"],
            help="Azure OpenAI API 版本"
        )
        st.session_state.model_config["api_version"] = api_version
        
        deployment_env_value = os.getenv(provider_config["deployment_env"], "")
        deployment = st.sidebar.text_input(
            "部署名称 (可选)",
            value=st.session_state.model_config.get("deployment") or deployment_env_value,
            help="Azure OpenAI 部署名称（如果与模型名称不同）"
        )
        st.session_state.model_config["deployment"] = deployment
    else:
        base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
        base_url = st.sidebar.text_input(
            "Base URL (可选)",
            value=st.session_state.model_config.get("base_url") or base_url_env_value or provider_config.get("default_base_url", ""),
            help=f"{provider_config['name']} API 的 Base URL（留空使用默认值）"
        )
        st.session_state.model_config["base_url"] = base_url
    
    available_models = provider_config["models"]
    
    # if st.sidebar.button("🔄 刷新模型列表", key="refresh_models"):
    #     with st.spinner("正在获取模型列表..."):
    #         if selected_provider == "AzureOpenAI":
    #             success, models, message = fetch_available_models(
    #                 selected_provider, api_key,
    #                 endpoint=st.session_state.model_config.get("endpoint"),
    #                 api_version=st.session_state.model_config.get("api_version")
    #             )
    #         else:
    #             success, models, message = fetch_available_models(
    #                 selected_provider, api_key,
    #                 base_url=st.session_state.model_config.get("base_url")
    #             )
            
    #         if success:
    #             st.sidebar.success(f"✓ {message}")
    #             if models:
    #                 MODEL_PROVIDERS[selected_provider]["models"] = models
    #                 st.session_state.model_config["model"] = models[0]
    #                 st.rerun()
    #         else:
    #             st.sidebar.error(f"✗ {message}")
    
    if st.session_state.model_config["model"] not in available_models:
        st.session_state.model_config["model"] = available_models[0]
    
    selected_model = st.sidebar.selectbox(
        "选择模型",
        options=available_models,
        index=available_models.index(st.session_state.model_config["model"]),
        help=f"选择 {provider_config['name']} 的具体模型"
    )
    st.session_state.model_config["model"] = selected_model
    
    st.sidebar.markdown("### 🎛️ 高级参数")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.model_config.get("temperature", 0.7),
        step=0.1,
        help="控制输出的随机性，值越高输出越随机"
    )
    st.session_state.model_config["temperature"] = temperature
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.sidebar.button("✅ 验证", use_container_width=True):
            is_valid, message = validate_model_config(
                selected_provider,
                api_key,
                base_url=st.session_state.model_config.get("base_url"),
                endpoint=st.session_state.model_config.get("endpoint"),
                api_version=st.session_state.model_config.get("api_version")
            )
            if is_valid:
                st.sidebar.success(f"✓ {message}")
            else:
                st.sidebar.error(f"✗ {message}")
    
    with col2:
        if st.sidebar.button("🔗 测试", use_container_width=True):
            is_valid, _ = validate_model_config(
                selected_provider,
                api_key,
                base_url=st.session_state.model_config.get("base_url"),
                endpoint=st.session_state.model_config.get("endpoint"),
                api_version=st.session_state.model_config.get("api_version")
            )
            if not is_valid:
                st.sidebar.error("✗ 请先完成配置验证")
            else:
                with st.spinner("正在测试连接..."):
                    if selected_provider == "AzureOpenAI":
                        success, message = test_model_connection(
                            selected_provider, selected_model, api_key,
                            endpoint=st.session_state.model_config.get("endpoint"),
                            api_version=st.session_state.model_config.get("api_version")
                        )
                    else:
                        success, message = test_model_connection(
                            selected_provider, selected_model, api_key,
                            base_url=st.session_state.model_config.get("base_url")
                        )
                    if success:
                        st.sidebar.success(f"✓ {message}")
                    else:
                        st.sidebar.error(f"✗ {message}")
    
    with col3:
        if st.sidebar.button("💾 保存", use_container_width=True):
            success, message = save_model_config(st.session_state.model_config)
            if success:
                st.sidebar.success(f"✓ {message}")
            else:
                st.sidebar.error(f"✗ {message}")
    
    st.sidebar.divider()
    
    return st.session_state.model_config

st.title("💬 AI智能对话")

st.sidebar.title("⚙️ 设置")

model_config = get_active_model_config()

is_config_valid, _ = validate_model_config(
    model_config["provider"],
    model_config.get("api_key", ""),
    base_url=model_config.get("base_url"),
    endpoint=model_config.get("endpoint"),
    api_version=model_config.get("api_version")
)

if not is_config_valid:
    st.warning("⚠️ 模型配置不完整，请到「模型配置」页面完成配置后使用")

st.sidebar.markdown("### 🔍 检索设置")
top_k = st.sidebar.slider(
    "检索块数量",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="从知识库中检索的相似文本块数量"
)
st.sidebar.caption("💡 较多的检索块可以提供更全面的信息，但可能增加响应时间")

st.sidebar.markdown("### 🔄 混合检索设置")
use_hybrid = st.sidebar.checkbox(
    "启用混合检索 (向量 + BM25)",
    value=True,
    help="结合向量搜索和关键词搜索，提高检索准确性"
)

use_query_expansion = st.sidebar.checkbox(
    "启用查询扩展",
    value=True,
    help="自动生成语义相近的查询变体，提高召回率"
)

st.sidebar.markdown("### 🌐 联网搜索设置")
web_search_enabled = st.sidebar.checkbox(
    "启用联网搜索",
    value=True,
    help="允许在知识库无法命中时联网检索实时信息"
)

st.sidebar.divider()

if st.sidebar.button("🗑️ 清除对话"):
    st.session_state.chat_messages = []
    st.rerun()

st.sidebar.divider()

st.markdown("""
<div style="padding: 10px; background-color: #e3f2fd; border-radius: 5px; margin-bottom: 20px;">
    <strong>💡 使用说明：</strong>系统会优先检索知识库中的内容回答您的问题。如果知识库中没有相关信息，将自动联网搜索获取答案。
</div>
""", unsafe_allow_html=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message.get("role") == "assistant":
            if message.get("used_knowledge"):
                st.info("📚 基于知识库回答")
                if message.get("knowledge_sources"):
                    with st.expander("📖 知识来源", expanded=False):
                        for i, source in enumerate(message["knowledge_sources"], 1):
                            st.markdown(f"**来源 {i}** (相关度: {source['score']:.2f})")
                            st.markdown(f"{source['content']}")
                            st.markdown(f"*来自: {source['filename']}*")
                            st.divider()
            elif message.get("search_results"):
                st.info("🌐 基于联网搜索回答")
                with st.expander("🔍 搜索来源", expanded=False):
                    for result in message["search_results"]:
                        st.markdown(f"- [{result.get('title', '无标题')}]({result.get('url', '')})")

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 思考中...")
        
        is_valid, validation_msg = validate_model_config(
            model_config["provider"],
            model_config.get("api_key", ""),
            base_url=model_config.get("base_url"),
            endpoint=model_config.get("endpoint"),
            api_version=model_config.get("api_version")
        )
        
        if not is_valid:
            message_placeholder.markdown(f"❌ 模型配置错误: {validation_msg}\n\n请到「模型配置」页面配置模型后重试。")
        else:
            assistant_response = ""
            metadata = None
            error = None
            
            embedding_config = st.session_state.get("embedding_config")
            for content, meta in send_chat_message_stream(st.session_state.chat_messages, top_k, use_hybrid, use_query_expansion, model_config, embedding_config, web_search_enabled=web_search_enabled):
                if meta and "error" in meta:
                    error = meta["error"]
                    break
                if meta:
                    metadata = meta
                if content:
                    assistant_response += content
                    message_placeholder.markdown(assistant_response)
            
            if error:
                message_placeholder.markdown(f"❌ 处理请求时出现错误: {error}")
            elif assistant_response:
                search_results = metadata.get("search_results") if metadata else None
                knowledge_sources = metadata.get("knowledge_sources") if metadata else None
                used_knowledge = metadata.get("used_knowledge") if metadata else False
                
                if used_knowledge:
                    st.info("📚 基于知识库回答")
                    if knowledge_sources:
                        with st.expander("📖 知识来源", expanded=False):
                            for i, source in enumerate(knowledge_sources, 1):
                                st.markdown(f"**来源 {i}** (相关度: {source['score']:.2f})")
                                st.markdown(f"{source['content']}")
                                st.markdown(f"*来自: {source['filename']}*")
                                st.divider()
                elif search_results:
                    st.info("🌐 基于联网搜索回答")
                    with st.expander("🔍 搜索来源", expanded=False):
                        for result in search_results:
                            st.markdown(f"- [{result.get('title', '无标题')}]({result.get('url', '')})")
                
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "search_results": search_results,
                    "knowledge_sources": knowledge_sources,
                    "used_knowledge": used_knowledge
                })
            else:
                message_placeholder.markdown("❌ 抱歉，处理您的请求时出现错误，请稍后重试。")
