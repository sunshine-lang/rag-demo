import os
import streamlit as st
from dotenv import load_dotenv

from model_config_utils import (
    MODEL_PROVIDERS,
    validate_model_config,
    test_model_connection,
    save_model_config,
    load_model_config,
)
from embedding_config_utils import (
    EMBEDDING_MODEL_PROVIDERS,
    validate_embedding_config,
    test_embedding_connection,
    save_embedding_config,
    load_embedding_config,
)

load_dotenv(override=True)

# Note: set_page_config is already called in app.py


def build_default_config(provider: str) -> dict:
    provider_config = MODEL_PROVIDERS[provider]
    api_key = os.getenv(provider_config["api_key_env"], "")
    config = {
        "provider": provider,
        "model": provider_config["models"][0],
        "api_key": api_key,
        "base_url": "",
        "endpoint": "",
        "api_version": "",
        "deployment": "",
        "temperature": 0.7,
    }

    if provider == "AzureOpenAI":
        config["endpoint"] = os.getenv(provider_config["endpoint_env"], "")
        config["api_version"] = os.getenv(provider_config["api_version_env"], "") or provider_config["default_api_version"]
        config["deployment"] = os.getenv(provider_config["deployment_env"], "")
    else:
        config["base_url"] = os.getenv(provider_config.get("base_url_env", ""), "") or provider_config.get("default_base_url", "")

    return config


def load_or_init_config() -> dict:
    saved_config = load_model_config()
    if saved_config and saved_config.get("provider") in MODEL_PROVIDERS:
        base = build_default_config(saved_config["provider"])
        base.update(saved_config)
        return base
    return build_default_config("OpenAI")


def render_model_config_page() -> dict:
    st.title("🤖 模型配置")
    st.caption("配置模型供应商、API Key 和推理参数。保存后对聊天页面生效。")

    if "model_config" not in st.session_state:
        st.session_state.model_config = load_or_init_config()

    provider_options = list(MODEL_PROVIDERS.keys())
    provider_display = [MODEL_PROVIDERS[p]["name"] for p in provider_options]

    current_provider = st.session_state.model_config.get("provider", "OpenAI")
    if current_provider not in MODEL_PROVIDERS:
        current_provider = "OpenAI"

    selected_provider_display = st.selectbox(
        "选择供应商",
        options=provider_display,
        index=provider_options.index(current_provider),
        help="选择要使用的 AI 模型供应商",
        key="provider_selectbox_main",
    )

    selected_provider = provider_options[provider_display.index(selected_provider_display)]

    if st.session_state.model_config.get("provider") != selected_provider:
        st.session_state.model_config = build_default_config(selected_provider)
        st.rerun()

    provider_config = MODEL_PROVIDERS[selected_provider]

    st.subheader("基础配置")
    st.caption(f"当前供应商：{provider_config['name']}")

    api_key_env_value = os.getenv(provider_config["api_key_env"], "")
    api_key = st.text_input(
        "API Key",
        value=st.session_state.model_config.get("api_key") or api_key_env_value,
        type="password",
        help=f"输入 {provider_config['name']} 的 API Key（也可通过环境变量 {provider_config['api_key_env']} 配置）",
    )
    st.session_state.model_config["api_key"] = api_key

    if selected_provider == "AzureOpenAI":
        endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
        endpoint = st.text_input(
            "Endpoint",
            value=st.session_state.model_config.get("endpoint") or endpoint_env_value,
            help="Azure OpenAI 服务端点，例如 https://your-resource.openai.azure.com",
        )
        st.session_state.model_config["endpoint"] = endpoint

        api_version_env_value = os.getenv(provider_config["api_version_env"], "")
        api_version = st.text_input(
            "API Version",
            value=st.session_state.model_config.get("api_version") or api_version_env_value or provider_config["default_api_version"],
            help="Azure OpenAI API 版本",
        )
        st.session_state.model_config["api_version"] = api_version

        deployment_env_value = os.getenv(provider_config["deployment_env"], "")
        deployment = st.text_input(
            "部署名称 (可选)",
            value=st.session_state.model_config.get("deployment") or deployment_env_value,
            help="Azure OpenAI 部署名称（如与模型名称不同）",
        )
        st.session_state.model_config["deployment"] = deployment
    else:
        base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
        base_url = st.text_input(
            "Base URL (可选)",
            value=st.session_state.model_config.get("base_url") or base_url_env_value or provider_config.get("default_base_url", ""),
            help=f"{provider_config['name']} API 的 Base URL（留空使用默认值）",
        )
        st.session_state.model_config["base_url"] = base_url

    available_models = provider_config["models"]

    if st.session_state.model_config.get("model") not in available_models:
        st.session_state.model_config["model"] = available_models[0]

    selected_model = st.selectbox(
        "选择模型",
        options=available_models,
        index=available_models.index(st.session_state.model_config["model"]),
        help=f"选择 {provider_config['name']} 的具体模型",
    )
    st.session_state.model_config["model"] = selected_model

    st.subheader("高级参数")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.model_config.get("temperature", 0.7),
        step=0.1,
        help="控制输出的随机性，值越高输出越随机",
    )
    st.session_state.model_config["temperature"] = temperature

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("验证", use_container_width=True):
            is_valid, message = validate_model_config(
                selected_provider,
                api_key,
                base_url=st.session_state.model_config.get("base_url"),
                endpoint=st.session_state.model_config.get("endpoint"),
                api_version=st.session_state.model_config.get("api_version"),
            )
            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    with col2:
        if st.button("测试连接", use_container_width=True):
            is_valid, _ = validate_model_config(
                selected_provider,
                api_key,
                base_url=st.session_state.model_config.get("base_url"),
                endpoint=st.session_state.model_config.get("endpoint"),
                api_version=st.session_state.model_config.get("api_version"),
            )
            if not is_valid:
                st.error("❌ 请先完成配置验证")
            else:
                with st.spinner("正在测试连接..."):
                    if selected_provider == "AzureOpenAI":
                        success, message = test_model_connection(
                            selected_provider,
                            selected_model,
                            api_key,
                            endpoint=st.session_state.model_config.get("endpoint"),
                            api_version=st.session_state.model_config.get("api_version"),
                        )
                    else:
                        success, message = test_model_connection(
                            selected_provider,
                            selected_model,
                            api_key,
                            base_url=st.session_state.model_config.get("base_url"),
                        )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

    with col3:
        if st.button("保存配置", use_container_width=True):
            success, message = save_model_config(st.session_state.model_config)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    return st.session_state.model_config


render_model_config_page()


def build_default_embedding_config(provider: str) -> dict:
    provider_config = EMBEDDING_MODEL_PROVIDERS[provider]
    api_key = os.getenv(provider_config["api_key_env"], "")
    config = {
        "provider": provider,
        "model": list(provider_config["models"].keys())[0],
        "api_key": api_key,
        "base_url": "",
        "endpoint": "",
        "api_version": "",
        "deployment": "",
    }

    if provider == "AzureOpenAI":
        config["endpoint"] = os.getenv(provider_config["endpoint_env"], "")
        config["api_version"] = os.getenv(provider_config["api_version_env"], "") or provider_config["default_api_version"]
        config["deployment"] = os.getenv(provider_config["deployment_env"], "")
    else:
        config["base_url"] = os.getenv(provider_config.get("base_url_env", ""), "") or provider_config.get("default_base_url", "")

    return config


def load_or_init_embedding_config() -> dict:
    saved_config = load_embedding_config()
    if saved_config and saved_config.get("provider") in EMBEDDING_MODEL_PROVIDERS:
        base = build_default_embedding_config(saved_config["provider"])
        base.update(saved_config)
        return base
    return build_default_embedding_config("OpenAI")


def render_embedding_config_section() -> dict:
    st.divider()
    st.header("🧩 向量模型配置")
    st.caption("配置知识库使用的向量/嵌入模型。保存后对知识库上传与检索生效。")

    if "embedding_config" not in st.session_state:
        st.session_state.embedding_config = load_or_init_embedding_config()

    provider_options = list(EMBEDDING_MODEL_PROVIDERS.keys())
    provider_display = [EMBEDDING_MODEL_PROVIDERS[p]["name"] for p in provider_options]

    current_provider = st.session_state.embedding_config.get("provider", "OpenAI")
    if current_provider not in EMBEDDING_MODEL_PROVIDERS:
        current_provider = "OpenAI"

    selected_provider_display = st.selectbox(
        "选择向量供应商",
        options=provider_display,
        index=provider_options.index(current_provider),
        help="选择要使用的向量模型供应商",
        key="embedding_provider_selectbox",
    )

    selected_provider = provider_options[provider_display.index(selected_provider_display)]

    if st.session_state.embedding_config.get("provider") != selected_provider:
        st.session_state.embedding_config = build_default_embedding_config(selected_provider)
        st.rerun()

    provider_config = EMBEDDING_MODEL_PROVIDERS[selected_provider]
    st.caption(f"当前向量供应商：{provider_config['name']}")

    api_key_env_value = os.getenv(provider_config["api_key_env"], "")
    api_key = st.text_input(
        "向量 API Key",
        value=st.session_state.embedding_config.get("api_key") or api_key_env_value,
        type="password",
        help=f"输入 {provider_config['name']} 的 API Key（也可通过环境变量 {provider_config['api_key_env']} 配置）",
        key="embedding_api_key",
    )
    st.session_state.embedding_config["api_key"] = api_key

    if selected_provider == "AzureOpenAI":
        endpoint_env_value = os.getenv(provider_config["endpoint_env"], "")
        endpoint = st.text_input(
            "向量 Endpoint",
            value=st.session_state.embedding_config.get("endpoint") or endpoint_env_value,
            help="Azure OpenAI 服务端点，例如 https://your-resource.openai.azure.com",
            key="embedding_endpoint",
        )
        st.session_state.embedding_config["endpoint"] = endpoint

        api_version_env_value = os.getenv(provider_config["api_version_env"], "")
        api_version = st.text_input(
            "向量 API Version",
            value=st.session_state.embedding_config.get("api_version") or api_version_env_value or provider_config["default_api_version"],
            help="Azure OpenAI API 版本",
            key="embedding_api_version",
        )
        st.session_state.embedding_config["api_version"] = api_version

        deployment_env_value = os.getenv(provider_config["deployment_env"], "")
        deployment = st.text_input(
            "向量部署名称 (可选)",
            value=st.session_state.embedding_config.get("deployment") or deployment_env_value,
            help="Azure OpenAI 部署名称（如与模型名称不同）",
            key="embedding_deployment",
        )
        st.session_state.embedding_config["deployment"] = deployment
    else:
        base_url_env_value = os.getenv(provider_config.get("base_url_env", ""), "")
        base_url = st.text_input(
            "向量 Base URL (可选)",
            value=st.session_state.embedding_config.get("base_url") or base_url_env_value or provider_config.get("default_base_url", ""),
            help=f"{provider_config['name']} API 的 Base URL（留空使用默认值）",
            key="embedding_base_url",
        )
        st.session_state.embedding_config["base_url"] = base_url

    available_models = list(provider_config["models"].keys())

    if st.session_state.embedding_config.get("model") not in available_models:
        st.session_state.embedding_config["model"] = available_models[0]

    selected_model = st.selectbox(
        "选择向量模型",
        options=available_models,
        index=available_models.index(st.session_state.embedding_config["model"]),
        help=f"选择 {provider_config['name']} 的具体向量模型",
        key="embedding_model",
    )
    st.session_state.embedding_config["model"] = selected_model

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("验证向量配置", use_container_width=True, key="embedding_validate"):
            is_valid, message = validate_embedding_config(
                selected_provider,
                api_key,
                base_url=st.session_state.embedding_config.get("base_url"),
                endpoint=st.session_state.embedding_config.get("endpoint"),
                api_version=st.session_state.embedding_config.get("api_version"),
            )
            if is_valid:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    with col2:
        if st.button("测试向量连接", use_container_width=True, key="embedding_test"):
            is_valid, _ = validate_embedding_config(
                selected_provider,
                api_key,
                base_url=st.session_state.embedding_config.get("base_url"),
                endpoint=st.session_state.embedding_config.get("endpoint"),
                api_version=st.session_state.embedding_config.get("api_version"),
            )
            if not is_valid:
                st.error("❌ 请先完成配置验证")
            else:
                with st.spinner("正在测试连接..."):
                    if selected_provider == "AzureOpenAI":
                        success, message = test_embedding_connection(
                            selected_provider,
                            selected_model,
                            api_key,
                            endpoint=st.session_state.embedding_config.get("endpoint"),
                            api_version=st.session_state.embedding_config.get("api_version"),
                        )
                    else:
                        success, message = test_embedding_connection(
                            selected_provider,
                            selected_model,
                            api_key,
                            base_url=st.session_state.embedding_config.get("base_url"),
                        )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

    with col3:
        if st.button("保存向量配置", use_container_width=True, key="embedding_save"):
            success, message = save_embedding_config(st.session_state.embedding_config)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    return st.session_state.embedding_config


render_embedding_config_section()
