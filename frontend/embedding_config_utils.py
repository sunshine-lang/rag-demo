import os
import json
from typing import Dict, List, Optional

import requests

EMBEDDING_CONFIG_FILE = "embedding_config.json"

EMBEDDING_MODEL_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "https://api.openai.com/v1",
        "models": {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        },
        "requires_base_url": False,
    },
    "AzureOpenAI": {
        "name": "Azure OpenAI",
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "endpoint_env": "AZURE_OPENAI_ENDPOINT",
        "deployment_env": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "api_version_env": "AZURE_OPENAI_API_VERSION",
        "default_api_version": "2024-02-15-preview",
        "models": {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        },
        "requires_endpoint": True,
    },
    "硅基流动": {
        "name": "硅基流动",
        "api_key_env": "SILICONFLOW_API_KEY",
        "base_url_env": "SILICONFLOW_BASE_URL",
        "default_base_url": "https://api.siliconflow.cn/v1",
        "models": {
            "BAAI/bge-large-zh-v1.5": 1024,
            "BAAI/bge-base-zh-v1.5": 768,
            "BAAI/bge-small-zh-v1.5": 512,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
        },
        "requires_base_url": False,
    },
    "阿里百炼": {
        "name": "阿里百炼",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url_env": "DASHSCOPE_BASE_URL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": {
            "text-embedding-v3": 1024,
            "text-embedding-v2": 1536,
            "text-embedding-async-v2": 1536,
        },
        "requires_base_url": False,
    },
}


def validate_embedding_config(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> tuple[bool, str]:
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


def test_embedding_connection(
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> tuple[bool, str]:
    try:
        if provider == "AzureOpenAI":
            url = f"{endpoint.rstrip('/')}/openai/deployments/{model}/embeddings?api-version={api_version}"
            headers = {"api-key": api_key, "Content-Type": "application/json"}
            payload = {"input": "test"}
        else:
            url = f"{base_url.rstrip('/')}/embeddings"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "input": "test"}

        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            return True, "连接测试成功"
        if response.status_code == 401:
            return False, "API Key 无效或已过期"
        if response.status_code == 404:
            return False, "模型不存在或未部署"
        if response.status_code == 429:
            return False, "请求频率超限，请稍后重试"

        error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
        return False, f"连接失败: {error_msg}"
    except requests.exceptions.Timeout:
        return False, "连接超时，请检查网络或服务地址"
    except requests.exceptions.ConnectionError:
        return False, "无法连接到服务器，请检查网络或服务地址"
    except Exception as e:
        return False, f"测试连接时出错: {str(e)}"


def fetch_embedding_models(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> tuple[bool, List[str], str]:
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
            return False, [], "未找到可用模型"

        return False, [], f"获取模型列表失败: HTTP {response.status_code}"
    except Exception as e:
        return False, [], f"获取模型列表时出错: {str(e)}"


def save_embedding_config(config: Dict):
    try:
        with open(EMBEDDING_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return True, "配置已保存"
    except Exception as e:
        return False, f"保存配置失败: {str(e)}"


def load_embedding_config() -> Optional[Dict]:
    try:
        if os.path.exists(EMBEDDING_CONFIG_FILE):
            with open(EMBEDDING_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    except Exception:
        return None
