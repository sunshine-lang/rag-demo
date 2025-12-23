# AI 智能助手

一个功能强大的 AI 智能助手，支持知识库管理、多模型对话、混合检索和联网搜索。

## 功能特性

- 🤖 **多模型支持**：支持 OpenAI、Azure OpenAI、硅基流动、阿里百炼、DeepSeek 等多种 LLM
- 📚 **知识库管理**：支持上传 PDF、DOCX、TXT 文档，自动分块并生成向量索引
- 🔍 **混合检索**：结合向量检索和 BM25 关键词检索，提高检索准确率
- 🌐 **联网搜索**：集成 Tavily 搜索，获取最新实时信息
- 🎯 **查询扩展**：自动生成查询变体，提高召回率
- 🎨 **Streamlit 界面**：简洁美观的 Web 界面，支持对话和知识库管理
- 🐳 **Docker 部署**：使用 Docker Compose 快速部署 Milvus 向量数据库

## 项目结构

```
ademo/
├── backend/
│   ├── main.py                 # FastAPI 后端服务
│   ├── uploads/                # 上传文件存储目录
│   ├── documents.json          # 文档元数据
│   └── bm25_index.json         # BM25 索引文件
├── frontend/
│   ├── app.py                  # Streamlit 主应用
│   ├── chat.py                 # AI 对话页面
│   ├── knowledge_management.py # 知识库管理页面
│   ├── model_config.json       # 模型配置
│   └── embedding_config.json   # 嵌入模型配置
├── .env                        # 环境变量（不提交）
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略文件
├── requirements.txt            # Python 依赖
├── docker-compose.yml          # Docker Compose 配置
└── README.md                   # 项目说明
```

## 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **向量数据库**：Milvus
- **LLM 框架**：LangChain
- **关键词检索**：BM25 + Jieba 分词
- **联网搜索**：Tavily

## 快速开始

### 前置要求

- Python 3.11+
- Docker & Docker Compose（用于 Milvus）

### 1. 克隆项目

```bash
git clone https://github.com/your-username/ademo.git
cd ademo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
copy .env.example .env
```

编辑 `.env` 文件，至少配置以下项：

```env
# 选择一个 LLM 提供商并配置
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# 或使用 Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-small

# Tavily 联网搜索（可选）
TAVILY_API_KEY=your_tavily_api_key

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 后端地址
API_URL=http://localhost:8000
```

### 4. 启动 Milvus 向量数据库

```bash
docker-compose up -d
```

等待 Milvus 启动完成（约 1-2 分钟）。

### 5. 启动后端服务

```bash
cd backend
python main.py
```

后端将在 http://localhost:8000 运行

### 6. 启动前端应用

```bash
cd frontend
streamlit run app.py
```

前端将在 http://localhost:8501 运行

## 使用说明

### AI 对话

1. 在左侧选择 "💬 AI对话" 页面
2. 配置模型提供商和模型名称
3. 输入问题开始对话
4. 系统会自动判断是否需要联网搜索
5. 如果知识库中有相关内容，会自动引用

### 知识库管理

1. 在左侧选择 "📚 知识库管理" 页面
2. 配置嵌入模型提供商和模型名称
3. 点击 "上传文档" 选择文件（支持 PDF、DOCX、TXT）
4. 设置分块大小和重叠大小
5. 等待文档处理完成
6. 可以查看已上传的文档列表

## API 端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/` | 服务状态 |
| GET | `/health` | 健康检查 |
| GET | `/documents` | 获取文档列表 |
| POST | `/upload` | 上传文档 |
| POST | `/chat` | AI 对话 |
| DELETE | `/documents/{doc_id}` | 删除文档 |

### 聊天请求示例

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "system_prompt": "You are a helpful assistant.",
  "top_k": 5,
  "use_hybrid": true,
  "use_query_expansion": true
}
```

### 聊天响应示例

```json
{
  "response": "你好！有什么我可以帮助你的吗？",
  "search_results": [],
  "knowledge_sources": [],
  "used_knowledge": false
}
```

## 支持的模型提供商

- **OpenAI**：GPT-4、GPT-3.5 等模型
- **Azure OpenAI**：Azure 托管的 OpenAI 模型
- **硅基流动**：DeepSeek、Qwen 等开源模型
- **阿里百炼**：通义千问系列模型
- **DeepSeek**：DeepSeek 系列模型

## 混合检索原理

本项目使用 **RRF (Reciprocal Rank Fusion)** 算法融合向量检索和 BM25 检索结果：

1. **向量检索**：使用嵌入向量进行语义相似度检索
2. **BM25 检索**：使用关键词匹配进行精确检索
3. **RRF 融合**：将两种检索结果按排名进行加权融合

## 常见问题

### Milvus 连接失败

确保 Docker 容器正在运行：

```bash
docker-compose ps
```

### 文档上传失败

检查文件格式是否支持（PDF、DOCX、TXT），并确保文件不是图片型 PDF。

### 模型调用失败

检查 `.env` 文件中的 API Key 和 Base URL 是否正确配置。

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！