# 🤖 《自动控制原理》智能助教

基于 RAG（检索增强生成）技术构建的《自动控制原理》课程智能助教系统，支持图文检索、公式推导与教材原图展示。

## ✨ 功能特性

- 📚 **图文双模态 RAG**：结合教材文本与图片，提供精准的知识检索
- 🧮 **LaTeX 公式支持**：自动使用标准 LaTeX 格式输出数学公式
- 🖼️ **教材原图展示**：检索到相关图表时直接在页面展示教材原图
- 💬 **流式对话**：打字机效果的实时回答，支持多轮对话历史
- 🔍 **语义检索**：基于 BGE 中文向量模型的高质量语义搜索

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| 前端界面 | Streamlit |
| 大语言模型 | 阿里云百炼 Qwen-Long |
| 视觉语言模型 | Qwen-VL-Plus / Ollama (本地) |
| 向量数据库 | ChromaDB |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| 框架 | LangChain |

## 📁 项目结构

```
├── app.py                  # Streamlit Web 应用主程序
├── ask_db.py               # 终端问答脚本
├── build_vector_db.py      # 从 Markdown 构建向量数据库
├── split_book.py           # 按章节切分 Markdown 文档
├── process_images_to_db.py # 云端 VLM 处理图片并入库
├── local_process_images.py # 本地 Ollama VLM 处理图片并入库
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
└── README.md
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/<你的用户名>/<仓库名>.git
cd <仓库名>
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的阿里云百炼 API Key：

```
DASHSCOPE_API_KEY=sk-你的真实密钥
```

> 💡 API Key 申请地址：[阿里云百炼平台](https://bailian.console.aliyun.com/)

### 4. 准备知识库

#### 4.1 准备教材 Markdown 文件

将教材 PDF 转换为 Markdown 格式，放入 `output/` 目录。

#### 4.2 构建文本向量数据库

```bash
python build_vector_db.py
```

#### 4.3 处理教材图片（二选一）

**方案 A：使用阿里云 VLM（需联网）**

```bash
python process_images_to_db.py
```

**方案 B：使用本地 Ollama（需安装 Ollama + qwen3-vl:8b）**

```bash
python local_process_images.py
```

### 5. 启动助教

**Web 界面（推荐）：**

```bash
streamlit run app.py
```

**终端模式：**

```bash
python ask_db.py
```

## ⚠️ 注意事项

- `chroma_db/` 和 `output/` 目录包含大量数据文件，已通过 `.gitignore` 排除，需要本地自行构建
- 请勿将 `.env` 文件提交到 Git，其中包含你的 API 密钥
- 首次运行会自动下载 `bge-small-zh-v1.5` 模型（约 100MB）

## 📄 License

MIT License
