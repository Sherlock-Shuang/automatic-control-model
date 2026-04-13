# 🤖 《自动控制原理》智能助教 & 自动批改系统 (MVP)

本项目是一个基于 **大视觉语言模型 (VLM)** 与 **检索增强生成 (RAG)** 技术的智能作业批改系统。它不仅能回答课程相关问题，更能实现“识别学生手写作业 -> 对比标准答案 -> 给出得分与扣分项分析 -> 引用教材知识点进行个性化反馈”的全流程。

---

## ✨ 核心功能

- 📝 **智能作业批改 (NEW)**：
    - **视觉步骤提取**：采用 Qwen-VL-Plus 识别手写公式、推导步骤及专业符号。
    - **逻辑判分引擎**：通过 Chain-of-Thought (CoT) 将学生解答与教师标答、评分细则 (Rubric) 进行逻辑比对。
    - **RAG 错因诊断**：一旦检出错误，自动检索本地课本向量库，给出专业教材出处的讲解反馈。
- 📚 **全能助教 Q&A**：
    - **图文检索**：支持教材内容、公式及原图的关联检索。
    - **LaTeX 支持**：所有计算推导过程均以标准 LaTeX 格式呈现。
- 💾 **持久化管理**：
    - **任务缓存**：本地 `tasks.json` 保存教师设定的评分规则，服务器重启依然有效。

---

## 🛠️ 技术架构

系统采用前后端分离架构，核心逻辑微服务化：

| 模块 | 技术实现 |
|------|------|
| **后端接口 (API)** | Python + FastAPI |
| **前端界面 (UI)** | Streamlit |
| **AI 编排框架** | LangChain |
| **视觉模型 (VLM)** | 阿里云通义千问 Qwen-VL-Plus (OpenAI 兼容模式) |
| **逻辑/对话模型** | 阿里云通义千问 Qwen-Long / Qwen-Max |
| **本地向量库** | ChromaDB + BAAI/bge-small-zh-v1.5 (本地推理) |

---

## 📁 项目结构

```text
├── backend/                # 核心批改引擎 & 后端服务
│   ├── api/                # API 路由 (作业上传、规则设定)
│   ├── services/           
│   │   ├── ai_pipeline.py  # AI 三阶段工作流 (视觉、匹配、RAG反馈)
│   │   └── local_db.py     # 本地 Chroma 数据库连接器
│   ├── main.py             # 后端入口 (FastAPI)
│   └── requirements.txt    # 后端依赖
├── frontend/               # 老师/助教演示控制台
│   ├── app.py              # Streamlit 界面
│   └── requirements.txt    # 前端依赖
├── build_vector_db.py      # 构建本地教科书向量库 (RAG 核心)
├── chroma_db/              # 本地向量库文件夹 (需自行构建)
├── .env                    # 环境配置文件 (存放 API Key)
└── README.md
```

---

## 🚀 部署与使用

### 1. 安装环境
建议使用虚拟环境（如 Conda 或 venv）：
```bash
# 安装后端依赖
pip install -r backend/requirements.txt

# 安装前端依赖
pip install -r frontend/requirements.txt
```

### 2. 配置密钥
在根目录下创建/修改 `.env` 文件：
```env
# 阿里云百炼 API Key
DASHSCOPE_API_KEY=sk-你的密钥

# 模型选择
VISION_MODEL=qwen-vl-plus
LOGIC_MODEL=qwen-long
```

### 3. 构建本地知识库 (RAG)
运行构建脚本，它会使用 BGE 模型将 `output/` 下的教材 Markdown 向量化：
```bash
python build_vector_db.py
```

### 4. 运行系统
需要同时开启两个终端进程：

**终端 A (后台服务)：**
```bash
python -m backend.main
```

**终端 B (互动界面)：**
```bash
streamlit run frontend/app.py
```

---

## 👨‍🏫 批改演示流程
1. **录入规则**：在网页 **Tab 1** 设定本次作业的题目描述和评分细则（支持 JSON 修改分值）。
2. **执行批改**：在 **Tab 2** 上传学生的作业照片。
3. **查看报告**：系统生成步骤级反馈，对于错题会额外显示“📚 RAG 课本知识引申”，引导学生查漏补缺。

---

## 📄 License
MIT License
