# 《自动控制原理》课程助教与作业辅助批改

课程问答使用本地教材检索；教师工作台支持题目与评分规则管理、图片/PDF 辅助批改、原作业保存和人工复核。AI 输出为建议分，教师确认后才标记为已复核。

## 安装

建议使用 Python 3.12，在项目根目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

把 `.env.example` 复制为 `.env`，填写自己的模型密钥。仓库只包含应用代码、依赖、配置模板及使用说明；教材、知识库、学生作业、任务记录、评测脚本与结果不随代码分发。

## 配置

| 配置项 | 用途 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 模型服务密钥；也支持 `OPENAI_API_KEY` |
| `OPENAI_BASE_URL` | OpenAI 兼容模型接口地址 |
| `VISION_MODEL` / `LOGIC_MODEL` | 视觉识别与评分/问答使用的模型 |
| `HOMEWORK_API_BASE_URL` | 教师工作台连接的后端根地址，默认 `http://localhost:8000` |
| `HOMEWORK_TASKS_PATH` | 本地任务文件，默认 `output/tasks.json`；相对路径基于项目根目录 |
| `CHROMA_DB_PATH` / `CHROMA_COLLECTION` | 已有教材知识库的位置与集合名 |
| `EMBEDDING_MODEL` | 建库时使用的嵌入模型名称或本地模型目录 |
| `EMBEDDING_LOCAL_FILES_ONLY` | 默认 `true`，只加载本地已有模型 |

没有填写任务时工作台保持空白，不加载固定示例题或预设答案。题干、标准答案和评分细则由教师输入并保存在本地任务文件中。

## 准备教材知识库

请通过团队约定的资料渠道取得教材索引与所需教材原图，放到本地目录，再配置 `CHROMA_DB_PATH`。克隆代码本身不会带回教材和学生资料。数据库不存在、为空或与嵌入模型不匹配时，应用会提示知识库不可用。

嵌入模型尚未缓存时，可显式允许首次下载，完成后恢复为本地加载。若已有模型目录，直接在 `.env` 中配置 `EMBEDDING_MODEL`。

```powershell
$env:EMBEDDING_LOCAL_FILES_ONLY = 'false'
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); from backend.services.local_db import get_vectorstore; get_vectorstore()"
$env:EMBEDDING_LOCAL_FILES_ONLY = 'true'
```

旧索引无法打开时，可从其中保存的原文重建到独立目录。下列工具先只读检查；加 `--run` 才写入新索引。源目录保留，目标目录必须为空或不存在。

```powershell
.\.venv\Scripts\python.exe scripts/rebuild_knowledge_base.py --source chroma_db --destination output/course-knowledge-base
.\.venv\Scripts\python.exe scripts/rebuild_knowledge_base.py --source chroma_db --destination output/course-knowledge-base --run
```

重建保留完整原文及分块位置，并独立重新打开索引检查持久化结果。完成清单状态为 `complete` 后再更新 `CHROMA_DB_PATH`，并重新启动应用。

Windows 下请从项目根目录运行。为兼容底层索引库的中文路径限制，应用在必要时使用指向同一目录的英文相对路径；不会移动教材或建立外部目录联接。

## 启动

Windows PowerShell 7 可使用后台启动器；日志保存在本地 `output/runtime/`。

```powershell
.\scripts\start_local.ps1 -Mode teacher
.\scripts\start_local.ps1 -Mode chat
```

- 教师工作台：http://127.0.0.1:8501
- 后端：http://127.0.0.1:8000
- 课程问答：http://127.0.0.1:8502

其他平台或手动启动，在激活虚拟环境后的三个终端中分别运行：

```text
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

`scripts/check_runtime.py` 检查本地依赖及已有索引，不调用远程模型。正常访问首页、编辑任务和查看历史不需要调用模型；识别、评分和生成问答会调用配置的服务。

## 使用流程

1. 创建题目，填写完整条件、教师确认的标准答案及逐项评分细则；也可加载已有本地任务。
2. 单份或批量上传作业。同一份多页作业请合并为 PDF；每个文件不超过 20 MB，PDF 最多 20 页，每批最多 100 份。
3. 对照原件、识别文字、逐项建议分和判分依据。无法辨认、教师红笔批注、无效模型输出或矛盾证据会进入待复核，不能当作零分统计。
4. 教师确认或调整逐项成绩，填写复核人及说明。系统保存原 AI 建议和人工复核记录。

教材讲解只引用教材正文，图片生成说明仅用于定位原图。讲解失败不会覆盖已校验的建议分；教材依据不足时不能编造结论。模型输出不等同于教师最终成绩。

当前面向本机、单后端进程使用。复核人字段属于本机记录，并非经过身份认证的电子签名。

## 代码结构

- `frontend/`：教师工作台及共享界面样式。
- `app.py` / `ask_db.py`：网页与命令行课程问答。
- `backend/api/`：任务、作业、报告与人工复核接口。
- `backend/services/`：转录、评分校验、教材检索、原件与报告存储。
- `scripts/`：本地环境检查、索引重建和启动工具。
- `output/`：本地生成的任务、报告、原件和运行记录，不进入仓库。
