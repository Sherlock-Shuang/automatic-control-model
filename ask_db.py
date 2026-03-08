import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. 基础配置与鉴权
# ==========================================
# 从 .env 文件中加载 API Key
load_dotenv()

# ==========================================
# 2. 连接本地智库与云端大脑
# ==========================================
db_path = "./chroma_db"
print("⏳ 正在加载本地向量模型和《自动控制原理》图文数据库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

print("⏳ 正在连接阿里云百炼大模型...")
llm = ChatTongyi(model="qwen-long", streaming=True)

# ==========================================
# 3. 设计助教的思想钢印 (Prompt)
# ==========================================
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一位非常专业的《自动控制原理》课程助教。
请你严格根据下面提供的【参考资料】来回答学生的问题。

【参考资料】:
{context}

要求：
1. 回答要通俗易懂，逻辑清晰，重点突出。
2. 遇到公式请使用标准 LaTeX 格式。
3. 如果提供的参考资料中没有能回答该问题的信息，请直接回答“根据教材当前范围，我无法给出准确解答”，绝对不能依靠自身的预训练记忆去胡编乱造。
"""),
    ("human", "学生问题：{question}")
])

print("\n" + "="*50)
print("✅ 图文双修的智能助教已上线！")
print("="*50)

# ==========================================
# 4. 终端问答与寻图循环
# ==========================================
while True:
    query = input("\n📝 请提问 (输入 '退出' 结束): ")
    if query.lower() in ["退出", "exit", "quit", "q"]:
        print("智能助教下线，再见！")
        break
        
    if not query.strip():
        continue

    # 第一阶段：开卷搜索 (在本地 Mac 运行)
    # 搜出最相关的 5 段书本原文或图片描述
    results = vectorstore.similarity_search(query, k=5)
    
    context_text = ""
    found_images = [] # 准备一个列表，专门用来装检索到的图片路径
    
    for i, doc in enumerate(results):
        # 拼接给大模型看的纯文本参考资料
        context_text += f"片段 {i+1}:\n{doc.page_content}\n\n"
        
        # 核心逻辑：精准拦截并提取图片路径！
        # 使用 .get() 方法更安全，因为有些纯文本 Markdown 数据没有 image_path
        img_path = doc.metadata.get("image_path")
        if img_path:
            found_images.append(img_path)

    print("\n[系统: 已从教材中检索到相关图文知识，正在思考...]\n")
    print("🤖 助教回答：")

    # 第二阶段：阅读并生成 (在云端运行)
    chain = prompt_template | llm
    
    # 启用流式输出，打字机效果
    for chunk in chain.stream({"context": context_text, "question": query}):
        print(chunk.content, end="", flush=True)
    print("\n")
    
    # 第三阶段：展示原图路径
    if found_images:
        print("\n" + "-"*30)
        print("🖼️ 附带参考教材原图 (终端暂只显示路径):")
        # 用 set 去重，防止同一个图片被引用多次
        for img_path in set(found_images): 
            print(f"   👉 {img_path}")
        print("-"*30)