import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate

# 忽略警告并防范代理劫持
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['ALL_PROXY'] = ''

# 从 .env 文件中加载 API Key
load_dotenv()

# ==========================================
# 1. 页面基本设置
# ==========================================
st.set_page_config(page_title="自控原理 AI 助教", page_icon="🤖", layout="wide")
st.title("🤖 《自动控制原理》全能智能助教")
st.caption("基于大创项目图文 RAG 知识库构建 | 支持公式推导与原图展示")

# ==========================================
# 2. 全局缓存加载 (超级关键！)
# Streamlit 每次对话都会重新运行整个脚本，必须用 @st.cache_resource 缓存大模型和数据库，防止每次提问都卡顿加载
# ==========================================
@st.cache_resource
def load_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = ChatTongyi(model="qwen-long", streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位非常专业的《自动控制原理》课程助教。
        请你严格根据下面提供的【参考资料】来回答学生的问题。
        【参考资料】:\n{context}\n
        要求：
        1. 回答通俗易懂，逻辑清晰。
        2. 遇到数学公式请必须使用标准 LaTeX 格式。
        3. 若参考资料中没有相关信息，请直接回答“知识库中暂无该部分内容”，不要自行捏造。"""),
        ("human", "学生问题：{question}")
    ])
    return vectorstore, llm, prompt

with st.spinner("🧠 正在唤醒助教记忆库..."):
    vectorstore, llm, prompt_template = load_rag_components()

# ==========================================
# 3. 初始化聊天记录 (Session State)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 将历史聊天记录渲染到页面上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果历史消息里有图片，也要渲染出来
        if "images" in msg and msg["images"]:
            for img in msg["images"]:
                if os.path.exists(img):
                    st.image(img, caption="教材参考原图")

# ==========================================
# 4. 核心问答交互流
# ==========================================
# 如果用户在输入框发出了问题
if query := st.chat_input("向助教提问，例如：什么是奈奎斯特稳定判据？"):
    
    # 4.1 把用户的问题显示在界面上并存入历史记录
    st.session_state.messages.append({"role": "user", "content": query, "images": []})
    with st.chat_message("user"):
        st.markdown(query)

    # 4.2 助教开始思考和回答
    with st.chat_message("assistant"):
        # a. 检索本地数据库
        results = vectorstore.similarity_search(query, k=5)
        context_text = ""
        found_images = []
        
        for i, doc in enumerate(results):
            context_text += f"片段 {i+1}:\n{doc.page_content}\n\n"
            img_path = doc.metadata.get("image_path")
            if img_path:
                found_images.append(img_path)
                
        # 图片去重
        found_images = list(set(found_images))
        
        # b. 流式输出回答 (打字机效果)
        chain = prompt_template | llm
        response_placeholder = st.empty() # 占位符，用来动态刷新字
        full_response = ""
        
        for chunk in chain.stream({"context": context_text, "question": query}):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌") # 光标闪烁效果
        
        response_placeholder.markdown(full_response) # 最后去掉光标
        
        # c. 如果找到了图片，直接在网页上渲染渲染出来！
        if found_images:
            st.divider() # 画一条分割线
            st.markdown("**🖼️ 检索到的教材参考图表：**")
            for img in found_images:
                if os.path.exists(img):
                    st.image(img, use_container_width=True)
                else:
                    st.error(f"⚠️ 图片文件丢失，路径: {img}")
                    
        # d. 把助教的回答存入历史记录
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response, 
            "images": found_images
        })