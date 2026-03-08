import os
import glob
import time
from dotenv import load_dotenv
from tqdm import tqdm
import dashscope
from dashscope import MultiModalConversation
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import warnings

# 忽略不必要的警告，保持终端整洁
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 1. 基础配置
# ==========================================
# 从 .env 文件中加载 API Key
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

image_folder = "output/自动控制原理/hybrid_auto/images"
db_path = "./chroma_db"
BATCH_SIZE = 20  # 每处理 20    张图自动存入一次数据库
    
VLM_PROMPT = """你是一位顶级的《自动控制原理》授课教授。
请仔细观察这张教材中的图片，并将其转化为极其详尽的纯文本描述，以便录入到知识库中供后续检索。

请遵循以下规则：
1. 【图表类型识别】：首先说明这是什么图（如：闭环系统结构框图、根轨迹图、开环对数频率特性曲线/波特图、奈奎斯特图、阶跃响应曲线等）。
2. 【核心参数提取】：
   - 如果是系统框图：必须用文字写出前向通道 G(s) 和反馈通道 H(s) 的具体传递函数。
   - 如果是曲线图：提取关键特征（如：起始点坐标、渐近线角度、分离点、穿越频率 ωc、相角裕度 γ 等）。
3. 【格式要求】：所有的变量、传递函数和数学公式，必须严格使用 LaTeX 格式。
4. 【结论】：如果图中有明显的文字标注或代表了某种稳定性结论，请一并总结出来。
"""

# ==========================================
# 2. 核心函数：调用 Qwen-VL-Max (带自动重试机制)
# ==========================================
def get_image_description(image_path, max_retries=3):
    abs_path = os.path.abspath(image_path)
    local_file_url = f"file://{abs_path}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": local_file_url},
                {"text": VLM_PROMPT}
            ]
        }
    ]
    
    # 自动重试机制：应对网络抖动和 API 限流
    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                model='qwen-vl-plus',
                messages=messages
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]['text']
            elif response.status_code == 429: # 触发并发限流
                wait_time = 2 ** attempt
                print(f"\n⚠️ 触发 API 限流，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ API 调用失败: {response.code} - {response.message}")
                return None
        except Exception as e:
            wait_time = 2 ** attempt
            print(f"\n❌ 网络异常: {e}，等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
            
    print(f"\n❌ 图片 {os.path.basename(image_path)} 多次尝试均失败，跳过。")
    return None

# ==========================================
# 3. 主流程：断点续传 + 分批入库
# ==========================================
def main():
    image_files = glob.glob(os.path.join(image_folder, "*.[jp][pn]g"))
    if not image_files:
        print(f"⚠️ 在 {image_folder} 下没有找到图片，请检查路径。")
        return

    print("⏳ 正在加载本地 Embedding 模型 (bge-small-zh-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    print("🔍 正在检查本地记忆库，准备断点续传...")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    try:
        # 提取已存入数据库的图片路径
        existing_data = vectorstore.get(include=['metadatas'])
        processed_paths = set(
            meta.get("image_path") for meta in existing_data['metadatas'] if meta and "image_path" in meta
        )
    except Exception:
        processed_paths = set()

    # 过滤掉已经处理过的图片
    unprocessed_files = [f for f in image_files if f not in processed_paths]
    
    if not unprocessed_files:
        print(f"\n🎉 太棒了！全部 {len(image_files)} 张图片都已经存在于数据库中，无需重复处理！")
        return

    print(f"✅ 发现记忆库中已存在 {len(processed_paths)} 张图。")
    print(f"📸 剩余 {len(unprocessed_files)} 张图片需要向云端请求，发车！\n")
    
    documents_to_add = []
    
    # 进度条开始
    for img_path in tqdm(unprocessed_files, desc="☁️ 云端看图中"):
        description = get_image_description(img_path)
        
        if description:
            doc = Document(
                page_content=description,
                metadata={
                    "source_type": "image",
                    "image_path": img_path,
                    "chapter": "图表库"
                }
            )
            documents_to_add.append(doc)
            
            # 分批保存逻辑：满 BATCH_SIZE 张就存一次
            if len(documents_to_add) >= BATCH_SIZE:
                vectorstore.add_documents(documents_to_add)
                documents_to_add = [] # 清空缓存，准备下一批
                # 稍微停顿，防止写入过快
                time.sleep(0.5) 
                
    # 循环结束后，把剩下不足 BATCH_SIZE 的尾巴存入数据库
    if documents_to_add:
        vectorstore.add_documents(documents_to_add)

    print("\n🎉 大功告成！所有云端提取的图片知识已注入你的本地助教大脑！")

if __name__ == "__main__":
    main()