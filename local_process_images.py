import argparse
import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
import glob
import time
import ollama
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==========================================
# 1. 基础配置
# ==========================================
image_folder = "output/自动控制原理/hybrid_auto/images"
db_path = "./chroma_db"

# 你的专属助教提示词
VLM_PROMPT = """你是一位顶级的《自动控制原理》授课教授。
请仔细观察这张教材中的图片，并将其转化为极其详尽的纯文本描述，以便录入到知识库中供后续检索。

请遵循以下规则：
1. 【图表类型识别】：首先说明这是什么图（如：结构框图、根轨迹、波特图、奈奎斯特图等）。
2. 【核心参数提取】：框图必须写出前向和反馈传递函数；曲线图必须提取关键点坐标、渐近线、穿越频率等。
3. 【格式要求】：所有的变量和公式必须严格使用 LaTeX 格式输出。
4. 【结论】：如果有明显的文字标注或代表了某种稳定性结论，请一并总结。
"""

# ==========================================
# 2. 调用本地 Ollama 视觉模型
# ==========================================
def get_local_image_description(image_path, retries: int = 3, backoff: float = 1.0):
    """调用本地 Ollama 提取图片描述。

    发生502等临时错误时会自动重试，最多 `retries` 次，指数退避。
    如果最终仍然失败则返回 None。
    """
    time.sleep(0.1)
    # 一些版本的 Ollama 可能更喜欢绝对路径，特别是当路径中包含中文时
    image_path = os.path.abspath(image_path)
    for attempt in range(1, retries + 1):
        try:
            # Ollama 接口极其简洁，直接把路径塞给它即可
            response = ollama.chat(
                model='qwen3-vl:8b',
                messages=[{
                    'role': 'user',
                    'content': VLM_PROMPT,
                    'images': [image_path]
                }],
                options={
                "num_ctx": 4096,      # 限制上下文窗口大小，防止图片转出的 Token 撑爆内存
                "num_predict": 256,   # 限制它最多只输出 256 个词的描述，见好就收
                "temperature": 0.1    # 降低发散度，让它专注提取事实
            }
            )
            return response['message']['content']
        except Exception as e:
            msg = str(e)
            print(f"\n❌ 解析图片 {image_path} 时出错: {msg}")
            # 只对 502 等网关错误进行重试
            if attempt < retries and "502" in msg:
                wait = backoff * (2 ** (attempt - 1))
                print(f"   ➤ 第 {attempt} 次失败，将在 {wait:.1f}s 后重试... (共 {retries} 次)")
                time.sleep(wait)
                continue
            # 其他错误或用尽重试次数直接放弃
            return None

# ==========================================
# 3. 主流程
# ==========================================
def main(limit: int | None = None):
    image_files = glob.glob(os.path.join(image_folder, "*.[jp][pn]g"))
    if not image_files:
        print(f"⚠️ 找不到图片，请检查路径: {image_folder}")
        return

    # ==========================================
    # [新增] 读取数据库，实现真正的断点续传
    # ==========================================
    print("\n🔍 正在检查记忆库，准备断点续传...")
    try:
        # 悄悄连接一下数据库
        vectorstore = Chroma(
            persist_directory=db_path, 
            embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        )
        # 获取数据库中所有已经存好的元数据 (metadata)
        existing_data = vectorstore.get(include=['metadatas'])
        
        # 提取出所有已经处理过的图片路径，存入一个集合 (Set) 中加速查找
        processed_paths = set(
            meta.get("image_path") for meta in existing_data['metadatas'] if meta and "image_path" in meta
        )
        
        # 核心：只保留还没被处理过的图片！
        image_files = [f for f in image_files if f not in processed_paths]
        print(f"✅ 发现记忆库中已存在 {len(processed_paths)} 张图。")
        
    except Exception as e:
        print("⚠️ 第一次运行或读取数据库为空，将从头开始。")

    if not image_files:
        print("\n🎉 太棒了！所有的图片都已经存在于数据库中，无需重复处理！")
        return
        
    # 支持通过命令行限制处理数量（便于测试）
    if limit is not None:
        image_files = image_files[: limit]

    print(f"📸 剩余 {len(image_files)} 张图片需要处理，开始调用本地 M5 算力进行看图...")
    
    documents_to_add = []
    
    # 进度条包裹循环，跑几千张图心里有底
    for img_path in tqdm(image_files, desc="🧠 本地看图中"):
        description = get_local_image_description(img_path)
        
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
            
            # 每处理完 100 张图就存一次数据库，防止断电白跑
            if len(documents_to_add) >= 100:
                print("\n💾 正在将这 100 张图片的记忆存入数据库...")
                vectorstore = Chroma(
                    persist_directory=db_path, 
                    embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
                )
                vectorstore.add_documents(documents_to_add)
                documents_to_add = [] # 清空列表，准备下一批
                
    # 循环结束后，把剩下不足 100 张的尾巴存入数据库
    if documents_to_add:
        print("\n💾 正在存入最后一部分图片记忆...")
        vectorstore = Chroma(
            persist_directory=db_path, 
            embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        )
        vectorstore.add_documents(documents_to_add)

    print("\n🎉 大功告成！所有图片已转化为本地大模型的数字记忆！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将指定文件夹内的图片送入 Ollama 视觉模型并存储为向量文档。"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="只处理前 N 张图片（便于调试）。"
    )
    args = parser.parse_args()
    main(limit=args.limit)