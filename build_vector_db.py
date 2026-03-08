import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. 读取并切分文档（这里复用你刚才成功的代码）
md_file_path = "output/自动控制原理/hybrid_auto/自动控制原理.md"
with open(md_file_path, "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Sub-section"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = splitter.split_text(markdown_document)
print(f"成功将文档切分为 {len(md_header_splits)} 个知识块，准备开始向量化...")

# 2. 加载 Embedding 模型（首次运行会自动下载这个轻量级中文模型，速度很快）
# BGE 是目前开源界非常强大的中文向量模型，对理工科公式兼容性很好
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 3. 将知识块存入 Chroma 向量数据库
# persist_directory 指定了数据库存放在本地的哪个文件夹里
db_path = "./chroma_db"
print("正在将知识块转化为向量并存入数据库，这可能需要一两分钟，请稍候...")

vectorstore = Chroma.from_documents(
    documents=md_header_splits,
    embedding=embeddings,
    persist_directory=db_path
)

print(f"🎉 太棒了！知识库已成功建立并保存在 {db_path} 文件夹中。")