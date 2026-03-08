from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. 填入你刚刚生成的 Markdown 文件的绝对路径
md_file_path = "output/自动控制原理/hybrid_auto/自动控制原理.md" # <-- 这里替换成你真实的 md 文件路径

# 读取文件
try:
    with open(md_file_path, "r", encoding="utf-8") as f:
        markdown_document = f.read()
    print("成功读取 Markdown 文件！\n")
except FileNotFoundError:
    print(f"找不到文件，请检查路径是否正确：{md_file_path}")
    exit()

# 2. 定义切分规则：告诉切分器遇到哪些标题就“切一刀”
headers_to_split_on = [
    ("#", "Chapter"),       # 一级标题（章）
    ("##", "Section"),      # 二级标题（节）
    ("###", "Sub-section"), # 三级标题（小节）
]

# 3. 初始化切分器并执行切分
print("正在按章节切分知识块，请稍候...")
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# 4. 验收成果
print(f"\n🎉 切分完成！总共切分出了 {len(md_header_splits)} 个知识块。\n")

# 打印前 3 个知识块看看效果
for i, chunk in enumerate(md_header_splits[:3]):
    print(f"{"="*40}")
    print(f"【知识块 {i+1} 标签】: {chunk.metadata}")
    print(f"【知识块 {i+1} 内容预览】:\n{chunk.page_content[:150]}......\n")