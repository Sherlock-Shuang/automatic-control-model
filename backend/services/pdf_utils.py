"""
PDF 工具模块：将 PDF 文件转换为图片，支持批量处理学生作业扫描件。
"""
import base64
import io
import fitz  # PyMuPDF


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[bytes]:
    """
    将 PDF 文件的每一页转换为 JPEG 图片的字节列表。
    
    Args:
        pdf_bytes: PDF 文件的原始字节
        dpi: 输出图片的分辨率，默认 200 DPI（平衡清晰度和文件大小）
    
    Returns:
        每页图片的 JPEG 字节列表
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    
    for page in doc:
        # 渲染页面为像素图
        pix = page.get_pixmap(dpi=dpi)
        # 转换为 JPEG 字节
        img_bytes = pix.tobytes("jpeg")
        images.append(img_bytes)
    
    doc.close()
    return images


def pdf_page_to_base64(pdf_bytes: bytes, page_index: int = 0, dpi: int = 150) -> str:
    """
    将 PDF 的指定页转换为 base64 编码的 JPEG 字符串。
    如果 PDF 有多页，将所有页面拼接为一张长图。
    
    Args:
        pdf_bytes: PDF 文件的原始字节
        page_index: 页码索引（0-based），-1 表示拼接所有页
        dpi: 输出图片的分辨率
    
    Returns:
        base64 编码的 JPEG 字符串
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    if page_index == -1:
        # 拼接所有页面为一张长图
        from PIL import Image
        
        page_images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append(img)
        
        doc.close()
        
        if not page_images:
            return ""
        
        # 计算总高度
        total_height = sum(img.height for img in page_images)
        max_width = max(img.width for img in page_images)
        
        # 创建长图
        combined = Image.new("RGB", (max_width, total_height), "white")
        y_offset = 0
        for img in page_images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
        
        # 转换为 base64
        buffer = io.BytesIO()
        combined.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        # 只处理指定页
        if page_index < 0 or page_index >= len(doc):
            doc.close()
            return ""
        
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("jpeg")
        doc.close()
        return base64.b64encode(img_bytes).decode("utf-8")


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """获取 PDF 的页数"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def extract_student_id_from_filename(filename: str) -> str:
    """
    从文件名中提取学生学号。
    支持格式：U202115226-贾惠云.pdf -> U202115226
    """
    if "-" in filename:
        return filename.split("-")[0]
    if "_" in filename:
        return filename.split("_")[0]
    # 去掉扩展名
    name = filename.rsplit(".", 1)[0]
    return name
