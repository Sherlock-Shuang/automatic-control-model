import json
import base64
import asyncio
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List

from backend.models.schemas import (
    TaskCreateRequest, GradingReportResponse, GradingReportItem, RubricItem,
    BatchGradingRequest, BatchGradingResponse, BatchGradingSummary
)
from backend.services.ai_pipeline import node_a_extract_steps, node_b_logic_matcher, node_c_rag_feedback
from backend.services.local_db import similarity_search
from backend.services.pdf_utils import pdf_to_images, extract_student_id_from_filename

import os

router = APIRouter()

TASKS_FILE = "tasks.json"
tasks_db = {}

# 启动时自动从本地文件加载已有的任务规则
if os.path.exists(TASKS_FILE):
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 将字典转换回 Pydantic 对象
            for tid, tdata in data.items():
                tasks_db[tid] = TaskCreateRequest(**tdata)
        print(f"✅ 已从本地加载了 {len(tasks_db)} 个作业任务规则。")
    except Exception as e:
        print(f"⚠️ 加载备份任务失败: {e}")

def save_tasks():
    """将内存中的任务保存到本地文件"""
    try:
        with open(TASKS_FILE, "w", encoding="utf-8") as f:
            json_data = {tid: task.model_dump() for tid, task in tasks_db.items()}
            json.dump(json_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ 保存任务失败: {e}")


def _extract_json_from_response(raw_text: str) -> str:
    """从 LLM 响应中提取 JSON 内容，兼容各种格式"""
    if "```json" in raw_text:
        return raw_text.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_text:
        parts = raw_text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()
    # 尝试直接找到 JSON 数组
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1]
    return raw_text.strip()


def _process_single_student(
    task_info: TaskCreateRequest,
    image_b64: str,
    student_id: str
) -> GradingReportResponse:
    """
    核心批改逻辑：处理单个学生的作业图片。
    三阶段 AI Pipeline：视觉提取 → 逻辑匹配 → RAG 反馈
    """
    # 1. Node A: Visual parsing
    student_steps_raw = node_a_extract_steps(image_b64)
    student_steps_raw = _extract_json_from_response(student_steps_raw)

    # 2. Node B: Logic matching
    match_result_raw = node_b_logic_matcher(student_steps_raw, task_info.standard_answer, task_info.rubric)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = json.loads(match_result_raw)

    # 3. Node C: RAG Feedback
    try:
        final_assessment = node_c_rag_feedback(grading_results, similarity_search)
    except Exception as e:
        print(f"⚠️ RAG 反馈生成失败，使用原始结果: {e}")
        final_assessment = {
            "enhanced_results": grading_results,
            "overall": "RAG 检索不可用，请参考上方步骤级结果。"
        }

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

    return GradingReportResponse(
        task_id=task_info.task_id,
        student_id=student_id,
        total_score=total_score,
        details=details,
        overall_feedback=final_assessment["overall"]
    )


def _process_pdf_pages(task_info: TaskCreateRequest, pdf_bytes: bytes, student_id: str) -> GradingReportResponse:
    """
    【迭代优化】PDF 逐页处理：
    将每页单独送入 VLM 提取步骤，然后合并结果进行逻辑匹配。
    避免多页拼接成一张大图导致 VLM 超时。
    """
    # 1. 将 PDF 每页转为独立图片
    page_images = pdf_to_images(pdf_bytes, dpi=150)
    if not page_images:
        raise ValueError("PDF 文件为空或无法解析。")

    print(f"📄 PDF 共 {len(page_images)} 页，开始逐页视觉提取...")

    # 2. 逐页调用 VLM 提取步骤
    all_steps = []
    step_counter = 1
    for i, img_bytes in enumerate(page_images):
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"  🔍 正在解析第 {i+1}/{len(page_images)} 页...")

        try:
            page_result = node_a_extract_steps(img_b64)
            page_result = _extract_json_from_response(page_result)
            page_steps = json.loads(page_result)
            # 重新编号步骤，确保连续
            for step in page_steps:
                step["step_id"] = step_counter
                step_counter += 1
            all_steps.extend(page_steps)
            print(f"    ✅ 第 {i+1} 页提取了 {len(page_steps)} 个步骤")
        except Exception as e:
            print(f"    ⚠️ 第 {i+1} 页解析失败: {e}")
            all_steps.append({
                "step_id": step_counter,
                "content": f"[第{i+1}页解析失败: {str(e)[:80]}]"
            })
            step_counter += 1

    if not all_steps:
        raise ValueError("所有页面均解析失败。")

    print(f"  ✅ 共提取 {len(all_steps)} 个步骤，开始逻辑匹配...")

    # 3. 合并后的步骤送入逻辑匹配
    student_steps_json = json.dumps(all_steps, ensure_ascii=False)

    match_result_raw = node_b_logic_matcher(student_steps_json, task_info.standard_answer, task_info.rubric)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = json.loads(match_result_raw)

    # 4. RAG Feedback
    try:
        final_assessment = node_c_rag_feedback(grading_results, similarity_search)
    except Exception as e:
        print(f"⚠️ RAG 反馈生成失败，使用原始结果: {e}")
        final_assessment = {
            "enhanced_results": grading_results,
            "overall": "RAG 检索不可用，请参考上方步骤级结果。"
        }

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

    return GradingReportResponse(
        task_id=task_info.task_id,
        student_id=student_id,
        total_score=total_score,
        details=details,
        overall_feedback=final_assessment["overall"]
    )


@router.post("/upload_task")
async def upload_task(task: TaskCreateRequest):
    """教师上传本次作业的题干、标准答案及得分细则 (Rubric)"""
    tasks_db[task.task_id] = task
    save_tasks()  # 存入本地文件
    return {"message": "Task created successfully", "task_id": task.task_id}


@router.get("/tasks")
async def list_tasks():
    """获取所有已创建的作业任务列表"""
    return {
        "tasks": [
            {
                "task_id": tid,
                "title": task.title,
                "rubric_count": len(task.rubric),
                "total_points": sum(r.points for r in task.rubric)
            }
            for tid, task in tasks_db.items()
        ]
    }


@router.post("/grade_homework", response_model=GradingReportResponse)
def grade_homework(
    task_id: str = Form(...),
    student_id: str = Form(...),
    image: UploadFile = File(...)
):
    """批改单个学生的作业（支持图片和 PDF 格式）"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail=f"任务 '{task_id}' 不存在，请先创建作业规则。")

    task_info = tasks_db[task_id]

    # 读取文件内容
    file_bytes = image.read()
    filename = image.filename or ""

    try:
        if filename.lower().endswith(".pdf"):
            return _process_pdf_pages(task_info, file_bytes, student_id)
        else:
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")
            return _process_single_student(task_info, image_b64, student_id)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI 返回格式解析失败，请重试。错误详情: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批改过程出错: {str(e)}")


@router.post("/grade_homework_batch")
async def grade_homework_batch(
    task_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    批量批改：一次上传多个学生作业文件（图片或 PDF）。
    自动从文件名提取学号，逐个批改并汇总结果。
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail=f"任务 '{task_id}' 不存在，请先创建作业规则。")

    task_info = tasks_db[task_id]
    results = []
    errors = []

    for file in files:
        filename = file.filename or "unknown"
        student_id = extract_student_id_from_filename(filename)

        try:
            file_bytes = await file.read()

            if filename.lower().endswith(".pdf"):
                # PDF：逐页处理（线程池）
                result = await asyncio.to_thread(_process_pdf_pages, task_info, file_bytes, student_id)
            else:
                # 图片：直接处理（线程池）
                image_b64 = base64.b64encode(file_bytes).decode("utf-8")
                result = await asyncio.to_thread(_process_single_student, task_info, image_b64, student_id)

            results.append(result)

        except Exception as e:
            errors.append({"filename": filename, "student_id": student_id, "error": str(e)})

    # 计算汇总统计
    scores = [r.total_score for r in results]
    summary = BatchGradingSummary(
        task_id=task_id,
        total_students=len(results) + len(errors),
        graded_students=len(results),
        failed_students=len(errors),
        average_score=round(sum(scores) / len(scores), 2) if scores else 0,
        max_score=max(scores) if scores else 0,
        min_score=min(scores) if scores else 0,
        errors=errors
    )

    return BatchGradingResponse(
        summary=summary,
        results=results
    )
