import json
import base64
import asyncio
import os
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List

from backend.models.schemas import (
    TaskCreateRequest, GradingReportResponse, GradingReportItem, RubricItem,
    BatchGradingRequest, BatchGradingResponse, BatchGradingSummary
)
from backend.services.ai_pipeline import (
    node_a_extract_steps, node_b_logic_matcher, node_c_rag_feedback,
    node_a_extract_steps_async, node_b_logic_matcher_async, node_c_rag_feedback_async
)
from backend.services.local_db import similarity_search
from backend.services.pdf_utils import pdf_to_images, extract_student_id_from_filename

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


import re

def _robust_json_loads(raw_text: str):
    """
    鲁棒的 JSON 解析器，能自动修复 LLM 返回的 JSON 中包含 LaTeX 未转义反斜杠 (\) 的问题。
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # 将所有非转义双引号的反斜杠替换为双反斜杠
        sanitized = re.sub(r'\\(?!")', r'\\\\', raw_text)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析仍然失败！\n--- 原始输入 ---\n{raw_text}\n--- 净化尝试 ---\n{sanitized}\n--- 错误详情 ---\n{e}", flush=True)
            raise e

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
    print(f"📸 收到单份图片作业批改请求 (学生: {student_id}), 开始视觉步骤提取...", flush=True)
    # 1. Node A: Visual parsing
    student_steps_raw = node_a_extract_steps(image_b64)
    print("  ✅ 视觉步骤提取完成，开始进行步骤逻辑匹配...", flush=True)
    student_steps_raw = _extract_json_from_response(student_steps_raw)

    # 2. Node B: Logic matching
    match_result_raw = node_b_logic_matcher(student_steps_raw, task_info.standard_answer, task_info.rubric)
    print("  ✅ 步骤逻辑匹配完成，开始检索本地教材库进行 RAG 反馈...", flush=True)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = _robust_json_loads(match_result_raw)

    # 3. Node C: RAG Feedback
    try:
        final_assessment = node_c_rag_feedback(grading_results, similarity_search)
    except Exception as e:
        print(f"⚠️ RAG 反馈生成失败，使用原始结果: {e}", flush=True)
        final_assessment = {
            "enhanced_results": grading_results,
            "overall": "RAG 检索不可用，请参考上方步骤级结果。"
        }

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]
    
    print(f"🎉 批改完成！最终得分: {total_score}", flush=True)

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

    print(f"📄 收到 PDF 作业批改请求 (学生: {student_id})，共 {len(page_images)} 页，开始逐页视觉提取...", flush=True)

    # 2. 逐页调用 VLM 提取步骤
    all_steps = []
    step_counter = 1
    for i, img_bytes in enumerate(page_images):
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        print(f"  🔍 正在解析第 {i+1}/{len(page_images)} 页...", flush=True)

        try:
            page_result = node_a_extract_steps(img_b64)
            page_result = _extract_json_from_response(page_result)
            page_steps = _robust_json_loads(page_result)
            # 重新编号步骤，确保连续
            for step in page_steps:
                step["step_id"] = step_counter
                step_counter += 1
            all_steps.extend(page_steps)
            print(f"    ✅ 第 {i+1} 页提取了 {len(page_steps)} 个步骤", flush=True)
        except Exception as e:
            print(f"    ⚠️ 第 {i+1} 页解析失败: {e}", flush=True)
            all_steps.append({
                "step_id": step_counter,
                "content": f"[第{i+1}页解析失败: {str(e)[:80]}]"
            })
            step_counter += 1

    if not all_steps:
        raise ValueError("所有页面均解析失败。")

    print(f"  ✅ 共提取 {len(all_steps)} 个步骤，开始逻辑匹配...", flush=True)

    # 3. 合并后的步骤送入逻辑匹配
    student_steps_json = json.dumps(all_steps, ensure_ascii=False)

    match_result_raw = node_b_logic_matcher(student_steps_json, task_info.standard_answer, task_info.rubric)
    print("  ✅ 逻辑匹配完成，开始检索教材库生成 RAG 反馈...", flush=True)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = _robust_json_loads(match_result_raw)

    # 4. RAG Feedback
    try:
        final_assessment = node_c_rag_feedback(grading_results, similarity_search)
    except Exception as e:
        print(f"⚠️ RAG 反馈生成失败，使用原始结果: {e}", flush=True)
        final_assessment = {
            "enhanced_results": grading_results,
            "overall": "RAG 检索不可用，请参考上方步骤级结果。"
        }

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]
    
    print(f"🎉 PDF 批改完成！最终得分: {total_score}", flush=True)

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


async def _process_pdf_pages_async(task_info: TaskCreateRequest, pdf_bytes: bytes, student_id: str) -> GradingReportResponse:
    """【异步并发版本】并行处理 PDF 页面识别和大模型 RAG 生成"""
    page_images = pdf_to_images(pdf_bytes, dpi=150)
    if not page_images:
        raise ValueError("PDF 文件为空或无法解析。")

    # 并发视觉步骤提取
    tasks = [node_a_extract_steps_async(base64.b64encode(img).decode("utf-8")) for img in page_images]
    pages_results = await asyncio.gather(*tasks, return_exceptions=True)

    all_steps = []
    step_counter = 1
    for i, page_result in enumerate(pages_results):
        if isinstance(page_result, Exception):
            all_steps.append({
                "step_id": step_counter,
                "content": f"[第{i+1}页解析失败: {str(page_result)[:80]}]"
            })
            step_counter += 1
        else:
            extracted_json = _extract_json_from_response(page_result)
            page_steps = _robust_json_loads(extracted_json)
            for step in page_steps:
                step["step_id"] = step_counter
                step_counter += 1
            all_steps.extend(page_steps)

    if not all_steps:
        raise ValueError("所有页面均解析失败。")

    student_steps_json = json.dumps(all_steps, ensure_ascii=False)

    # 逻辑匹配
    match_result_raw = await node_b_logic_matcher_async(student_steps_json, task_info.standard_answer, task_info.rubric)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = _robust_json_loads(match_result_raw)

    # 并发 RAG
    final_assessment = await node_c_rag_feedback_async(grading_results, similarity_search)

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

    return GradingReportResponse(
        task_id=task_info.task_id,
        student_id=student_id,
        total_score=total_score,
        details=details,
        overall_feedback=final_assessment["overall"]
    )


async def _process_single_student_async(task_info: TaskCreateRequest, image_b64: str, student_id: str) -> GradingReportResponse:
    """【异步并发版本】图片单份批改逻辑"""
    student_steps_raw = await node_a_extract_steps_async(image_b64)
    student_steps_raw = _extract_json_from_response(student_steps_raw)

    match_result_raw = await node_b_logic_matcher_async(student_steps_raw, task_info.standard_answer, task_info.rubric)
    match_result_raw = _extract_json_from_response(match_result_raw)
    grading_results = _robust_json_loads(match_result_raw)

    final_assessment = await node_c_rag_feedback_async(grading_results, similarity_search)

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

    return GradingReportResponse(
        task_id=task_info.task_id,
        student_id=student_id,
        total_score=total_score,
        details=details,
        overall_feedback=final_assessment["overall"]
    )


async def _grade_homework_generator(task_id: str, student_id: str, file_bytes: bytes, filename: str):
    """【生成器】执行异步并发作业批改，同时向前端流式返回进度状态"""
    if task_id not in tasks_db:
        yield f"ERROR: 任务 '{task_id}' 不存在，请先创建作业规则。\n"
        return

    task_info = tasks_db[task_id]

    try:
        if filename.lower().endswith(".pdf"):
            # 1. PDF 图片转换
            yield "PROGRESS: 📄 正在将 PDF 转换为图片...\n"
            page_images = await asyncio.to_thread(pdf_to_images, file_bytes, 150)
            if not page_images:
                yield "ERROR: PDF 文件为空或无法解析。\n"
                return

            # 2. 并发提取步骤
            yield f"PROGRESS: 📄 成功转换 {len(page_images)} 页图片。正在并发调度 VLM 视觉解析...\n"
            tasks = [node_a_extract_steps_async(base64.b64encode(img).decode("utf-8")) for img in page_images]
            pages_results = await asyncio.gather(*tasks, return_exceptions=True)

            all_steps = []
            step_counter = 1
            for i, page_result in enumerate(pages_results):
                if isinstance(page_result, Exception):
                    yield f"PROGRESS: ⚠️ 第 {i+1} 页大模型解析异常: {str(page_result)[:40]}...\n"
                    all_steps.append({
                        "step_id": step_counter,
                        "content": f"[第{i+1}页解析失败]"
                    })
                    step_counter += 1
                else:
                    try:
                        extracted_json = _extract_json_from_response(page_result)
                        page_steps = _robust_json_loads(extracted_json)
                        for step in page_steps:
                            step["step_id"] = step_counter
                            step_counter += 1
                        all_steps.extend(page_steps)
                        yield f"PROGRESS: ✅ 第 {i+1} 页完成步骤识别 (提取到 {len(page_steps)} 步)\n"
                    except Exception as e:
                        yield f"PROGRESS: ⚠️ 第 {i+1} 页步骤解析失败，格式非 JSON\n"
                        all_steps.append({
                            "step_id": step_counter,
                            "content": f"[第{i+1}页数据格式化失败]"
                        })
                        step_counter += 1

            if not all_steps:
                yield "ERROR: PDF 所有页面视觉提取均告失败，批改中止。\n"
                return

            # 3. 步骤逻辑比对
            yield "PROGRESS: ⚖️ 开始由 AI 逻辑裁判进行标准答案与得分点对照...\n"
            student_steps_json = json.dumps(all_steps, ensure_ascii=False)
            match_result_raw = await node_b_logic_matcher_async(student_steps_json, task_info.standard_answer, task_info.rubric)
            match_result_raw = _extract_json_from_response(match_result_raw)
            grading_results = _robust_json_loads(match_result_raw)

            # 4. 教材匹配 RAG
            yield "PROGRESS: 📚 逻辑判分完毕，正在并发检索教材向量库生成知识引伸...\n"
            final_assessment = await node_c_rag_feedback_async(grading_results, similarity_search)

            total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
            details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

            result_obj = GradingReportResponse(
                task_id=task_info.task_id,
                student_id=student_id,
                total_score=total_score,
                details=details,
                overall_feedback=final_assessment["overall"]
            )
            yield f"RESULT: {result_obj.model_dump_json(by_alias=True)}\n"

        else:
            # 单张图片处理
            yield "PROGRESS: 📸 正在加载图片并初始化视觉识别任务...\n"
            image_b64 = base64.b64encode(file_bytes).decode("utf-8")

            yield "PROGRESS: 🔍 正在调度 VLM 进行手写步骤提取与 LaTeX 公式转化...\n"
            student_steps_raw = await node_a_extract_steps_async(image_b64)
            student_steps_raw = _extract_json_from_response(student_steps_raw)

            yield "PROGRESS: ⚖️ 步骤识别成功。开始由 AI 逻辑裁判比对标准答案与得分点...\n"
            match_result_raw = await node_b_logic_matcher_async(student_steps_raw, task_info.standard_answer, task_info.rubric)
            match_result_raw = _extract_json_from_response(match_result_raw)
            grading_results = _robust_json_loads(match_result_raw)

            yield "PROGRESS: 📚 逻辑比对完成，正在检索本地自控教材向量库生成知识引伸...\n"
            final_assessment = await node_c_rag_feedback_async(grading_results, similarity_search)

            total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])
            details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

            result_obj = GradingReportResponse(
                task_id=task_info.task_id,
                student_id=student_id,
                total_score=total_score,
                details=details,
                overall_feedback=final_assessment["overall"]
            )
            yield f"RESULT: {result_obj.model_dump_json(by_alias=True)}\n"

    except Exception as e:
        yield f"ERROR: 批改引擎异常: {str(e)}\n"


@router.post("/grade_homework")
async def grade_homework(
    task_id: str = Form(...),
    student_id: str = Form(...),
    image: UploadFile = File(...)
):
    """【流式接口】支持并发批改，并以 SSE 形式将进度实时返回给前端"""
    file_bytes = await image.read()
    filename = image.filename or ""

    return StreamingResponse(
        _grade_homework_generator(task_id, student_id, file_bytes, filename),
        media_type="text/event-stream"
    )


@router.post("/grade_homework_batch")
async def grade_homework_batch(
    task_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    【加速版本】批量批改：一次上传多个学生作业文件（图片或 PDF）。
    使用内部的异步并发流程加速处理。
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
                # 并发 PDF 处理
                result = await _process_pdf_pages_async(task_info, file_bytes, student_id)
            else:
                image_b64 = base64.b64encode(file_bytes).decode("utf-8")
                result = await _process_single_student_async(task_info, image_b64, student_id)

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
