import json
import base64
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List

from backend.models.schemas import TaskCreateRequest, GradingReportResponse, GradingReportItem, RubricItem
from backend.services.ai_pipeline import node_a_extract_steps, node_b_logic_matcher, node_c_rag_feedback
from backend.services.supabase_db import similarity_search
from langchain_openai import OpenAIEmbeddings

router = APIRouter()
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

tasks_db = {} # MVP: memory dictionary to store standard tasks

@router.post("/upload_task")
async def upload_task(task: TaskCreateRequest):
    """教师上传本次作业的题干、标准答案及得分细则 (Rubric)"""
    tasks_db[task.task_id] = task
    return {"message": "Task created successfully", "task_id": task.task_id}

@router.post("/grade_homework", response_model=GradingReportResponse)
async def grade_homework(
    task_id: str = Form(...),
    student_id: str = Form(...),
    image: UploadFile = File(...)
):
    """批量或单张导入学生提交的答卷图片，核心打分流"""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
        
    task_info = tasks_db[task_id]
    
    # 1. Image preparation
    image_bytes = await image.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # 2. Node A: Visual parsing
    try:
        student_steps_raw = node_a_extract_steps(image_b64)
        # Handle markdown JSON extraction
        if "```json" in student_steps_raw:
            student_steps_raw = student_steps_raw.split("```json")[1].split("```")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM Extraction failed: {str(e)}")

    # 3. Node B: Logic matching
    try:
        match_result_raw = node_b_logic_matcher(student_steps_raw, task_info.standard_answer, task_info.rubric)
        if "```json" in match_result_raw:
            match_result_raw = match_result_raw.split("```json")[1].split("```")[0]
        grading_results = json.loads(match_result_raw)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Logic Matching failed: {str(e)}")

    # 4. Node C: RAG Feedback
    def embed_func(text: str) -> List[float]:
        return embed_model.embed_query(text)
        
    try:
        final_assessment = node_c_rag_feedback(grading_results, similarity_search, embed_func)
    except Exception as e:
        # Fallback if DB doesn't work, don't crash MVP entirely
        final_assessment = {
            "enhanced_results": grading_results,
            "overall": "RAG Reterival unavailable, review node B results."
        }

    total_score = sum(step.get("points_awarded", 0) for step in final_assessment["enhanced_results"])

    details = [GradingReportItem(**step) for step in final_assessment["enhanced_results"]]

    return GradingReportResponse(
        task_id=task_id,
        student_id=student_id,
        total_score=total_score,
        details=details,
        overall_feedback=final_assessment["overall"]
    )
