from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RubricItem(BaseModel):
    step_id: int
    description: str
    points: float


class TaskCreateRequest(BaseModel):
    task_id: str
    title: str
    standard_answer: str
    rubric: List[RubricItem]


class GradingReportItem(BaseModel):
    step_id: int
    is_correct: bool
    points_awarded: float
    student_step_description: str
    error_type: Optional[str] = None
    feedback: Optional[str] = None
    rag_knowledge: Optional[str] = None


class GradingReportResponse(BaseModel):
    task_id: str
    student_id: Optional[str] = None
    total_score: float
    details: List[GradingReportItem]
    overall_feedback: str


class BatchGradingRequest(BaseModel):
    """批量批改请求（非文件上传方式，保留扩展性）"""
    task_id: str
    student_ids: List[str]


class BatchGradingSummary(BaseModel):
    """批量批改汇总统计"""
    task_id: str
    total_students: int
    graded_students: int
    failed_students: int
    average_score: float
    max_score: float
    min_score: float
    errors: List[Dict[str, Any]] = []


class BatchGradingResponse(BaseModel):
    """批量批改响应"""
    summary: BatchGradingSummary
    results: List[GradingReportResponse]
