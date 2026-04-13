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
