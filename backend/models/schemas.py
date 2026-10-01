"""Validated contracts for suggested grades and teacher review."""
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, StrictBool, StrictInt, field_validator, model_validator


class RubricItem(BaseModel):
    step_id: int = Field(gt=0, strict=True)
    description: str = Field(min_length=1)
    points: float = Field(gt=0, allow_inf_nan=False, strict=True)

    @field_validator("description")
    @classmethod
    def meaningful_description(cls, value):
        if not value.strip():
            raise ValueError("评分项描述不能为空")
        return value.strip()


class TaskCreateRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=500)
    question_text: str = ""  # Legacy tasks remain readable but cannot be graded.
    standard_answer: str = Field(min_length=1)
    rubric: List[RubricItem] = Field(min_length=1, max_length=100)

    @field_validator("task_id", "title", "standard_answer")
    @classmethod
    def meaningful_text(cls, value):
        if not value.strip():
            raise ValueError("必填内容不能为空")
        return value.strip()

    @model_validator(mode="after")
    def unique_rubric(self):
        ids = [item.step_id for item in self.rubric]
        if len(ids) != len(set(ids)):
            raise ValueError("评分项编号不能重复")
        if not math.isfinite(sum(item.points for item in self.rubric)):
            raise ValueError("评分总分必须为有限数值")
        return self


class ExtractedStep(BaseModel):
    step_id: int = Field(gt=0, strict=True)
    content: str = Field(min_length=1)
    page_number: int = Field(default=1, gt=0)

    @field_validator("content")
    @classmethod
    def valid_transcription(cls, value):
        # A single JSON backslash before frac/beta/theta/right/nu is a valid
        # escape but silently destroys a formula. Keep each OCR step one line.
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("识别文本为空或含有可疑转义字符")
        return value


class GradingReportItem(BaseModel):
    step_id: int = Field(gt=0, strict=True)
    is_correct: StrictBool
    points_awarded: float = Field(ge=0, allow_inf_nan=False, strict=True)
    student_step_description: str
    student_step_ids: List[StrictInt] = Field(default_factory=list)
    error_type: Optional[str] = None
    feedback: Optional[str] = None
    rag_knowledge: Optional[str] = None
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)
    needs_review: StrictBool = False
    review_reason: Optional[str] = None


class ReviewScore(BaseModel):
    step_id: int = Field(gt=0, strict=True)
    points_awarded: float = Field(ge=0, allow_inf_nan=False, strict=True)
    feedback: str = ""


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=4000)
    scores: List[ReviewScore] = Field(min_length=1)

    @field_validator("reviewer", "reason")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("复核人和复核说明不能为空")
        return value.strip()


class ReviewRecord(ReviewRequest):
    reviewed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    previous_score: Optional[float] = None
    previous_status: str


class GradingReportResponse(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_id: str
    student_id: Optional[str] = None
    filename: str = ""
    source_available: bool = False
    source_sha256: Optional[str] = None
    status: Literal["graded", "needs_review", "reviewed"] = "graded"
    total_score: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    max_score: float = Field(default=0, ge=0, allow_inf_nan=False)
    details: List[GradingReportItem] = Field(default_factory=list)
    overall_feedback: str = ""
    extracted_steps: List[ExtractedStep] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    rubric: List[RubricItem] = Field(default_factory=list)
    question_text: str = ""
    standard_answer: str = ""
    ai_total_score: Optional[float] = None
    ai_details: List[GradingReportItem] = Field(default_factory=list)
    review_history: List[ReviewRecord] = Field(default_factory=list)


class BatchGradingRequest(BaseModel):
    task_id: str
    student_ids: List[str]


class BatchGradingSummary(BaseModel):
    task_id: str
    total_students: int
    graded_students: int
    needs_review_students: int = 0
    failed_students: int
    average_score: Optional[float] = None
    max_score: Optional[float] = None
    min_score: Optional[float] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class BatchGradingResponse(BaseModel):
    summary: BatchGradingSummary
    results: List[GradingReportResponse]
