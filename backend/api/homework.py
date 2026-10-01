"""Local homework API with shared grading, recoverable reports, and teacher review."""
import asyncio
import json
import os
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from backend.models.schemas import (
    BatchGradingResponse, BatchGradingSummary, GradingReportResponse,
    ReviewRequest, TaskCreateRequest,
)
from backend.services.grading import MAX_FILE_BYTES, grade_submission
from backend.services.pdf_utils import extract_student_id_from_filename
from backend.services import report_store

router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKS_FILE = Path(os.getenv("HOMEWORK_TASKS_PATH", "").strip() or "output/tasks.json").expanduser()
if not TASKS_FILE.is_absolute():
    TASKS_FILE = PROJECT_ROOT / TASKS_FILE


def load_tasks():
    if not TASKS_FILE.exists():
        return {}
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return {tid: TaskCreateRequest.model_validate(value) for tid, value in data.items()}


tasks_db = load_tasks()


def save_tasks(tasks=None):
    current = tasks_db if tasks is None else tasks
    report_store.atomic_write_json(TASKS_FILE, {
        key: value.model_dump(mode="json") for key, value in current.items()
    })


def _get_task(task_id):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="任务不存在，请先创建作业规则。")
    return tasks_db[task_id]


async def _read_upload(file):
    if Path(file.filename or "").suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise HTTPException(status_code=422, detail="仅支持 JPG、PNG 和 PDF。")
    data = await file.read(MAX_FILE_BYTES + 1)
    if not data or len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="文件不能为空且不能超过 20 MB。")
    return data


@router.post("/upload_task")
async def upload_task(task: TaskCreateRequest):
    if not task.question_text.strip():
        raise HTTPException(status_code=422, detail="请填写具体题干、参数与已知条件。")
    updated = dict(tasks_db)
    updated[task.task_id] = task
    try:
        save_tasks(updated)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="任务未保存，请检查本地存储权限。") from exc
    tasks_db[task.task_id] = task
    return {"message": "Task created successfully", "task_id": task.task_id}


@router.get("/tasks")
async def list_tasks():
    return {"tasks": [{
        "task_id": task.task_id, "title": task.title, "rubric_count": len(task.rubric),
        "total_points": sum(item.points for item in task.rubric),
        "ready_for_grading": bool(task.question_text.strip()),
    } for task in tasks_db.values()]}


@router.get("/tasks/{task_id}", response_model=TaskCreateRequest)
async def get_task(task_id: str):
    return _get_task(task_id)


async def _grade_homework_generator(task_id, student_id, file_bytes, filename):
    task = _get_task(task_id)
    yield "PROGRESS: 正在核对题干、识别作答并校验评分依据；不确定的结果将转为待复核。\n"
    try:
        report = await grade_submission(task, file_bytes, filename, student_id)
        await asyncio.to_thread(report_store.save_submission, report, file_bytes, filename)
        yield f"RESULT: {report.model_dump_json()}\n"
    except ValueError:
        yield "ERROR: 文件或批改数据无效，请检查后重试。\n"
    except Exception:
        yield "ERROR: 批改或报告保存失败，请检查服务配置和本地存储后重试。\n"


@router.post("/grade_homework")
async def grade_homework(
    task_id: str = Form(...), student_id: str = Form(...), image: UploadFile = File(...)
):
    _get_task(task_id)
    data = await _read_upload(image)
    return StreamingResponse(
        _grade_homework_generator(task_id, student_id, data, image.filename or ""),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/grade_homework_batch", response_model=BatchGradingResponse)
async def grade_homework_batch(task_id: str = Form(...), files: list[UploadFile] = File(...)):
    task = _get_task(task_id)
    if not files or len(files) > 100:
        raise HTTPException(status_code=422, detail="每批请上传 1 至 100 份文件。")
    results, errors = [], []
    # Files are processed in order; the grading workflow bounds page concurrency.
    for file in files:
        filename = file.filename or "unknown"
        student_id = extract_student_id_from_filename(filename)
        try:
            data = await _read_upload(file)
            report = await grade_submission(task, data, filename, student_id)
            await asyncio.to_thread(report_store.save_submission, report, data, filename)
            results.append(report)
        except HTTPException as exc:
            errors.append({"filename": filename, "student_id": student_id, "error": exc.detail})
        except Exception:
            errors.append({"filename": filename, "student_id": student_id,
                           "error": "批改或报告保存失败，请检查服务和文件后重试。"})
    scores = [r.total_score for r in results if r.status in {"graded", "reviewed"}
              and r.total_score is not None]
    summary = BatchGradingSummary(
        task_id=task_id, total_students=len(files), graded_students=len(scores),
        needs_review_students=sum(r.status == "needs_review" for r in results),
        failed_students=len(errors),
        average_score=round(sum(scores) / len(scores), 2) if scores else None,
        max_score=max(scores) if scores else None, min_score=min(scores) if scores else None,
        errors=errors,
    )
    return BatchGradingResponse(summary=summary, results=results)


@router.get("/reports")
async def reports():
    return {"reports": await asyncio.to_thread(report_store.list_reports)}


@router.get("/reports/{report_id}", response_model=GradingReportResponse)
async def report_detail(report_id: str):
    try:
        return await asyncio.to_thread(report_store.get_report, report_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="报告不存在或无法读取。") from exc


@router.post("/reports/{report_id}/review", response_model=GradingReportResponse)
async def review(report_id: str, request: ReviewRequest):
    try:
        return await asyncio.to_thread(report_store.review_report, report_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="报告不存在。") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="复核结果保存失败，请重试。") from exc


@router.get("/reports/{report_id}/source")
async def report_source(report_id: str):
    try:
        path, filename = await asyncio.to_thread(report_store.get_source, report_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="原作业不存在或完整性校验失败。") from exc
    return FileResponse(path, filename=filename)
