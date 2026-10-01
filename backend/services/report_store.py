"""Local report persistence; all generated data lives under ignored output/."""
import json
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from uuid import UUID

from backend.models.schemas import GradingReportItem, GradingReportResponse, ReviewRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
_lock = threading.RLock()


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _report_path(report_id):
    # A public report id must never become an arbitrary filesystem path.
    return Path(REPORTS_DIR) / f"{UUID(str(report_id))}.json"


def save_report(report):
    with _lock:
        atomic_write_json(_report_path(report.report_id), report.model_dump(mode="json"))


def save_submission(report, file_bytes, filename):
    """Keep the exact input alongside its report for later teacher verification."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise ValueError("不支持的原作业格式")
    with _lock:
        folder = _report_path(report.report_id).with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        source = folder / ("source" + suffix)
        with source.open("xb") as stream:
            stream.write(file_bytes)
        report.filename = name
        report.source_available = True
        report.source_sha256 = hashlib.sha256(file_bytes).hexdigest()
        save_report(report)


def get_source(report_id):
    report = get_report(report_id)
    suffix = Path(report.filename).suffix.lower()
    if not report.source_available or suffix not in {".jpg", ".jpeg", ".png", ".pdf"}:
        raise FileNotFoundError("原作业未保存")
    source = _report_path(report_id).with_suffix("") / ("source" + suffix)
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != report.source_sha256:
        raise FileNotFoundError("原作业缺失或完整性校验失败")
    return source, report.filename


def get_report(report_id):
    path = _report_path(report_id)
    return GradingReportResponse.model_validate_json(path.read_text(encoding="utf-8"))


def list_reports(limit=None):
    paths = sorted(Path(REPORTS_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    reports = []
    for path in paths[:limit]:
        try:
            report = GradingReportResponse.model_validate_json(path.read_text(encoding="utf-8"))
            reports.append({
                "report_id": report.report_id, "created_at": report.created_at,
                "task_id": report.task_id, "student_id": report.student_id,
                "filename": report.filename, "source_available": report.source_available,
                "status": report.status, "total_score": report.total_score,
                "max_score": report.max_score,
            })
        except (ValueError, OSError):
            # Surface an unreadable local record instead of silently omitting it.
            reports.append({"report_id": path.stem, "status": "unreadable",
                            "error": "本地报告文件损坏或无法读取"})
    return reports


def review_report(report_id, request):
    from decimal import Decimal

    with _lock:
        report = get_report(report_id)
        rubric = {item.step_id: item for item in report.rubric}
        ids = [item.step_id for item in request.scores]
        if len(ids) != len(set(ids)) or set(ids) != set(rubric):
            raise ValueError("复核必须为全部评分项各提供一次得分")
        for item in request.scores:
            if item.points_awarded > rubric[item.step_id].points:
                raise ValueError(f"评分项 {item.step_id} 超过该项满分")
        previous = {item.step_id: item for item in report.details}
        report.review_history.append(ReviewRecord(
            **request.model_dump(), previous_score=report.total_score, previous_status=report.status))
        scores = {item.step_id: item for item in request.scores}
        report.details = []
        for item in report.rubric:
            score = scores[item.step_id]
            old = previous.get(item.step_id)
            report.details.append(GradingReportItem(
                step_id=item.step_id, is_correct=score.points_awarded == item.points,
                points_awarded=score.points_awarded,
                student_step_description=old.student_step_description if old else "教师依据原作业复核",
                student_step_ids=old.student_step_ids if old else [],
                feedback=score.feedback or request.reason,
            ))
        report.total_score = float(sum(Decimal(str(item.points_awarded)) for item in request.scores))
        report.status = "reviewed"
        report.overall_feedback = f"教师复核完成：{request.reason}"
        # Original AI details and review reasons remain available as the audit trail.
        save_report(report)
        return report
