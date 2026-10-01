"""One fail-closed grading workflow shared by single and batch submissions."""
import asyncio
import base64
from decimal import Decimal
from functools import partial
import io
import json
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import ValidationError

from backend.models.schemas import ExtractedStep, GradingReportItem, GradingReportResponse
from backend.services.ai_pipeline import (
    node_a_extract_steps_async, node_b_logic_matcher_async, node_c_rag_feedback_async,
)
from backend.services.local_db import similarity_search
from backend.services.pdf_utils import pdf_to_images
from backend.services.provider_errors import provider_error_hint

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PAGES = 20
MAX_IMAGE_PIXELS = 20_000_000
PAGE_CONCURRENCY = 3


class ModelJSONError(ValueError):
    """JSON that cannot be interpreted without discarding model evidence."""


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelJSONError(f"模型 JSON 存在重复字段 {key[:80]}，无法确定可信值。")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ModelJSONError("模型 JSON 含 NaN 或 Infinity 等非标准数值。")


def parse_model_json(raw):
    """Do not silently repair LaTeX escapes: a repair can change mathematical meaning."""
    if not isinstance(raw, str):
        raise ValueError("模型输出不是文本")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("模型输出格式不完整")
        text = "\n".join(lines[1:-1])
    # Default json.loads silently keeps the final duplicate key and accepts
    # NaN/Infinity. Both can conceal an earlier uncertainty flag or evidence.
    return json.loads(text, object_pairs_hook=_unique_json_object,
                      parse_constant=_reject_json_constant)


def _prepare_pages(file_bytes, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        images = pdf_to_images(file_bytes, dpi=150)
        if not images or len(images) > MAX_PAGES:
            raise ValueError(f"PDF 必须包含 1 至 {MAX_PAGES} 页")
        return images
    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("仅支持 JPG、PNG 和 PDF")
    with Image.open(io.BytesIO(file_bytes)) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("图片内容与支持的格式不符")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("图片像素过大，请缩小后重试")
        image = ImageOps.exif_transpose(image)
        # Preserve readability for transparent PNGs when converting to JPEG.
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        output = io.BytesIO()
        background.convert("RGB").save(output, format="JPEG", quality=95)
        return [output.getvalue()]


def _review(report, reasons):
    report.status = "needs_review"
    report.total_score = None
    report.review_reasons = list(dict.fromkeys(reasons))
    report.overall_feedback = "资料或模型判断存在不确定性，暂不出分，请教师核对识别内容与评分依据。"
    return report


def _parse_page(raw, page_number, offset):
    payload = parse_model_json(raw)
    reasons = []
    if isinstance(payload, dict):
        if not {"steps", "needs_review", "teacher_annotations_present", "review_reasons"} <= payload.keys():
            reasons.append(f"第 {page_number} 页缺少完整识别质量状态，请教师复核。")
        for flag in ("needs_review", "teacher_annotations_present"):
            if flag in payload and not isinstance(payload[flag], bool):
                raise ValueError("识别状态格式不正确")
        notes = payload.get("review_reasons", [])
        if not isinstance(notes, list) or any(not isinstance(s, str) for s in notes):
            raise ValueError("识别复核理由格式不正确")
        if payload.get("needs_review") is False:
            if payload.get("teacher_annotations_present") is True:
                reasons.append(f"第 {page_number} 页识别质量状态矛盾：检测到教师批注，却标记为无需复核。")
            if notes:
                reasons.append(f"第 {page_number} 页识别质量状态矛盾：给出了复核理由，却标记为无需复核。")
        if payload.get("needs_review") or notes:
            reasons.append(f"第 {page_number} 页识别需复核：" + "；".join(notes or ["存在无法确认的内容"]))
        if payload.get("teacher_annotations_present"):
            reasons.append(f"第 {page_number} 页含教师批注或评分，请先区分学生原文，避免评分泄漏。")
        payload = payload.get("steps")
    elif isinstance(payload, list):
        reasons.append(f"第 {page_number} 页使用旧识别格式，缺少质量状态，请教师复核。")
    if not isinstance(payload, list) or not payload:
        raise ValueError("未提取到学生作答步骤")
    steps = []
    local_ids = set()
    for item in payload:
        step = ExtractedStep.model_validate(item)
        if not step.content.strip() or step.step_id in local_ids:
            raise ValueError("识别步骤为空或编号重复")
        if "[无法辨认]" in step.content:
            reasons.append(f"第 {page_number} 页有无法辨认的内容，请核对原作业。")
        local_ids.add(step.step_id)
        steps.append(ExtractedStep(step_id=offset + len(steps) + 1,
                                   content=step.content, page_number=page_number))
    return steps, reasons


def validate_grades(raw, rubric, extracted_steps):
    payload = parse_model_json(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError("模型未返回完整评分列表")
    required = {"step_id", "is_correct", "points_awarded", "student_step_description",
                "student_step_ids", "feedback", "needs_review", "review_reason"}
    if any(not isinstance(item, dict) or not required <= item.keys() for item in payload):
        raise ValueError("模型评分缺少判分依据或复核状态")
    details = [GradingReportItem.model_validate(item) for item in payload]
    expected = {item.step_id: item for item in rubric}
    actual = [item.step_id for item in details]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("评分项有重复、遗漏或未知编号")
    evidence_ids = {step.step_id for step in extracted_steps}
    for item in details:
        maximum = expected[item.step_id].points
        if not item.student_step_description.strip() or not (item.feedback or "").strip():
            raise ValueError(f"评分项 {item.step_id} 缺少具体判分依据")
        if item.points_awarded > maximum:
            raise ValueError(f"评分项 {item.step_id} 超出该项满分")
        if item.needs_review or item.review_reason:
            raise ValueError(item.review_reason or f"评分项 {item.step_id} 需教师复核")
        if any(type(n) is not int or n not in evidence_ids for n in item.student_step_ids):
            raise ValueError(f"评分项 {item.step_id} 引用了不存在的作答步骤")
        if len(item.student_step_ids) != len(set(item.student_step_ids)):
            raise ValueError("学生作答证据编号重复")
        if item.points_awarded > 0 and not item.student_step_ids:
            raise ValueError(f"评分项 {item.step_id} 得分缺少学生作答证据")
        if item.is_correct != (item.points_awarded == maximum):
            raise ValueError(f"评分项 {item.step_id} 的对错判断与得分矛盾")
    return sorted(details, key=lambda item: list(expected).index(item.step_id))


async def grade_submission(task, file_bytes: bytes, filename: str, student_id: str):
    report = GradingReportResponse(
        task_id=task.task_id, student_id=student_id,
        max_score=float(sum(Decimal(str(item.points)) for item in task.rubric)),
        rubric=[item.model_copy(deep=True) for item in task.rubric],
        question_text=task.question_text, standard_answer=task.standard_answer,
    )
    if not task.question_text.strip():
        return _review(report, ["任务缺少具体题干，请补齐题目条件后重新批改。"])
    if not file_bytes or len(file_bytes) > MAX_FILE_BYTES:
        raise ValueError("文件不能为空且不能超过 20 MB")
    try:
        pages = await asyncio.to_thread(_prepare_pages, file_bytes, filename)
    except Exception:
        return _review(report, [f"文件无法读取，请检查图片或 PDF（最多 {MAX_PAGES} 页）后重试。"])

    semaphore = asyncio.Semaphore(PAGE_CONCURRENCY)

    async def extract(image):
        async with semaphore:
            return await node_a_extract_steps_async(
                base64.b64encode(image).decode("ascii"), mime_type="image/jpeg")

    pages_raw = await asyncio.gather(*(extract(page) for page in pages), return_exceptions=True)
    reasons = []
    for number, raw in enumerate(pages_raw, 1):
        if isinstance(raw, Exception):
            hint = provider_error_hint(raw)
            reasons.append(f"第 {number} 页识别服务失败。{hint}" if hint else
                           f"第 {number} 页识别服务失败，请检查模型配置或稍后重试。")
            continue
        try:
            steps, page_reasons = _parse_page(raw, number, len(report.extracted_steps))
            report.extracted_steps.extend(steps)
            reasons.extend(page_reasons)
        except ModelJSONError as exc:
            reasons.append(f"第 {number} 页识别结果无效：{exc}不能据此扣分。")
        except Exception:
            reasons.append(f"第 {number} 页识别为空或格式无效，不能据此扣分。")
    if reasons or not report.extracted_steps:
        return _review(report, reasons or ["未获得可用于判分的作答。"])

    student_json = json.dumps([item.model_dump() for item in report.extracted_steps], ensure_ascii=False)
    try:
        raw = await node_b_logic_matcher_async(
            student_json, task.standard_answer, task.rubric, question_text=task.question_text)
    except Exception as exc:
        hint = provider_error_hint(exc)
        return _review(report, [f"评分服务暂时不可用。{hint}" if hint else
                                "评分服务暂时不可用，请检查模型配置或稍后重试。"])
    try:
        details = validate_grades(raw, task.rubric, report.extracted_steps)
    except (ValidationError, json.JSONDecodeError):
        return _review(report, ["评分结果格式无效或字段类型不正确，请教师核对。"])
    except ValueError as exc:
        return _review(report, [str(exc)[:600]])
    except Exception:
        return _review(report, ["评分结果无法校验，请教师核对。"])

    total = float(sum(Decimal(str(item.points_awarded)) for item in details))
    report.details = details
    report.total_score = total
    report.ai_total_score = total
    report.overall_feedback = "AI 建议分已生成，请教师结合原作业确认。"
    try:
        assessment = await node_c_rag_feedback_async(
            [item.model_dump() for item in details], partial(similarity_search, top_k=6),
            question_text=task.question_text, rubric=[item.model_dump() for item in task.rubric],
            standard_answer=task.standard_answer)
        # RAG may annotate evidence and feedback, but it cannot change any grade.
        annotations = {item["step_id"]: item for item in assessment["enhanced_results"]}
        for detail in report.details:
            note = annotations.get(detail.step_id, {})
            detail.rag_knowledge = note.get("rag_knowledge")
            detail.rag_sources = note.get("rag_sources", [])
        report.warnings.extend(assessment.get("warnings", []))
        if assessment.get("overall"):
            report.overall_feedback = assessment["overall"]
    except Exception:
        report.warnings.append("教材检索或讲解暂时不可用；建议分已保留，未生成无来源的教材解释。")
    report.ai_details = [item.model_copy(deep=True) for item in report.details]
    return report
