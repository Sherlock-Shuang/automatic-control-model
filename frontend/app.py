"""Teacher workspace for task setup, accountable grading, and local report review."""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
import requests
import streamlit as st

from frontend.theme import apply_theme, empty_state, note_card, render_hero, render_sidebar, section_heading

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="Control Lab · 教师工作台", page_icon="◉", layout="wide")
apply_theme()
render_sidebar()
render_hero()

API_BASE = (os.getenv("HOMEWORK_API_BASE_URL", "").strip() or "http://localhost:8000").rstrip("/") + "/api"
REQUEST_TIMEOUT = (5, 30)
GRADING_TIMEOUT = (10, 300)
BATCH_TIMEOUT = (10, 1800)

DEFAULT_RUBRIC = [
    {"step_id": 1, "description": "", "points": 0.0},
]
DEFAULT_EDITOR = {
    "editor_task_id": "",
    "editor_title": "",
    "editor_question": "",
    "editor_answer": "",
    "editor_rubric": json.dumps(DEFAULT_RUBRIC, ensure_ascii=False, indent=2),
}
for name, value in DEFAULT_EDITOR.items():
    st.session_state.setdefault(name, value)
st.session_state.setdefault("updated_reports", {})
st.session_state.setdefault("source_files", {})
st.session_state.setdefault("rubric_editor_version", 0)


def edit_rubric_rows(action):
    """Update form structure before rendering, preserving every submitted field."""
    version = st.session_state.rubric_editor_version
    rubric = []
    for index, item in enumerate(json.loads(st.session_state.editor_rubric)):
        row_key = f"rubric_{version}_{index}"
        if action == "remove" and st.session_state.get(f"{row_key}_remove", False):
            continue
        rubric.append({
            "step_id": item["step_id"],
            "description": st.session_state.get(f"{row_key}_description", item.get("description", "")),
            "points": st.session_state.get(f"{row_key}_points", item.get("points", 0)),
        })
    if action == "add":
        next_id = max((item["step_id"] for item in rubric), default=0) + 1
        rubric.append({"step_id": next_id, "description": "", "points": 1.0})
    st.session_state.editor_rubric = json.dumps(rubric, ensure_ascii=False)
    st.session_state.rubric_editor_version += 1


def api_json(method, path, **kwargs):
    response = requests.request(method, f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
    if not response.ok:
        raise ValueError(f"请求失败（{response.status_code}）：{response.text}")
    return response.json()


def scored(report):
    value = report.get("total_score")
    return (
        report.get("status") in {"graded", "reviewed"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def score_label(report):
    if report.get("status") == "unreadable":
        return "文件损坏 / 无法读取"
    if not scored(report):
        return "待复核 / 未出分"
    prefix = "教师已复核" if report.get("status") == "reviewed" else "建议得分"
    return f"{prefix}：{report['total_score']:g} / {report.get('max_score', 0):g} 分"


def latest_report(report):
    return st.session_state.updated_reports.get(report.get("report_id"), report)


def review_time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M（北京时间）")
    except (AttributeError, TypeError, ValueError):
        return "时间未记录"


def render_source(report, key):
    """Fetch the retained original only after the teacher requests it."""
    report_id = report.get("report_id")
    if not report_id or not report.get("source_available"):
        st.caption("本报告未保存原始文件，请对照本地原始答卷进行复核。")
        return
    if st.button("加载原始答卷", key=f"{key}_load_source"):
        try:
            response = requests.request(
                "GET", f"{API_BASE}/reports/{quote(str(report_id), safe='')}/source", timeout=REQUEST_TIMEOUT,
            )
            if not response.ok:
                raise ValueError(f"请求失败（{response.status_code}）：{response.text}")
            mime_type = response.headers.get("Content-Type", "application/octet-stream").split(";")[0]
            if mime_type not in {"image/jpeg", "image/png", "application/pdf"}:
                raise ValueError("原始文件格式不可预览，请检查报告存储。")
            st.session_state.source_files[report_id] = {"data": response.content, "type": mime_type}
        except (requests.RequestException, ValueError) as exc:
            st.error(f"原始答卷加载失败：{exc}")
    source = st.session_state.source_files.get(report_id)
    if source:
        if source["type"].startswith("image/"):
            try:
                st.image(source["data"], caption="原始答卷，请与下方识别内容对照", use_column_width=True)
            except (OSError, ValueError):
                st.warning("原始图片暂不能预览，可下载后核对。")
        else:
            st.caption("PDF 原始答卷可下载后逐页对照。")
        st.download_button(
            "下载原始答卷", source["data"], file_name=report.get("filename") or "original",
            mime=source["type"], key=f"{key}_source_download",
        )


def render_review(report, key):
    rubric = report.get("rubric", [])
    report_id = report.get("report_id")
    st.markdown("#### 教师复核")
    if not rubric or not report_id:
        st.info("此报告没有保存完整评分细则，暂不能在这里复核。请重新创建批改报告。")
        return
    st.caption("请先核对原始答卷、识别原文和题目。未提供建议分的评分点初始为 0，请逐项确认。")
    if report.get("status") == "reviewed":
        st.caption("再次提交将保留新的复核记录。")
    details = {item["step_id"]: item for item in report.get("details", [])}
    with st.form(f"{key}_review_form"):
        reviewer = st.text_input("复核教师", key=f"{key}_reviewer")
        reason = st.text_area("复核说明（必填）", key=f"{key}_review_reason")
        scores = []
        for item in rubric:
            step_id = item["step_id"]
            maximum = float(item["points"])
            current = details.get(step_id, {})
            value = current.get("points_awarded", 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                value = 0
            st.markdown(f"**评分点 {step_id}**")
            st.markdown(item["description"])
            awarded = st.number_input(
                f"评分点 {step_id} 得分（满分 {maximum:g}）",
                min_value=0.0, max_value=maximum, value=min(max(float(value), 0), maximum),
                step=0.5, key=f"{key}_points_{step_id}",
            )
            feedback = st.text_area(
                f"评分点 {step_id} 复核评语", value=current.get("feedback") or "",
                key=f"{key}_feedback_{step_id}", height=68,
            )
            scores.append({"step_id": step_id, "points_awarded": awarded, "feedback": feedback})
        submitted = st.form_submit_button("确认复核并保存成绩", use_container_width=True)
    if submitted:
        if not reviewer.strip() or not reason.strip():
            st.error("请填写复核教师和复核说明后再提交。")
            return
        try:
            updated = api_json("POST", f"/reports/{quote(str(report_id), safe='')}/review", json={
                "reviewer": reviewer.strip(), "reason": reason.strip(), "scores": scores,
            })
            st.session_state.updated_reports[report_id] = updated
            st.session_state["review_notice"] = "教师复核已保存，报告与统计已更新。"
            st.rerun()
        except (requests.RequestException, ValueError) as exc:
            st.error(f"复核保存失败：{exc}")


def render_details(details, rubric, show_scores):
    """Read-only rendering shared by the current assessment and AI audit trail."""
    for detail in details:
        with st.container(border=True):
            step_id = detail.get("step_id")
            st.markdown(f"**评分点 {step_id}**")
            if step_id in rubric:
                st.markdown(rubric[step_id]["description"])
            if show_scores:
                st.caption(f"得分：{detail.get('points_awarded')} 分")
            else:
                st.caption("该评分点尚待教师确认")
            ids = detail.get("student_step_ids", [])
            st.caption("对应识别步骤：" + ("、".join(map(str, ids)) if ids else "未匹配"))
            st.markdown(detail.get("student_step_description") or "未提供作答说明。")
            if detail.get("feedback"):
                st.markdown("**评阅说明**")
                st.markdown(detail["feedback"])
            if detail.get("rag_knowledge"):
                st.markdown("**教材辅助讲解**")
                st.markdown(detail["rag_knowledge"])
            for source in detail.get("rag_sources", []):
                st.markdown("**教材出处**")
                location = [str(source.get("title") or "教材")]
                if source.get("chapter"):
                    location.append(str(source["chapter"]))
                if source.get("page") is not None:
                    location.append(f"第 {source['page']} 页")
                st.caption(" · ".join(location))
                st.caption(f"来源编号：{source.get('source_id', '')}")
                st.markdown(source.get("excerpt") or "未提供摘录。")
                if source.get("image_path"):
                    st.caption("本地教材图片路径")
                    st.code(source["image_path"], language=None)


def render_report(report, key):
    """One rendering path for single, batch, and archived reports."""
    report = latest_report(report)
    if not scored(report):
        st.warning("待复核 / 未出分：请核对下面的问题，由教师确认后保存成绩。")
    elif report.get("status") == "reviewed":
        st.success(score_label(report))
    else:
        st.info(f"{score_label(report)}。这是辅助评阅结果，可在下方由教师复核。")
    st.caption(f"报告 {report.get('report_id', '未保存')} · 任务 {report.get('task_id', '')} · 学号 {report.get('student_id') or '未提供'}")
    render_source(report, key)
    if report.get("review_reasons") and report.get("status") == "reviewed":
        st.caption("初次评阅触发的复核原因（已完成教师复核，保留供追溯）")
        for reason in report["review_reasons"]:
            st.markdown(reason)
    else:
        for reason in report.get("review_reasons", []):
            st.warning(reason)
    for warning in report.get("warnings", []):
        st.warning(warning)
    if report.get("question_text"):
        st.markdown("#### 本次题干")
        st.markdown(report["question_text"])
    if report.get("standard_answer"):
        st.markdown("#### 本次标准答案")
        st.markdown(report["standard_answer"])

    st.markdown("#### 识别原文")
    steps = report.get("extracted_steps", [])
    if not steps:
        st.info("报告中没有可用的识别原文，请核对原始答卷。")
    for step in steps:
        st.caption(f"第 {step.get('page_number', '未知')} 页 · 识别步骤 {step.get('step_id', '')}")
        st.markdown(step.get("content") or "（空）")

    st.markdown("#### 评分点与教材依据")
    rubric = {item["step_id"]: item for item in report.get("rubric", [])}
    render_details(report.get("details", []), rubric, scored(report))
    if report.get("status") == "reviewed":
        # A toggle also works inside the batch report's expander; nesting another
        # expander is unsupported by the project's Streamlit version.
        if st.toggle("展开原始 AI 建议与教材来源", key=f"{key}_ai_audit"):
            st.caption("以下为教师复核前保存的 AI 建议和当时引用的教材，供追溯查阅。当前成绩以上方教师复核结果为准。")
            ai_score = report.get("ai_total_score")
            st.info("原始 AI 未出建议分" if ai_score is None else f"原始 AI 建议分：{ai_score:g} 分")
            if report.get("ai_details"):
                render_details(report["ai_details"], rubric, ai_score is not None)
            else:
                st.caption("本报告没有可用的原始 AI 评分明细。")
    if report.get("overall_feedback"):
        st.markdown("#### 综合评价")
        st.markdown(report["overall_feedback"])
    if report.get("review_history"):
        st.markdown("#### 复核记录")
        for review in report["review_history"]:
            st.caption(f"复核人：{review.get('reviewer', '未记录')} · {review_time(review.get('reviewed_at'))}")
            st.markdown(review.get("reason") or "未提供说明。")
            for score in review.get("scores", []):
                st.caption(f"评分点 {score.get('step_id')}：{score.get('points_awarded')} 分")
                if score.get("feedback"):
                    st.markdown(score["feedback"])
    st.download_button(
        "下载完整报告（JSON）", data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=f"report_{report.get('report_id') or 'result'}.json", mime="application/json",
        key=f"{key}_download",
    )
    render_review(report, key)


def current_batch(result):
    reports = [latest_report(report) for report in result.get("results", [])]
    values = [report["total_score"] for report in reports if scored(report)]
    summary = dict(result.get("summary", {}))
    summary.update({
        "graded_students": len(values),
        "needs_review_students": sum(not scored(report) for report in reports),
        "average_score": round(sum(values) / len(values), 2) if values else None,
        "max_score": max(values) if values else None,
        "min_score": min(values) if values else None,
    })
    return {"summary": summary, "results": reports}


def display_number(value):
    return "未出分" if value is None else f"{value:g}"


if st.session_state.get("review_notice"):
    st.success(st.session_state.pop("review_notice"))

tab_setup, tab_single, tab_batch, tab_history = st.tabs([
    "01  题目与评分规则", "02  单份批改", "03  批量批改", "04  历史与复核",
])

with tab_setup:
    heading, refresh = st.columns([3, 1])
    with heading:
        section_heading("01", "准备一道好题，从清晰的规则开始", "填写完整题干与标准答案，让每个评分点都有据可依。")
    with refresh:
        refresh_tasks = st.button("刷新已有任务", key="refresh_tasks", use_container_width=True)
    editor, guide = st.columns([2.8, 1], gap="large")
    with editor:
        if refresh_tasks:
            try:
                st.session_state["task_list"] = api_json("GET", "/tasks").get("tasks", [])
            except (requests.RequestException, ValueError) as exc:
                st.error(f"任务列表加载失败：{exc}")
        tasks = st.session_state.get("task_list", [])
        if tasks:
            task_map = {task["task_id"]: task for task in tasks}
            selected_task = st.selectbox(
                "选择已保存任务", list(task_map),
                format_func=lambda task_id: f"{task_id} · {task_map[task_id]['title']}" + (
                    " · 需补充题干" if not task_map[task_id].get("ready_for_grading", False) else ""
                ),
            )
            if not task_map[selected_task].get("ready_for_grading", False):
                st.warning("此任务尚未具备批改条件。请加载任务，补齐题干并核验答案后保存。")
            if st.button("加载任务到下方编辑器", key="load_task"):
                try:
                    task = api_json("GET", f"/tasks/{quote(selected_task, safe='')}")
                    for widget_key, field in [
                        ("editor_task_id", "task_id"), ("editor_title", "title"),
                        ("editor_question", "question_text"), ("editor_answer", "standard_answer"),
                    ]:
                        st.session_state[widget_key] = task.get(field) or ""
                    st.session_state["editor_rubric"] = json.dumps(task.get("rubric", []), ensure_ascii=False, indent=2)
                    st.session_state.rubric_editor_version += 1
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"任务加载失败：{exc}")
        elif "task_list" in st.session_state:
            st.info("暂无已保存任务。填写下方内容，创建第一道题目。")

        with st.form("task_editor"):
            st.markdown("#### 题目内容")
            col_a, col_b = st.columns([1, 2])
            with col_a:
                task_id = st.text_input("作业任务 ID", key="editor_task_id", help="批改作业时用此 ID 选择对应题目。相同 ID 再次保存会更新原任务。")
            with col_b:
                title = st.text_input("任务名称", key="editor_title")
            question = st.text_area("完整题干（必填，含参数、条件和各小题）", key="editor_question", height=115)
            answer = st.text_area("标准答案（具体推导和计算结果，支持公式）", key="editor_answer", height=135)
            st.divider()
            st.markdown("#### 评分细则")
            st.caption("逐项填写评分要求与分值，可新增评分点或删除选中项。")
            rubric = []
            for index, item in enumerate(json.loads(st.session_state.editor_rubric)):
                row_key = f"rubric_{st.session_state.rubric_editor_version}_{index}"
                desc_col, points_col = st.columns([4, 1])
                with desc_col:
                    description = st.text_area(
                        f"评分点 {item['step_id']} · 评分要求", value=item.get("description", ""),
                        key=f"{row_key}_description", height=80,
                    )
                with points_col:
                    points = st.number_input(
                        f"分值 · 第 {item['step_id']} 项", min_value=0.0,
                        value=float(item.get("points", 0)), step=0.5, key=f"{row_key}_points",
                    )
                    st.checkbox("删除此项", key=f"{row_key}_remove")
                rubric.append({"step_id": item["step_id"], "description": description, "points": points})
            add_col, remove_col = st.columns(2)
            with add_col:
                st.form_submit_button("＋ 新增评分点", use_container_width=True, on_click=edit_rubric_rows, args=("add",))
            with remove_col:
                st.form_submit_button("删除选中评分点", use_container_width=True, on_click=edit_rubric_rows, args=("remove",))
            st.caption("保存会更新相同任务 ID 的题目和规则。")
            save_task = st.form_submit_button("保存题干、答案和评分规则", type="primary", use_container_width=True)
        if save_task:
            try:
                if not all(text.strip() for text in [task_id, title, question, answer]):
                    raise ValueError("任务 ID、名称、完整题干和标准答案均须填写。")
                if not rubric:
                    raise ValueError("请至少添加一个评分点。")
                step_ids = [item.get("step_id") for item in rubric]
                if any(not isinstance(step_id, (int, float)) or not math.isfinite(step_id) or step_id < 1 or int(step_id) != step_id for step_id in step_ids) or len(set(step_ids)) != len(step_ids):
                    raise ValueError("评分点序号须为不重复的正整数。")
                if any(not str(item.get("description") or "").strip() or not isinstance(item.get("points"), (int, float)) or not math.isfinite(item["points"]) or item["points"] <= 0 for item in rubric):
                    raise ValueError("请为每个评分点填写评分要求和大于 0 的分值。")
                api_json("POST", "/upload_task", json={
                    "task_id": task_id.strip(), "title": title.strip(), "question_text": question.strip(),
                    "standard_answer": answer.strip(), "rubric": rubric,
                })
                st.success("题干、标准答案和评分规则已保存。同一任务 ID 的规则会更新。")
                st.session_state.pop("task_list", None)
            except (requests.RequestException, ValueError) as exc:
                st.error(f"保存失败：{exc}")
    with guide:
        note_card("BEFORE YOU BEGIN", "好反馈，始于好标准。", "完整的条件、明确的答案、合理的分值，是每次辅助评阅的起点。")
        with st.container(border=True):
            st.markdown("##### 准备清单")
            st.markdown("**01 · 题干完整**")
            st.caption("写明系统、参数与所有小题。")
            st.markdown("**02 · 答案核验**")
            st.caption("确认推导过程与最终计算结果。")
            st.markdown("**03 · 按步骤给分**")
            st.caption("每个评分点对应明确的作答要求。")
        st.caption("请填写对应题目及经教师核验的答案，或加载已保存任务后继续编辑。")

with tab_single:
    section_heading("02", "一份作答，一次细致的反馈", "上传答卷，核对识别内容，再由教师确认成绩。")
    col_input, col_report = st.columns([1, 1.65], gap="large")
    with col_input:
        st.markdown("#### 提交作业")
        single_task_id = st.text_input("批改任务 ID", key="single_task_id")
        student_id = st.text_input("学生学号", key="single_student_id")
        uploaded = st.file_uploader("选择一份作业（JPG / PNG / PDF）", type=["jpg", "jpeg", "png", "pdf"], key="single_upload")
        if uploaded is not None and uploaded.type in {"image/jpeg", "image/png"}:
            try:
                st.image(uploaded.getvalue(), caption="本次上传的答卷", use_column_width=True)
            except (OSError, ValueError):
                st.warning("图片无法预览，请检查上传文件。")
        grade = st.button("开始单份批改", type="primary", use_container_width=True)
        st.caption("JPG / PNG / PDF · 单文件 ≤ 20 MB · PDF ≤ 20 页")
        st.caption("每个文件对应当前任务的一份作答。批改前请核对任务 ID。")
    with col_report:
        st.markdown("#### 评阅结果")
        if grade:
            if uploaded is None or not single_task_id.strip():
                st.warning("请填写任务 ID 并上传作业。")
            else:
                progress = st.empty()
                st.session_state.pop("single_report", None)
                try:
                    with requests.post(
                        f"{API_BASE}/grade_homework", files={"image": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        data={"task_id": single_task_id.strip(), "student_id": student_id.strip()},
                        stream=True, timeout=GRADING_TIMEOUT,
                    ) as response:
                        if not response.ok:
                            raise ValueError(f"请求失败（{response.status_code}）：{response.text}")
                        result = None
                        stream_error = None
                        for line in response.iter_lines():
                            if not line:
                                continue
                            decoded = line.decode("utf-8")
                            if decoded.startswith("PROGRESS:"):
                                progress.info(decoded[9:])
                            elif decoded.startswith("ERROR:"):
                                stream_error = decoded[6:]
                            elif decoded.startswith("RESULT:"):
                                result = json.loads(decoded[7:])
                        if stream_error:
                            raise ValueError(stream_error)
                        if result is None:
                            raise ValueError("连接已结束，但未收到报告。请刷新历史报告确认是否已保存。")
                        st.session_state["single_report"] = result
                except (requests.RequestException, ValueError) as exc:
                    st.error(f"批改请求失败：{exc}")
                finally:
                    progress.empty()
        if st.session_state.get("single_report"):
            report = st.session_state.single_report
            render_report(report, f"single_{report.get('report_id', 'result')}")
        else:
            empty_state("等待第一份作答", "在左侧选择题目并上传作业。识别原文、逐项建议分和教材依据将在这里呈现。")

with tab_batch:
    section_heading("03", "批量处理，逐份留痕", "一次上传同一任务的多份作业，集中查看建议分与待复核项。")
    batch_input, batch_guide = st.columns([2, 1], gap="large")
    with batch_input:
        with st.container(border=True):
            batch_task_id = st.text_input("批量使用的任务 ID", key="batch_task_id")
            uploads = st.file_uploader("选择多份作业", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True, key="batch_upload")
            st.caption("每批 ≤ 100 份 · 单文件 ≤ 20 MB · PDF ≤ 20 页")
            if uploads:
                st.caption(f"已选择 {len(uploads)} 份作业，合计 {sum(upload.size for upload in uploads) / 1024 / 1024:.1f} MB")
            start_batch = st.button("开始批量批改", type="primary", use_container_width=True)
    with batch_guide:
        note_card("BATCH REVIEW", "一起上传，分别复核。", "每个文件计为一份提交。建议以“学号-姓名”命名；同一学生的多页作答请合并为一个 PDF。所有文件须对应同一任务。")
    if start_batch:
        if not uploads or not batch_task_id.strip():
            st.warning("请填写任务 ID 并上传至少一份作业。")
        else:
            st.session_state.pop("batch_result", None)
            try:
                with st.spinner(f"正在处理 {len(uploads)} 份提交，报告将保存到本地…"):
                    response = requests.post(
                        f"{API_BASE}/grade_homework_batch",
                        files=[("files", (upload.name, upload.getvalue(), upload.type)) for upload in uploads],
                        data={"task_id": batch_task_id.strip()}, timeout=BATCH_TIMEOUT,
                    )
                    if not response.ok:
                        raise ValueError(f"请求失败（{response.status_code}）：{response.text}")
                    st.session_state["batch_result"] = response.json()
            except (requests.RequestException, ValueError) as exc:
                st.error(f"批量请求失败：{exc}。可刷新历史报告查看已保存的部分结果。")
    if st.session_state.get("batch_result"):
        batch = current_batch(st.session_state.batch_result)
        summary = batch["summary"]
        if summary["needs_review_students"]:
            st.warning(f"有 {summary['needs_review_students']} 份提交待教师复核，尚未出分。")
        elif summary.get("failed_students", 0):
            st.warning("部分文件处理失败，请检查下方记录。")
        else:
            st.info("批量处理完成，请核对建议得分和教材依据。")
        columns = st.columns(4)
        for column, label, value in zip(columns, ["提交份数", "已出建议分 / 已复核", "待复核份数", "处理失败份数"], [
            summary.get("total_students", len(batch["results"])), summary["graded_students"],
            summary["needs_review_students"], summary.get("failed_students", 0),
        ]):
            column.metric(label, value)
        col_avg, col_max, col_min = st.columns(3)
        col_avg.metric("平均分", display_number(summary["average_score"]))
        col_max.metric("最高分", display_number(summary["max_score"]))
        col_min.metric("最低分", display_number(summary["min_score"]))
        st.caption("平均分、最高分和最低分只统计已生成建议分或已由教师复核的报告。")
        for error in summary.get("errors", []):
            st.error(f"{error.get('filename', '文件')} · {error.get('student_id', '')}：{error.get('error', '处理失败')}")
        st.download_button("下载本批次全部报告（JSON）", json.dumps(batch, ensure_ascii=False, indent=2),
                           file_name="grading_batch.json", mime="application/json", key="batch_download")
        for index, report in enumerate(batch["results"]):
            with st.expander(f"第 {index + 1} 份 · 学号 {report.get('student_id') or '未提供'} · {score_label(report)}"):
                render_report(report, f"batch_{index}_{report.get('report_id', 'result')}")
    else:
        empty_state("把重复工作交给流程", "批量处理完成后，这里会汇总得分、待复核作业和每份报告。教师可逐份核对原件，保留完整评阅记录。", "≡")

with tab_history:
    history_heading, history_refresh = st.columns([3, 1])
    with history_heading:
        section_heading("04", "每一次评阅，都有迹可循", "查看已保存的原件、建议分与复核记录，继续完成教师确认。")
    with history_refresh:
        refresh_reports = st.button("刷新历史报告", key="refresh_reports", use_container_width=True)
    st.caption("报告保存在本机。点击刷新读取历史记录，原始答卷按需加载。")
    if refresh_reports:
        try:
            st.session_state["report_list"] = api_json("GET", "/reports").get("reports", [])
        except (requests.RequestException, ValueError) as exc:
            st.error(f"历史报告读取失败：{exc}")
    reports = st.session_state.get("report_list", [])
    if reports:
        history_map = {report["report_id"]: latest_report(report) for report in reports}
        selected_report = st.selectbox(
            "选择报告", list(history_map),
            format_func=lambda report_id: f"{report_id} · {history_map[report_id].get('student_id') or '未提供学号'} · {score_label(history_map[report_id])}",
        )
        if history_map[selected_report].get("error"):
            st.error(history_map[selected_report]["error"])
        if st.button("打开报告", key="load_report"):
            try:
                loaded = api_json("GET", f"/reports/{quote(selected_report, safe='')}")
                st.session_state["history_report"] = loaded
                st.session_state.updated_reports[selected_report] = loaded
            except (requests.RequestException, ValueError) as exc:
                st.error(f"报告加载失败：{exc}")
    elif "report_list" in st.session_state:
        empty_state("还没有评阅记录", "完成第一份作业批改后，报告会保存在这里，方便随时回看与复核。", "▤")
    else:
        empty_state("让每一次反馈都有记录", "点击“刷新历史报告”读取本机记录，查看作答原件与完整评阅过程。", "▤")
    if st.session_state.get("history_report"):
        report = st.session_state.history_report
        render_report(report, f"history_{report.get('report_id', 'result')}")

st.markdown('<div class="app-footer">CONTROL LAB &nbsp; / &nbsp; 以教材为依据，以教师复核为准。</div>', unsafe_allow_html=True)
