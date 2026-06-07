import streamlit as st
import requests
import json

st.set_page_config(page_title="智能作业批改 MVP", layout="wide")

API_BASE = "http://localhost:8000/api"

st.title("👨‍🏫 智能作业批改系统 (MVP版)")

tab1, tab2, tab3 = st.tabs(["1. 设定标准答案与规则", "2. 批改单份作业", "3. 批量批改作业"])

# ==========================================
# Tab 1: 设定标准答案与规则
# ==========================================
with tab1:
    st.header("Step 1: 初始化作业规则")
    task_id = st.text_input("作业任务 ID", value="hw_001")
    title = st.text_input("题目描述", value="求闭环传递函数并判断系统稳定性")
    standard_answer = st.text_area("标准答案步骤 (支持公式意象描述)", 
        value="【步骤1】: 列出前向传递函数 G(s)。\n【步骤2】: 列出特征方程 1+G(s)H(s)=0。\n【步骤3】: 根据劳斯判据列劳斯表，判断首列符号，得出不稳定结论。")
    
    st.subheader("得分细则 (Rubric)")
    rubric_str = st.text_area("JSON 格式配置", value='''[
        {"step_id": 1, "description": "列出前向传递函数", "points": 2.0},
        {"step_id": 2, "description": "计算特征方程并确保无计算常数错误", "points": 4.0},
        {"step_id": 3, "description": "正确使用劳斯判据并得出是否稳定", "points": 4.0}
    ]''', height=150)
    
    if st.button("提交标准答案"):
        try:
            rubric_data = json.loads(rubric_str)
            resp = requests.post(f"{API_BASE}/upload_task", json={
                "task_id": task_id,
                "title": title,
                "standard_answer": standard_answer,
                "rubric": rubric_data
            })
            if resp.status_code == 200:
                st.success("✅ 标准答案与细则已就绪！")
            else:
                st.error(resp.text)
        except Exception as e:
            st.error(f"JSON 格式错误或网络错误: {str(e)}")

    # 显示已有任务
    st.divider()
    if st.button("📋 查看已有任务"):
        try:
            resp = requests.get(f"{API_BASE}/tasks")
            if resp.status_code == 200:
                tasks = resp.json().get("tasks", [])
                if tasks:
                    for t in tasks:
                        st.info(f"**{t['task_id']}** - {t['title']} | {t['rubric_count']} 个评分项 | 总分 {t['total_points']}")
                else:
                    st.warning("暂无已创建的任务。")
        except Exception as e:
            st.error(f"获取任务列表失败: {str(e)}")

# ==========================================
# Tab 2: 批改单份作业
# ==========================================
with tab2:
    st.header("Step 2: 上传学生答卷")
    hw_task_id = st.text_input("要批改的具体任务 ID", value="hw_001", key="single_task_id")
    student_id = st.text_input("学生学号", value="stu_0920")
    uploaded_file = st.file_uploader(
        "选择批改文件（支持 JPG/PNG 图片和 PDF）", 
        type=["jpg", "png", "jpeg", "pdf"],
        key="single_upload"
    )
    
    if uploaded_file is not None and st.button("开始智能批改 🧠"):
        with st.spinner("系统正在解析步骤、对比逻辑、检索知识库进行 RAG 反馈... (预计15-30秒)"):
            try:
                files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {"task_id": hw_task_id, "student_id": student_id}
                
                res = requests.post(f"{API_BASE}/grade_homework", files=files, data=data)
                
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"批改完成！最终得分: {result['total_score']}")
                    
                    st.markdown("### 📝 步骤级成绩单")
                    for d in result["details"]:
                        correctness = "✅" if d.get("is_correct") else "❌"
                        st.markdown(f"**Step {d['step_id']}** {correctness} (得分: **{d['points_awarded']}**)")
                        st.markdown(f"> 学生步骤总结: {d.get('student_step_description')}")
                        if not d.get("is_correct"):
                            st.error(f"**错因分析:** {d.get('feedback')}")
                            if d.get("rag_knowledge"):
                                st.info(f"📚 **RAG 课本知识引申:**\n{d.get('rag_knowledge')}")
                        st.divider()
                        
                    st.markdown("### 🧠 综合诊断评价")
                    st.write(result["overall_feedback"])
                    
                else:
                    st.error(f"批改接口失败: {res.text}")
                    
            except Exception as e:
                 st.error(f"请求失败，确保后端已启动：{str(e)}")

# ==========================================
# Tab 3: 批量批改作业
# ==========================================
with tab3:
    st.header("Step 3: 批量上传学生作业")
    st.caption("支持同时上传多个文件（图片或 PDF），系统会自动从文件名提取学号。文件名格式建议：学号-姓名.pdf")
    
    batch_task_id = st.text_input("批改任务 ID", value="hw_001", key="batch_task_id")
    batch_files = st.file_uploader(
        "选择多个学生作业文件",
        type=["jpg", "png", "jpeg", "pdf"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if batch_files and st.button("开始批量批改 🚀"):
        st.info(f"已选择 {len(batch_files)} 个文件，开始处理...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 准备文件数据
            files_data = []
            for f in batch_files:
                files_data.append(("files", (f.name, f, f.type)))
            
            data = {"task_id": batch_task_id}
            
            status_text.text("正在发送文件到后端...")
            res = requests.post(f"{API_BASE}/grade_homework_batch", files=files_data, data=data)
            
            progress_bar.progress(100)
            
            if res.status_code == 200:
                result = res.json()
                summary = result["summary"]
                
                # 显示汇总
                st.success("批量批改完成！")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("总学生数", summary["total_students"])
                col2.metric("成功批改", summary["graded_students"])
                col3.metric("平均分", summary["average_score"])
                col4.metric("最高分 / 最低分", f"{summary['max_score']} / {summary['min_score']}")
                
                # 显示错误
                if summary.get("errors"):
                    st.warning(f"有 {summary['failed_students']} 个文件处理失败：")
                    for err in summary["errors"]:
                        st.error(f"  - {err['filename']} ({err['student_id']}): {err['error']}")
                
                # 显示每个学生的详细结果
                st.divider()
                st.markdown("### 📊 各学生成绩明细")
                
                for r in result["results"]:
                    with st.expander(f"👤 {r['student_id']} — 得分: {r['total_score']}", expanded=False):
                        for d in r["details"]:
                            correctness = "✅" if d.get("is_correct") else "❌"
                            st.markdown(f"**Step {d['step_id']}** {correctness} (得分: **{d['points_awarded']}**)")
                            st.markdown(f"> {d.get('student_step_description')}")
                            if not d.get("is_correct"):
                                st.error(f"错因: {d.get('feedback')}")
                                if d.get("rag_knowledge"):
                                    st.info(f"📚 RAG 反馈: {d.get('rag_knowledge')}")
                        st.markdown(f"**综合评价:** {r['overall_feedback']}")
                        
            else:
                st.error(f"批量批改接口失败: {res.text}")
                
        except Exception as e:
            st.error(f"请求失败，确保后端已启动：{str(e)}")
