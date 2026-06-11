import streamlit as st
import requests
import json

st.set_page_config(page_title="智能作业批改 MVP", layout="wide")

# ==========================================
# 注入高级 UI 样式 (含字体、拟态卡片、高光 Banner、KPI 卡片)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Glassmorphism Title Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(192, 132, 252, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 25px 25px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-subtitle {
    font-size: 1rem;
    color: #9CA3AF;
    margin: 0;
}

/* Beautiful Custom Cards */
.premium-card {
    background: #151B2C;
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
}

.step-card {
    background: #111625;
    border-left: 4px solid #6366F1;
    border-top: 1px solid rgba(255, 255, 255, 0.03);
    border-right: 1px solid rgba(255, 255, 255, 0.03);
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 0 12px 12px 0;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px 0 rgba(0,0,0,0.15);
}

.step-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.step-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Custom Status Chips */
.chip {
    padding: 3px 10px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.chip-success {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.chip-error {
    background: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Custom Alert Boxes for Feedback */
.feedback-box {
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 8px;
    padding: 12px 15px;
    margin-top: 10px;
    color: #FCA5A5;
    font-size: 0.92rem;
}

/* Academic Citation/RAG Card Style */
.rag-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.06) 0%, rgba(99, 102, 241, 0.01) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-left: 4px solid #818CF8;
    border-radius: 0 10px 10px 0;
    padding: 15px 18px;
    margin-top: 12px;
    box-shadow: inset 0 0 10px rgba(99, 102, 241, 0.05);
}

.rag-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #A5B4FC;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.rag-content {
    font-size: 0.9rem;
    color: #CBD5E1;
    line-height: 1.5;
}

/* KPI Custom Grid Dashboard */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-bottom: 25px;
}

@media (max-width: 1024px) {
    .kpi-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 640px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

.kpi-card {
    background: #151B2C;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
}

.kpi-card.blue { border-top: 4px solid #3B82F6; }
.kpi-card.green { border-top: 4px solid #10B981; }
.kpi-card.indigo { border-top: 4px solid #6366F1; }
.kpi-card.orange { border-top: 4px solid #F59E0B; }

.kpi-title {
    font-size: 0.8rem;
    color: #9CA3AF;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}

.kpi-val {
    font-size: 1.7rem;
    font-weight: 700;
    color: #F9FAFB;
}

.overall-feedback-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(192, 132, 252, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 22px;
    margin-top: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.overall-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #E0E7FF;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000/api"

# Header Banner
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🤖 《自动控制原理》智能作业批改系统</h1>
    <p class="hero-subtitle">基于大视觉语言模型 (VLM) 与教材知识库 (RAG) 的智能化作业批改与诊断演示台</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["⚙️ 设定标准答案与规则", "📝 批改单份作业", "🚀 批量批改作业"])

# ==========================================
# Tab 1: 设定标准答案与规则
# ==========================================
with tab1:
    st.markdown('<h3 style="color:#6366F1; margin-bottom:15px;">⚙️ 初始化作业评阅规则</h3>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        task_id = st.text_input("作业任务 ID", value="hw_001")
        title = st.text_input("题目描述", value="求闭环传递函数并判断系统稳定性")
    with col_b:
        standard_answer = st.text_area("标准答案步骤 (支持公式意象描述)", 
            value="【步骤1】: 列出前向传递函数 G(s)。\n【步骤2】: 列出特征方程 1+G(s)H(s)=0。\n【步骤3】: 根据劳斯判据列劳斯表，判断首列符号，得出不稳定结论。", height=122)
    
    st.subheader("得分细则 (Rubric)")
    rubric_str = st.text_area("JSON 格式配置", value='''[
        {"step_id": 1, "description": "列出前向传递函数", "points": 2.0},
        {"step_id": 2, "description": "计算特征方程并确保无计算常数错误", "points": 4.0},
        {"step_id": 3, "description": "正确使用劳斯判据并得出是否稳定", "points": 4.0}
    ]''', height=150)
    
    if st.button("提交标准答案", use_container_width=True):
        try:
            rubric_data = json.loads(rubric_str)
            resp = requests.post(f"{API_BASE}/upload_task", json={
                "task_id": task_id,
                "title": title,
                "standard_answer": standard_answer,
                "rubric": rubric_data
            })
            if resp.status_code == 200:
                st.success("✅ 标准答案与细则已成功录入数据库！")
            else:
                st.error(resp.text)
        except Exception as e:
            st.error(f"JSON 格式错误或网络错误: {str(e)}")

    # 显示已有任务
    st.divider()
    if st.button("📋 查看已有任务", use_container_width=True):
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
    st.markdown('<h3 style="color:#6366F1; margin-bottom:15px;">📝 批改单份作业</h3>', unsafe_allow_html=True)
    
    col_c, col_d = st.columns([1, 2])
    with col_c:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        hw_task_id = st.text_input("要批改的具体任务 ID", value="hw_001", key="single_task_id")
        student_id = st.text_input("学生学号", value="stu_0920")
        uploaded_file = st.file_uploader(
            "选择批改文件（支持 JPG/PNG/PDF）", 
            type=["jpg", "png", "jpeg", "pdf"],
            key="single_upload"
        )
        grade_btn = st.button("开始智能批改 🧠", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_d:
        if uploaded_file is not None and grade_btn:
            status_placeholder = st.empty()
            try:
                files = {"image": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                data = {"task_id": hw_task_id, "student_id": student_id}
                
                # 流式发送请求获取进度推送
                res = requests.post(f"{API_BASE}/grade_homework", files=files, data=data, stream=True)
                
                if res.status_code == 200:
                    result = None
                    error_msg = None
                    
                    for line in res.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode("utf-8")
                        
                        if decoded_line.startswith("PROGRESS:"):
                            progress_text = decoded_line[9:]
                            status_placeholder.info(f"⏳ [批改进度] {progress_text}")
                        elif decoded_line.startswith("ERROR:"):
                            error_msg = decoded_line[6:]
                            status_placeholder.error(f"❌ 批改出错: {error_msg}")
                        elif decoded_line.startswith("RESULT:"):
                            result = json.loads(decoded_line[7:])
                            status_placeholder.empty()  # 清理进度条
                            
                    if error_msg:
                        pass
                    elif result:
                        st.markdown(f'<div style="background:rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding:15px; border-radius:10px; margin-bottom:20px; font-size:1.15rem; font-weight:600; color:#34D399; text-align:center;">🎉 批改完成！最终得分: {result["total_score"]} 分</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div style="margin-bottom:15px;"><h4 style="color:#E2E8F0; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:8px; margin:0;">📝 步骤级评阅明细</h4></div>', unsafe_allow_html=True)
                        for d in result["details"]:
                            is_correct = d.get("is_correct")
                            status_chip = f'<span class="chip chip-success">✅ 步骤正确</span>' if is_correct else f'<span class="chip chip-error">❌ 检出错误</span>'
                            points = d.get('points_awarded', 0)
                            
                            step_html = f"""
                            <div class="step-card">
                                <div class="step-card-header">
                                    <div class="step-title">📍 步骤 {d['step_id']}</div>
                                    <div>
                                        {status_chip}
                                        <span style="margin-left: 10px; font-weight:700; color:#818CF8;">得分: {points} 分</span>
                                    </div>
                                </div>
                                <div style="color: #CBD5E1; font-size:0.92rem; line-height: 1.5;">
                                    <strong>学生解答内容:</strong> {d.get('student_step_description')}
                                </div>
                            """
                            
                            if not is_correct:
                                step_html += f"""
                                <div class="feedback-box">
                                    <strong>🔍 错因诊断:</strong> {d.get('feedback')}
                                </div>
                                """
                                
                                if d.get("rag_knowledge"):
                                    step_html += f"""
                                    <div class="rag-card">
                                        <div class="rag-header">📚 RAG 教材对照释义</div>
                                        <div class="rag-content">{d.get('rag_knowledge')}</div>
                                    </div>
                                    """
                            step_html += "</div>"
                            st.markdown(step_html, unsafe_allow_html=True)
                            
                        # Overall feedback
                        overall_html = f"""
                        <div class="overall-feedback-box">
                            <div class="overall-title">🧠 智能诊断综合评价</div>
                            <div style="color:#E2E8F0; font-size:1rem; line-height:1.6; white-space: pre-wrap;">{result['overall_feedback']}</div>
                        </div>
                        """
                        st.markdown(overall_html, unsafe_allow_html=True)
                else:
                    st.error(f"批改接口失败 (Status {res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"请求失败，确保后端已启动：{str(e)}")
        else:
            st.info("👈 请在左侧配置任务信息、上传答卷，并点击“开始智能批改”按钮。")

# ==========================================
# Tab 3: 批量批改作业
# ==========================================
with tab3:
    st.markdown('<h3 style="color:#6366F1; margin-bottom:15px;">🚀 批量批改作业</h3>', unsafe_allow_html=True)
    st.caption("支持同时上传多个文件（图片或 PDF），系统会自动从文件名提取学号。文件名格式建议：学号-姓名.pdf")
    
    col_e, col_f = st.columns([1, 2])
    with col_e:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        batch_task_id = st.text_input("批改任务 ID", value="hw_001", key="batch_task_id")
        batch_files = st.file_uploader(
            "选择多个学生作业文件",
            type=["jpg", "png", "jpeg", "pdf"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        batch_btn = st.button("开始批量批改 🚀", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_f:
        if batch_files and batch_btn:
            st.info(f"已选择 {len(batch_files)} 个文件，开始处理...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
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
                    
                    st.markdown('<div style="background:rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); padding:15px; border-radius:10px; margin-bottom:20px; font-size:1.15rem; font-weight:600; color:#34D399; text-align:center;">🎉 批量批改处理完成！</div>', unsafe_allow_html=True)
                    
                    # KPIs
                    kpi_html = f"""
                    <div class="kpi-grid">
                        <div class="kpi-card blue">
                            <div class="kpi-title">👥 总学生数</div>
                            <div class="kpi-val">{summary["total_students"]}</div>
                        </div>
                        <div class="kpi-card green">
                            <div class="kpi-title">✅ 成功批改</div>
                            <div class="kpi-val">{summary["graded_students"]}</div>
                        </div>
                        <div class="kpi-card indigo">
                            <div class="kpi-title">📈 平均得分</div>
                            <div class="kpi-val">{summary["average_score"]}</div>
                        </div>
                        <div class="kpi-card orange">
                            <div class="kpi-title">🏆 最高 / 最低分</div>
                            <div class="kpi-val">{summary["max_score"]} / {summary["min_score"]}</div>
                        </div>
                    </div>
                    """
                    st.markdown(kpi_html, unsafe_allow_html=True)
                    
                    if summary.get("errors"):
                        st.markdown('<h5 style="color:#EF4444;">⚠️ 部分文件处理失败：</h5>', unsafe_allow_html=True)
                        for err in summary["errors"]:
                            st.markdown(f"""
                            <div style="background:rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); padding: 8px 12px; border-radius: 6px; margin-bottom:8px; font-size:0.9rem; color:#FCA5A5;">
                                ❌ <strong>{err['filename']}</strong> ({err['student_id']}): {err['error']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                    st.divider()
                    st.markdown('<h4 style="color:#E2E8F0; margin-bottom:15px;">📊 各学生成绩明细</h4>', unsafe_allow_html=True)
                    
                    for r in result["results"]:
                        with st.expander(f"👤 学号: {r['student_id']} — 最终得分: {r['total_score']} 分", expanded=False):
                            for d in r["details"]:
                                is_correct = d.get("is_correct")
                                status_chip = f'<span class="chip chip-success">✅ 正确</span>' if is_correct else f'<span class="chip chip-error">❌ 错误</span>'
                                points = d.get('points_awarded', 0)
                                
                                step_html = f"""
                                <div class="step-card" style="background: rgba(255,255,255,0.01);">
                                    <div class="step-card-header">
                                        <div class="step-title" style="font-size:0.95rem;">📍 步骤 {d['step_id']}</div>
                                        <div>
                                            {status_chip}
                                            <span style="margin-left: 8px; font-weight:700; color:#818CF8; font-size:0.9rem;">得分: {points}</span>
                                        </div>
                                    </div>
                                    <div style="color: #94A3B8; font-size:0.88rem; line-height: 1.4;">
                                        <strong>解答:</strong> {d.get('student_step_description')}
                                    </div>
                                """
                                if not is_correct:
                                    step_html += f"""
                                    <div class="feedback-box" style="font-size:0.85rem; padding: 8px 12px;">
                                        <strong>诊断:</strong> {d.get('feedback')}
                                    </div>
                                    """
                                    if d.get("rag_knowledge"):
                                        step_html += f"""
                                        <div class="rag-card" style="padding: 10px 12px; font-size:0.85rem;">
                                            <div class="rag-header" style="font-size:0.75rem;">📚 RAG 教材对照</div>
                                            <div class="rag-content" style="font-size:0.85rem;">{d.get('rag_knowledge')}</div>
                                        </div>
                                        """
                                step_html += "</div>"
                                st.markdown(step_html, unsafe_allow_html=True)
                                
                            st.markdown(f"""
                            <div style="background:rgba(99, 102, 241, 0.05); border:1px solid rgba(99, 102, 241, 0.15); border-radius:8px; padding:12px; margin-top:10px; font-size:0.9rem; color:#E2E8F0;">
                                <strong>🎯 综合评价:</strong> {r['overall_feedback']}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error(f"批量批改接口失败: {res.text}")
            except Exception as e:
                st.error(f"请求失败，确保后端已启动：{str(e)}")
        else:
            st.info("👈 请在左侧上传学生作业包（支持多选），并点击“开始批量批改”按钮。")
