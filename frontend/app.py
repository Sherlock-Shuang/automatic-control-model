import streamlit as st
import requests
import json

st.set_page_config(page_title="智能作业批改 MVP", layout="wide")

API_BASE = "http://localhost:8000/api"

st.title("👨‍🏫 智能作业批改系统 (MVP版)")

tab1, tab2 = st.tabs(["1. 设定标准答案与规则", "2. 批改作业照片"])

with tab1:
    st.header("Step 1: 初始化作业规则")
    task_id = st.text_input("作业任务 ID", value="hw_001")
    title = st.text_input("题目描述", value="求闭环传递函数并判断系统稳定性")
    standard_answer = st.text_area("标准答案步骤 (支持公式意象描述)", 
        value="【步骤1】: 列出前向传递函数 G(s)。\n【步骤2】: 列出特征方程 1+G(s)H(s)=0。\n【步骤3】: 根据劳斯判据列劳斯表，判断首列符号，得出不稳定结论。")
    
    st.subheader("得分细则 (Rubric)")
    # Simplification for MVP: We use fixed JSON to mock rubric builder
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

with tab2:
    st.header("Step 2: 上传学生答卷相片")
    hw_task_id = st.text_input("要批改的具体任务 ID", value="hw_001")
    student_id = st.text_input("学生学号", value="stu_0920")
    uploaded_file = st.file_uploader("选择批改图片文件", type=["jpg", "png", "jpeg"])
    
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
