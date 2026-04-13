import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Model selection - you can configure via litellm/OpenAI compatible API to point to Qwen / Claude etc.
llm_vlm = ChatOpenAI(model="gpt-4o", temperature=0.1) # For visual tasks
llm_logic = ChatOpenAI(model="gpt-4o", temperature=0.2) # For logic matching

def node_a_extract_steps(image_base64: str) -> str:
    """视觉解析器 (OCR & Layout Extraction)"""
    prompt = """
    你是一个公式与手写识别专家。请将图片中的解答步骤逐一提取，遇到数学公式请严格使用 LaTeX 语法输出。
    最后以 JSON 数组返回格式：
    [
        {"step_id": 1, "content": "写出的传递函数..."},
        {"step_id": 2, "content": "列出的特征方程..."}
    ]
    不要输出任何其他多余文本，仅输出合法的 JSON 形态。
    """
    # Note: In a real implementation we would pass the base64 image array to ChatOpenAI.
    # Below is pseudo-code representation for MVP.
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}
    ]
    response = llm_vlm.invoke(messages)
    return response.content

def node_b_logic_matcher(student_steps_json: str, standard_answer: str, rubric: list) -> str:
    """逻辑裁判引擎 (Logic Matcher)"""
    prompt = f"""
    比较学生步骤与标准答案。判断学生的每一步是否逻辑成立。允许同义替换或等价代数变形。
    如果发现错误，明确指出在第几步、错误类型（计算错/概念错/逻辑断裂）。
    
    【标准答案】:
    {standard_answer}
    
    【得分细则 (Rubric)】:
    {json.dumps([r.model_dump() for r in rubric], ensure_ascii=False)}
    
    【学生作答提取记录】:
    {student_steps_json}
    
    输出格式为 JSON 列表：
    [
        {{"step_id": 1, "is_correct": true, "points_awarded": 2.0, "student_step_description": "...", "error_type": null, "feedback": "..."}}
    ]
    不要包含其他多余描述内容，直接输出合法的JSON数组。
    """
    response = llm_logic.invoke([{"role": "user", "content": prompt}])
    return response.content

def node_c_rag_feedback(grading_results: list, similarity_search_func, embed_func) -> dict:
    """RAG 增强反馈生成 (Feedback Generator)"""
    overall_feedback = ""
    # Process incorrect steps
    for step in grading_results:
        if not step.get("is_correct"):
            error_context = step.get("feedback", "")
            # Get embedding of error context
            query_vector = embed_func(error_context)
            # Query db
            kb_matches = similarity_search_func(query_vector)
            
            kb_text = "\n".join([doc["content"] for doc in kb_matches]) if kb_matches else "无额外知识库参考"
            
            # Formulate specific feedback
            rag_prompt = f"""
            学生在答题中出现了以下错误：{error_context}
            请结合以下教材知识，给予学生指导和错误剖析：
            【教材参考点】: {kb_text}
            """
            rag_result = llm_logic.invoke([{"role": "user", "content": rag_prompt}])
            step["rag_knowledge"] = rag_result.content
            overall_feedback += f"第{step['step_id']}步出错: {rag_result.content}\n\n"
    
    return {
        "enhanced_results": grading_results,
        "overall": overall_feedback if overall_feedback else "太棒了，这道题完全正确！"
    }
