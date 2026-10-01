"""Shared, local-only presentation for the two Control Lab workspaces."""

from html import escape
from pathlib import Path

import streamlit as st


def apply_theme():
    """Load trusted styles only; student/model content never passes through here."""
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_sidebar(mode="teacher"):
    is_chat = mode == "chat"
    with st.sidebar:
        st.markdown('''<div class="lab-brand"><span class="lab-monogram">C<span>↗</span></span>
<div>CONTROL LAB<small>自动控制 · 教与学</small></div></div>''', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">课程空间 / WORKSPACE</div>', unsafe_allow_html=True)
        title, subtitle = ("课程问答", "从概念到推导，连接每一个知识点") if is_chat else ("教师工作台", "让评阅有依据，让反馈更清晰")
        st.markdown(f'<div class="workspace-label"><span class="status-dot"></span>{title}<small>{subtitle}</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">' + ("学习路径" if is_chat else "评阅流程") + '</div>', unsafe_allow_html=True)
        steps = [("提出问题", "描述概念、公式或推导疑问"), ("检索教材", "从课程资料中寻找依据"), ("理解与追问", "结合公式和原图继续探索")] if is_chat else [("准备题目", "核对题干、答案与评分细则"), ("上传作业", "支持单份与批量评阅"), ("复核成绩", "对照原件，确认每一项得分")]
        for i, (heading, caption) in enumerate(steps, 1):
            st.markdown(f'<div class="sidebar-step"><span>0{i}</span><div>{heading}<small>{caption}</small></div></div>', unsafe_allow_html=True)
        st.divider()
        with st.expander("使用小贴士", expanded=False):
            if is_chat:
                st.write("问题越具体，越容易找到对应教材内容。可写明系统、参数和疑问所在的步骤。")
            else:
                st.write("先保存题目，再填写对应任务 ID 上传作业。同一学生的多页作答请合为一个 PDF。")
                st.write("AI 提供建议分，成绩由教师核对原件后确认。")
        st.markdown('<div class="sidebar-footer"><span class="status-dot"></span>本地课程工作空间<br><small>CONTROL THEORY · LEARNING WITH EVIDENCE</small></div>', unsafe_allow_html=True)


def render_hero(mode="teacher"):
    is_chat = mode == "chat"
    title = "把抽象的原理，<br>变成清晰的理解。" if is_chat else "教学有据，<br>评阅有序。"
    subtitle = "与课程助教一起，从教材出发，理解概念、推导公式与系统响应。" if is_chat else "从题目准备到成绩复核，为每一份作答提供清晰、可追溯的反馈。"
    label = "课程问答" if is_chat else "教师工作台"
    st.markdown(f'<div class="page-topline"><span>课程空间 <b>/</b> 自动控制原理</span><span class="role-badge">{label}</span></div>', unsafe_allow_html=True)
    st.markdown(f'''<section class="course-hero">
<div class="hero-copy"><div class="eyebrow">CONTROL THEORY / 自动控制原理</div>
<h1>{title}</h1><p>{subtitle}</p>
<div class="hero-tags"><span>教材知识支持</span><span>{"公式与原图" if is_chat else "教师最终复核"}</span></div></div>
<div class="hero-figure" aria-label="负反馈系统示意图"><div class="figure-label">A LITTLE FEEDBACK. A BETTER SYSTEM.</div>
<svg viewBox="0 0 380 144" role="img" aria-label="输入经过传递函数 G(s)，输出通过 H(s) 负反馈">
<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="currentColor"/></marker></defs>
<g fill="none" stroke="currentColor" stroke-width="1.5"><path d="M18 50 H80 M111 50 H158 M250 50 H348" marker-end="url(#arrow)"/>
<circle cx="96" cy="50" r="15"/><rect x="160" y="26" width="90" height="48" rx="5"/>
<path d="M303 50 V117 H250 M160 117 H96 V69" marker-end="url(#arrow)"/><rect x="160" y="96" width="90" height="42" rx="5"/></g>
<g fill="currentColor" font-family="Georgia, serif" text-anchor="middle"><text x="31" y="34" font-size="14">R(s)</text><text x="205" y="57" font-size="23" font-style="italic">G(s)</text><text x="337" y="34" font-size="14">C(s)</text><text x="205" y="123" font-size="19" font-style="italic">H(s)</text><text x="95" y="55" font-size="17">+</text><text x="81" y="83" font-size="16">−</text></g></svg>
<div class="figure-caption"><span>闭环，让反馈发挥价值。</span><span>FIG. 01</span></div></div></section>''', unsafe_allow_html=True)


def section_heading(number, title, description):
    st.markdown(f'<div class="section-heading"><span class="section-number">{escape(number)}</span><div><h2>{escape(title)}</h2><p>{escape(description)}</p></div></div>', unsafe_allow_html=True)


def empty_state(title, description, symbol="◎"):
    st.markdown(f'<div class="empty-state"><div class="empty-symbol">{escape(symbol)}</div><h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def note_card(kicker, title, description):
    st.markdown(f'<div class="note-card"><div class="eyebrow">{escape(kicker)}</div><h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)
