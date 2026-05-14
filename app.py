"""
CareerVision AI — Smart Career Prediction System
A Streamlit app that predicts top career paths from a user's
course, skills and interests, and generates a personalized roadmap.
"""

import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF, FPDFException

# ----------------------------- PAGE CONFIG -----------------------------
st.set_page_config(
    page_title=" Smart Career Prediction System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- CUSTOM CSS ------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes floatGlow {
  0%,100% { transform: translateY(0); box-shadow: 0 0 60px rgba(124,92,255,0.25); }
  50%     { transform: translateY(-4px); box-shadow: 0 0 90px rgba(54,197,255,0.35); }
}
@keyframes fadeUp {
  from { opacity:0; transform: translateY(14px); }
  to   { opacity:1; transform: translateY(0); }
}
@keyframes pulseRing {
  0% { box-shadow: 0 0 0 0 rgba(124,92,255,0.55); }
  70% { box-shadow: 0 0 0 14px rgba(124,92,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(124,92,255,0); }
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
      radial-gradient(1200px 600px at 85% -10%, rgba(124,92,255,0.25), transparent 60%),
      radial-gradient(900px 500px at -10% 20%, rgba(54,197,255,0.18), transparent 60%),
      radial-gradient(800px 600px at 50% 110%, rgba(255,90,200,0.15), transparent 60%),
      linear-gradient(180deg, #05060f 0%, #07091a 50%, #03040c 100%);
    color: #ffffff;
    background-attachment: fixed;
}
.stApp::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
      linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(ellipse at center, black 30%, transparent 75%);
}
.block-container { position:relative; z-index:1; padding-top: 2rem; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(8,10,28,0.92), rgba(3,4,12,0.96));
    border-right: 1px solid rgba(124,92,255,0.25);
    backdrop-filter: blur(20px);
    box-shadow: 4px 0 40px rgba(0,0,0,0.5);
}
section[data-testid="stSidebar"] * { color:#e9ecff !important; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color:#ffffff; letter-spacing:-0.02em; }
p, span, label, li { color:#e3e6ff; }

.hero {
    position:relative;
    padding: 2.6rem 2.2rem;
    border-radius: 26px;
    background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(54,197,255,0.12) 60%, rgba(255,90,200,0.10));
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
    box-shadow: 0 20px 60px -20px rgba(124,92,255,0.55), inset 0 1px 0 rgba(255,255,255,0.08);
    margin-bottom: 1.8rem;
    animation: fadeUp .7s ease both, floatGlow 7s ease-in-out infinite;
    overflow:hidden;
}
.hero::after{
    content:""; position:absolute; inset:-2px; border-radius:26px; padding:1px;
    background: linear-gradient(120deg, #7c5cff, #36c5ff, #ff5ac8, #7c5cff);
    background-size: 300% 300%;
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    animation: gradientShift 8s ease infinite;
    pointer-events:none;
}
.hero h1 {
    font-size: 2.9rem; margin: 0; font-weight:700;
    background: linear-gradient(90deg, #c9b8ff, #6ee7ff 50%, #ff9ad6);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
}
.hero p { color: #cfd3f7; margin-top: .6rem; font-size: 1.08rem; }

.glass {
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(18px) saturate(130%);
    -webkit-backdrop-filter: blur(18px) saturate(130%);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 12px 40px -16px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
    animation: fadeUp .6s ease both;
}
.glass:hover {
    transform: translateY(-4px);
    border-color: rgba(124,92,255,0.45);
    box-shadow: 0 20px 60px -18px rgba(124,92,255,0.55);
}

.career-title {
    font-size: 1.45rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bff, #36c5ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.match-badge {
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    background: linear-gradient(90deg, #7c5cff, #36c5ff);
    color: white; font-weight: 600; font-size: 0.85rem;
    box-shadow: 0 6px 18px rgba(54,197,255,0.35);
    animation: pulseRing 2.4s infinite;
}
.tag {
    display:inline-block; padding:4px 11px; margin:3px;
    border-radius:999px; font-size:0.78rem;
    background: linear-gradient(135deg, rgba(124,92,255,0.22), rgba(54,197,255,0.18));
    color:#e6e1ff;
    border:1px solid rgba(124,92,255,0.4);
    backdrop-filter: blur(8px);
    transition: all .2s ease;
}
.tag:hover { transform: translateY(-1px); border-color:#7c5cff; color:#fff; }

.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#7c5cff,#36c5ff,#ff5ac8);
    background-size: 200% 100%;
    animation: gradientShift 3s ease infinite;
}
.stButton>button {
    background: linear-gradient(90deg,#7c5cff,#36c5ff);
    color:white; border:none; border-radius:14px;
    padding: .65rem 1.5rem; font-weight:600; letter-spacing:.3px;
    box-shadow: 0 8px 24px -8px rgba(124,92,255,0.6);
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    filter: brightness(1.15) saturate(1.1);
    box-shadow: 0 14px 32px -8px rgba(54,197,255,0.7);
}

.metric-card {
    background: linear-gradient(135deg, rgba(124,92,255,0.20), rgba(54,197,255,0.10));
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(14px);
    border-radius:18px; padding:1.1rem 1.2rem; text-align:center;
    transition: transform .25s ease, box-shadow .25s ease;
    animation: fadeUp .6s ease both;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 14px 36px -12px rgba(124,92,255,0.55); }
.metric-card .v {
    font-size:2rem; font-weight:700;
    background: linear-gradient(90deg,#c9b8ff,#6ee7ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-card .l { font-size:.78rem; color:#aab0d8; text-transform:uppercase; letter-spacing:1.5px;}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(124,92,255,0.30) !important;
    color:#000 !important;
    border-radius: 12px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(0,0,0,0.45) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color:#36c5ff !important;
    box-shadow: 0 0 0 3px rgba(54,197,255,0.18) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 8px 16px; color:#cfd3f7;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(124,92,255,0.35), rgba(54,197,255,0.25)) !important;
    border-color: rgba(124,92,255,0.6) !important; color:#fff !important;
}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color:#fff !important;
}

hr { border-color: rgba(255,255,255,0.08); }
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background: #06081a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg,#7c5cff,#36c5ff);
    border-radius: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------- DATA LOAD -------------------------------
@st.cache_data
def load_careers():
    df = pd.read_csv("careers.csv")
    for col in ["required_skills", "interests", "courses", "certifications",
                "roadmap", "tools", "projects", "interview_topics"]:
        df[col + "_list"] = df[col].fillna("").apply(
            lambda s: [x.strip().lower() for x in s.replace("|", ",").split(",") if x.strip()]
        )
    return df

careers_df = load_careers()

# ----------------------------- SCORING ENGINE --------------------------
def tokenize(text: str):
    return [t.strip().lower() for t in text.replace(";", ",").split(",") if t.strip()]

def overlap_score(user_items, target_items):
    if not user_items or not target_items:
        return 0.0
    uset, tset = set(user_items), set(target_items)
    matched = sum(1 for u in uset if any(u in t or t in u for t in tset))
    return min(matched / max(len(tset), 1), 1.0)

@st.cache_data
def predict_careers(course, skills, interests, strengths, top_n=3):
    user_skills = tokenize(skills) + tokenize(strengths)
    user_interests = tokenize(interests)
    user_course = course.strip().lower()

    rows = []
    for _, row in careers_df.iterrows():
        skill_s = overlap_score(user_skills, row["required_skills_list"])
        interest_s = overlap_score(user_interests, row["interests_list"])
        course_s = 1.0 if any(user_course and (user_course in c or c in user_course)
                              for c in row["courses_list"]) else 0.3
        final = round((skill_s * 0.55 + interest_s * 0.30 + course_s * 0.15) * 100, 2)
        matched_skills = sorted(set(s for s in user_skills
                                    if any(s in t or t in s for t in row["required_skills_list"])))
        missing_skills = [s for s in row["required_skills_list"] if s not in matched_skills][:6]
        rows.append({
            "career": row["career"],
            "match": min(final, 99.5),
            "skill_score": round(skill_s * 100, 1),
            "interest_score": round(interest_s * 100, 1),
            "course_score": round(course_s * 100, 1),
            "future_scope": row["future_scope"],
            "certifications": row["certifications_list"],
            "roadmap": row["roadmap_list"],
            "tools": row["tools_list"],
            "projects": row["projects_list"],
            "interview": row["interview_topics_list"],
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
        })
    rows.sort(key=lambda r: (r["match"], r["career"]), reverse=True)
    return rows[:top_n], rows

# ----------------------------- PDF EXPORT ------------------------------
def _normalize_pdf_text(value):
    return str(value or "").replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")


def _write_pdf_text(pdf, text, line_height=6):
    text = _normalize_pdf_text(text)
    try:
        pdf.multi_cell(0, line_height, text, align="L")
    except FPDFException:
        pdf.multi_cell(0, line_height, text, align="L", wrapmode="CHAR")


def build_pdf(user_info, predictions):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "CareerVision AI - Personalized Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Your Profile", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for k, v in user_info.items():
        _write_pdf_text(pdf, f"{k}: {v}")
    pdf.ln(2)
    for i, p in enumerate(predictions, 1):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, f"{i}. {p['career']}  -  {p['match']:.1f}% match", ln=True)
        pdf.set_font("Helvetica", "", 10)
        _write_pdf_text(pdf, f"Future scope: {p['future_scope']}", line_height=5)
        _write_pdf_text(pdf, "Roadmap: " + ", ".join(p["roadmap"]), line_height=5)
        _write_pdf_text(pdf, "Certifications: " + ", ".join(p["certifications"]), line_height=5)
        _write_pdf_text(pdf, "Tools: " + ", ".join(p["tools"]), line_height=5)
        _write_pdf_text(pdf, "Project ideas: " + ", ".join(p["projects"]), line_height=5)
        _write_pdf_text(pdf, "Skill gaps to close: " + ", ".join(p["missing_skills"]), line_height=5)
        pdf.ln(3)
    return bytes(pdf.output(dest="S"))

# ----------------------------- SIDEBAR --------------------------------
with st.sidebar:
    st.markdown("## 🚀 Smart Career Prediction System")
    st.caption("Smart career prediction & roadmap engine")
    page = st.radio("Navigate",
                    ["🏠 Home", "🎯 Predict Career", "📊 Dashboard",
                     "🔥 Trending Careers", "🤖 AI Assistant", "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Tip:** Be specific in skills & interests for sharper predictions.")

# ----------------------------- HOME -----------------------------------
def render_hero(title, subtitle):
    st.markdown(f"""
    <div class="hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

if page == "🏠 Home":
    render_hero("Smart Career Prediction System",
                "An intelligent career guidance system that predicts your best-fit careers and builds your personalized learning roadmap.")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [("10+", "Careers"), ("AI", "Powered"), ("100%", "Personalized"), ("PDF", "Export")]
    for col, (v, l) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f'<div class="metric-card"><div class="v">{v}</div><div class="l">{l}</div></div>',
                     unsafe_allow_html=True)
    st.markdown("### ✨ How it works")
    cols = st.columns(3)
    steps = [
        ("1. Tell us about you", "Share your course, skills, interests and strengths."),
        ("2. AI matches careers", "Our scoring engine ranks 10+ careers by skill, interest & course fit."),
        ("3. Get your roadmap", "Step-by-step plan, certs, tools, projects & a downloadable PDF."),
    ]
    for col, (h, p) in zip(cols, steps):
        col.markdown(f'<div class="glass"><h4>{h}</h4><p style="color:#bcc1e6">{p}</p></div>',
                     unsafe_allow_html=True)

# ----------------------------- PREDICT --------------------------------
elif page == "🎯 Predict Career":
    render_hero("Predict your best career", "Fill the form — the AI engine will rank your top matches.")
    with st.form("predict"):
        c1, c2 = st.columns(2)
        with c1:
            course = st.text_input("🎓 Course / Degree", placeholder="e.g. B.Tech, BCA, MBA, B.Sc")
            skills = st.text_area("🛠️ Technical Skills (comma separated)",
                                  placeholder="python, sql, machine learning, react")
        with c2:
            interests = st.text_area("💡 Interests",
                                     placeholder="data, ai, design, web, security")
            strengths = st.text_area("⭐ Strengths",
                                     placeholder="problem solving, communication, math")
        submitted = st.form_submit_button("🚀 Predict My Career")
    if submitted:
        if not (skills.strip() or interests.strip()):
            st.warning("Please enter at least your skills or interests.")
        else:
            with st.spinner("🤖 Analyzing your profile..."):
                top, all_scores = predict_careers(course, skills, interests, strengths)
            st.session_state["top"] = top
            st.session_state["all_scores"] = all_scores
            st.session_state["user_info"] = {
                "Course": course, "Skills": skills,
                "Interests": interests, "Strengths": strengths,
            }
    if "top" in st.session_state:
        top = st.session_state["top"]
        st.markdown("## 🎯 Your Top Career Matches")
        for i, p in enumerate(top, 1):
            st.markdown(f"""
            <div class="glass">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <div class="career-title">#{i} · {p['career']}</div>
                <div class="match-badge">{p['match']:.1f}% match</div>
              </div>
              <p style="color:#c8cdf0;margin-top:.6rem;">{p['future_scope']}</p>
            </div>""", unsafe_allow_html=True)
            st.progress(min(int(p["match"]), 100))
            with st.expander(f"📍 Roadmap & resources for {p['career']}"):
                a, b = st.columns(2)
                with a:
                    st.markdown("**🗺️ Step-by-step roadmap**")
                    for step in p["roadmap"]:
                        st.markdown(f"- {step.title()}")
                    st.markdown("**🛠️ Tools & technologies**")
                    st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in p["tools"]]),
                                unsafe_allow_html=True)
                with b:
                    st.markdown("**📜 Certifications**")
                    for c in p["certifications"]:
                        st.markdown(f"- {c.title()}")
                    st.markdown("**💡 Project ideas**")
                    for pr in p["projects"]:
                        st.markdown(f"- {pr.title()}")
                    st.markdown("**🎤 Interview prep**")
                    st.markdown(" ".join([f'<span class="tag">{t}</span>' for t in p["interview"]]),
                                unsafe_allow_html=True)
                st.markdown("**🧩 Skill gap analysis**")
                if p["missing_skills"]:
                    st.markdown(" ".join([f'<span class="tag">{s}</span>' for s in p["missing_skills"]]),
                                unsafe_allow_html=True)
                else:
                    st.success("You already cover the core skills! 🎉")

        # PDF download
        pdf_bytes = build_pdf(st.session_state["user_info"], top)
        st.download_button("⬇️ Download Roadmap as PDF", pdf_bytes,
                           file_name="careervision_roadmap.pdf", mime="application/pdf")

# ----------------------------- DASHBOARD ------------------------------
elif page == "📊 Dashboard":
    render_hero("Your AI Dashboard", "Visual breakdown of your career matches and skill profile.")
    if "all_scores" not in st.session_state:
        st.info("Run a prediction first from the **Predict Career** page.")
    else:
        scores = st.session_state["all_scores"]
        df = pd.DataFrame(scores)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df.sort_values("match"), x="match", y="career", orientation="h",
                         color="match", color_continuous_scale="Plasma",
                         title="Career Match %")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="#e7e9ff", height=480)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top1 = scores[0]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[top1["skill_score"], top1["interest_score"], top1["course_score"],
                   top1["match"], 100 - len(top1["missing_skills"]) * 10],
                theta=["Skills", "Interests", "Course Fit", "Overall Match", "Readiness"],
                fill="toself", line_color="#7c5cff"))
            radar.update_layout(title=f"Profile fit · {top1['career']}",
                                polar=dict(bgcolor="rgba(255,255,255,0.03)",
                                           radialaxis=dict(range=[0, 100], color="#aab0d8")),
                                paper_bgcolor="rgba(0,0,0,0)", font_color="#e7e9ff", height=480)
            st.plotly_chart(radar, use_container_width=True)

        st.markdown("### 🔍 AI Insights")
        st.markdown(f"""
        <div class="glass">
        <p>✅ Best fit: <b>{scores[0]['career']}</b> with <b>{scores[0]['match']:.1f}%</b> confidence.</p>
        <p>📈 Strongest signal: <b>{'Skills' if scores[0]['skill_score']>=scores[0]['interest_score'] else 'Interests'}</b>.</p>
        <p>🧩 Close skill gaps in: <b>{', '.join(scores[0]['missing_skills'][:4]) or 'none — you’re ready!'}</b></p>
        </div>""", unsafe_allow_html=True)

# ----------------------------- TRENDING -------------------------------
elif page == "🔥 Trending Careers":
    render_hero("Trending Careers 2026", "Most in-demand roles, ranked by market signal.")
    trend = pd.DataFrame({
        "Career": careers_df["career"],
        "Demand Index": np.random.randint(70, 99, size=len(careers_df)),
        "Avg Salary (LPA)": np.random.randint(6, 28, size=len(careers_df)),
    }).sort_values("Demand Index", ascending=False)
    fig = px.scatter(trend, x="Avg Salary (LPA)", y="Demand Index",
                     size="Demand Index", color="Career", text="Career",
                     title="Demand vs Salary")
    fig.update_traces(textposition="top center")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e7e9ff", height=560, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(trend, use_container_width=True)

# ----------------------------- CHATBOT --------------------------------
elif page == "🤖 AI Assistant":
    render_hero("AI Career Assistant", "Ask me about careers, skills, or roadmaps.")
    if "chat" not in st.session_state:
        st.session_state.chat = [("ai", "Hi! I'm your CareerVision assistant. Ask me anything about careers 🚀")]
    for role, msg in st.session_state.chat:
        with st.chat_message("assistant" if role == "ai" else "user"):
            st.write(msg)
    q = st.chat_input("Ask about a career, skill, or roadmap...")
    if q:
        st.session_state.chat.append(("user", q))
        ql = q.lower()
        reply = None
        for _, row in careers_df.iterrows():
            if row["career"].lower() in ql:
                reply = (f"**{row['career']}** — {row['future_scope']}\n\n"
                         f"**Key skills:** {row['required_skills']}\n\n"
                         f"**Roadmap:** {row['roadmap'].replace('|', ' → ')}")
                break
        if not reply:
            if "skill" in ql:
                reply = "Pick 3-5 core skills, build projects around them, and showcase on GitHub. Quality > quantity."
            elif "roadmap" in ql or "learn" in ql:
                reply = "Use the **Predict Career** page to get a full personalized roadmap with certifications and projects."
            elif "salary" in ql:
                reply = "Salaries vary by role and region. Check the **Trending Careers** page for current ranges."
            else:
                reply = "Try asking about a specific career like *Data Analyst*, *AI/ML Engineer*, or *Cloud Engineer*."
        st.session_state.chat.append(("ai", reply))
        st.rerun()

# ----------------------------- ABOUT ----------------------------------
else:
    render_hero("About CareerVision AI",
                "A final-year style AI mini-project showcasing intelligent career prediction with Streamlit.")
    st.markdown("""
    <div class="glass">
    <h3>Tech Stack</h3>
    <p>Python · Streamlit · Pandas · NumPy · Plotly · Matplotlib · FPDF</p>
    <h3>Features</h3>
    <ul>
        <li>AI-style scoring engine (skills · interests · course)</li>
        <li>Top-3 career predictions with match %</li>
        <li>Personalized roadmaps, certifications, project ideas</li>
        <li>Interactive Plotly dashboard & radar chart</li>
        <li>Skill gap analysis & PDF export</li>
        <li>AI assistant chatbot</li>
    </ul>
    <h3>Deployment</h3>
    <p>Push to GitHub → connect on <b>share.streamlit.io</b> → deploy. Done.</p>
    </div>""", unsafe_allow_html=True)
