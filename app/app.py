
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is available for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from model_loader import load_all_models
from inference import compare_models


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AlignLab | AI Summarization",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(80, 75, 180, 0.16), transparent 38%),
        radial-gradient(ellipse at 90% 10%, rgba(0, 180, 170, 0.09), transparent 35%),
        #090D18;
    color: #E8ECF5;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    padding: 25px 0 30px 0;
}

.eyebrow {
    color: #A99CFF;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(38px, 5vw, 58px);
    font-weight: 700;
    letter-spacing: -2.5px;
    line-height: 1.08;
    color: #F5F6FF;
    margin: 0;
}

.gradient-text {
    background: linear-gradient(90deg, #A99CFF, #65D8CE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #929DB2;
    font-size: 16px;
    margin-top: 15px;
    max-width: 650px;
    line-height: 1.7;
}

.section-label {
    color: #B7C0D2;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 600;
    margin: 20px 0 12px 0;
}

.input-hint {
    color: #78859D;
    font-size: 12px;
    margin-top: -8px;
    margin-bottom: 15px;
}

div[data-testid="stTextArea"] textarea {
    background: #111827;
    color: #E8ECF5;
    border: 1px solid #293449;
    border-radius: 12px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    line-height: 1.7;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #8F83F7;
    box-shadow: 0 0 0 1px #8F83F7;
}

.stButton > button {
    background: linear-gradient(100deg, #7568E8, #5B9DEB);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    min-height: 48px;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(117, 104, 232, 0.25);
}

.model-card {
    background: linear-gradient(145deg, #151D2D, #101725);
    border: 1px solid #28344B;
    border-radius: 16px;
    padding: 20px;
    min-height: 270px;
    margin-bottom: 12px;
}

.model-name {
    color: #F0F2FA;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 700;
}

.model-subtitle {
    color: #8492AA;
    font-size: 12px;
    margin-top: 4px;
}

.model-divider {
    border: none;
    border-top: 1px solid #293449;
    margin: 17px 0;
}

.summary-label {
    color: #A99CFF;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.summary-text {
    color: #D5DCE9;
    font-size: 14px;
    line-height: 1.85;
    overflow-wrap: anywhere;
}

.score-label {
    color: #8492AA;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.score-value {
    color: #71D9C7;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 5px;
}

.info-box {
    background: rgba(117, 104, 232, 0.08);
    border: 1px solid rgba(117, 104, 232, 0.22);
    border-radius: 12px;
    padding: 15px 18px;
    color: #AEB9CE;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 20px;
}

.footer {
    border-top: 1px solid #222C3D;
    margin-top: 55px;
    padding-top: 20px;
    color: #68758B;
    font-size: 12px;
    text-align: center;
}

div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #293449;
    padding: 12px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #9BA8BD;
}

div[data-testid="stMetricValue"] {
    color: #71D9C7;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading SFT, PPO, DPO and Reward models...")
def get_models():
    return load_all_models()


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ HUMAN FEEDBACK · LANGUAGE MODEL ALIGNMENT</div>
    <h1>From Text to <span class="gradient-text">Aligned Summaries.</span></h1>
    <div class="hero-description">
        Explore how human preference optimization changes language model
        behavior. Compare Supervised Fine-Tuning, PPO and DPO on the same
        Reddit post, with reward model evaluation.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown('<div class="section-label">01 / Enter your Reddit post</div>',
            unsafe_allow_html=True)

post = st.text_area(
    "Reddit post",
    height=210,
    placeholder="Subreddit: technology\n\nTitle: ...\n\nBody: Paste your Reddit post here...",
    label_visibility="collapsed"
)

st.markdown(
    '<div class="input-hint">Paste the title and body of a Reddit post to generate summaries using all three models.</div>',
    unsafe_allow_html=True
)

button_col, info_col = st.columns([1, 3])

with button_col:
    generate_clicked = st.button(
        "✦  Compare Models",
        use_container_width=True
    )

with info_col:
    st.caption("Three models · One input · Shared reward model")


# --------------------------------------------------
# GENERATION
# --------------------------------------------------

if generate_clicked:

    if not post.strip():
        st.warning("Please enter a Reddit post first.")

    else:
        try:
            models = get_models()

            with st.spinner("Generating summaries and calculating reward scores..."):
                results = compare_models(post, models)

            st.session_state["comparison_results"] = results
            st.session_state["comparison_post"] = post

        except Exception as error:
            st.error(f"Something went wrong: {error}")


# --------------------------------------------------
# RESULTS SECTION
# --------------------------------------------------

if "comparison_results" in st.session_state:

    results = st.session_state["comparison_results"]

    st.markdown("---")
    st.markdown(
        '<div class="section-label">02 / Model comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="input-hint">Each model receives the same input post. Summaries are generated independently.</div>',
        unsafe_allow_html=True
    )

    model_details = [
        ("sft", "SFT", "Supervised Fine-Tuning", "Baseline model"),
        ("ppo", "PPO", "Proximal Policy Optimization", "RLHF optimization"),
        ("dpo", "DPO", "Direct Preference Optimization", "Preference alignment")
    ]

    columns = st.columns(3, gap="medium")

    for column, (key, name, subtitle, tag) in zip(columns, model_details):

        result = results[key]
        summary = result["summary"]
        score = result["reward_score"]

        with column:
            st.markdown('<div class="model-card">', unsafe_allow_html=True)

            st.markdown(
                f'<div class="model-name">{name}</div>'
                f'<div class="model-subtitle">{subtitle}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="model-subtitle">✦ {tag}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<hr class="model-divider">', unsafe_allow_html=True)

            st.markdown(
                '<div class="summary-label">Generated summary</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="summary-text">{summary if summary else "No summary generated."}</div>',
                unsafe_allow_html=True
            )

            st.markdown('<hr class="model-divider">', unsafe_allow_html=True)

            st.markdown(
                '<div class="score-label">Reward model score</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="score-value">{score:.4f}</div>',
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

    # Reward comparison
    st.markdown(
        '<div class="section-label">03 / Reward score comparison</div>',
        unsafe_allow_html=True
    )

    score_data = {
        "SFT": results["sft"]["reward_score"],
        "PPO": results["ppo"]["reward_score"],
        "DPO": results["dpo"]["reward_score"]
    }

    chart_col, explanation_col = st.columns([2, 1])

    with chart_col:
        st.bar_chart(
            score_data,
            horizontal=True,
            height=190
        )

    with explanation_col:
        st.markdown("""
        <div class="info-box">
            <b>Understanding the scores</b><br><br>
            All three summaries are evaluated by the same trained Reward Model.
            A higher score indicates a higher predicted preference under this
            reward model. It is not a human evaluation or a universal quality score.
        </div>
        """, unsafe_allow_html=True)

    # Input preview
    with st.expander("View original Reddit post"):
        st.write(st.session_state["comparison_post"])

    if st.button("↻  Clear comparison"):
        del st.session_state["comparison_results"]
        del st.session_state["comparison_post"]
        st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    ALIGNLAB · SFT → REWARD MODEL → PPO / DPO<br>
    Built to explore learning from human preferences.
</div>
""", unsafe_allow_html=True)
