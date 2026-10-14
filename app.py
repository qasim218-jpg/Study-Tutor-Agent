import streamlit as st
from pypdf import PdfReader

from agent import ask_tutor
from memory import add_message, get_memory_text


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Study Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — ACCESSIBLE DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --bg: #0b1120;
        --panel: #111b2e;
        --panel-soft: #162238;
        --border: #2a3a54;
        --text: #f1f5f9;
        --muted: #b6c3d6;
        --accent: #67e8f9;
        --accent-strong: #38bdf8;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(56,189,248,.10), transparent 28%),
            radial-gradient(circle at 95% 8%, rgba(129,140,248,.10), transparent 25%),
            #0b1120;
        color: var(--text);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
    }

    /* Make all normal text readable, including markdown and captions */
    .stApp, .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--text);
    }

    .stApp [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    a { color: #7dd3fc !important; }

    /* Hero */
    .hero {
        padding: 30px 32px;
        margin-bottom: 24px;
        border: 1px solid #2c4867;
        border-radius: 22px;
        background: linear-gradient(125deg, #14253d 0%, #111a30 65%, #1a1d3a 100%);
        box-shadow: 0 16px 42px rgba(0,0,0,.22);
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 11px;
        margin-bottom: 13px;
        border-radius: 999px;
        background: rgba(103,232,249,.10);
        border: 1px solid rgba(103,232,249,.30);
        color: #a5f3fc !important;
        font-size: 12px;
        font-weight: 750;
        letter-spacing: .09em;
    }

    .hero-title {
        color: #f8fafc !important;
        font-size: clamp(30px, 4.5vw, 50px);
        font-weight: 850;
        line-height: 1.08;
        letter-spacing: -.04em;
        margin: 0;
    }

    .hero-subtitle {
        color: #c6d3e5 !important;
        font-size: 16px;
        line-height: 1.65;
        margin: 12px 0 0;
        max-width: 720px;
    }

    /* Feature cards */
    .feature-card {
        height: 100%;
        min-height: 150px;
        padding: 20px;
        border-radius: 17px;
        background: linear-gradient(145deg, #142238, #101a2d);
        border: 1px solid #2a3b54;
        box-shadow: 0 8px 24px rgba(0,0,0,.16);
        transition: border-color .2s ease, transform .2s ease;
    }

    .feature-card:hover {
        border-color: #4d789d;
        transform: translateY(-2px);
    }

    .feature-icon { font-size: 27px; margin-bottom: 9px; }
    .feature-title {
        color: #f8fafc !important;
        font-size: 16px;
        font-weight: 750;
        margin-bottom: 7px;
    }
    .feature-description {
        color: #c0cde0 !important;
        font-size: 13px;
        line-height: 1.6;
    }

    .section-title {
        color: #f8fafc !important;
        font-size: 21px;
        font-weight: 800;
        margin: 26px 0 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1424 0%, #101b2d 100%);
        border-right: 1px solid #273850;
    }

    [data-testid="stSidebar"] * {
        color: #edf4ff;
    }

    .sidebar-logo { text-align: center; padding: 12px 0 20px; }
    .sidebar-logo-icon { font-size: 40px; margin-bottom: 5px; }
    .sidebar-logo-title {
        color: #a5f3fc !important;
        font-size: 20px;
        font-weight: 850;
    }
    .sidebar-logo-subtitle {
        color: #b8c6da !important;
        font-size: 12px;
        margin-top: 4px;
    }

    .status-card {
        padding: 14px;
        border-radius: 14px;
        background: #142238;
        border: 1px solid #2b405b;
        margin-top: 12px;
    }
    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #d5e0ef !important;
        font-size: 13px;
        margin: 8px 0;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        box-shadow: 0 0 8px rgba(74,222,128,.45);
        flex: 0 0 auto;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #111d30;
        border: 1px solid #293c56;
        border-radius: 16px;
        margin-bottom: 12px;
        padding: 12px 14px;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #f1f5f9 !important;
    }

    /* Inputs, select menus, text areas and chat input */
    input, textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: #f8fafc !important;
        background-color: #101b2d !important;
        -webkit-text-fill-color: #f8fafc !important;
        caret-color: #67e8f9 !important;
    }

    [data-baseweb="select"] > div {
        color: #f8fafc !important;
        background-color: #101b2d !important;
        border-color: #354a66 !important;
        border-radius: 10px !important;
    }

    [data-baseweb="select"] span,
    [data-baseweb="select"] input {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    [role="listbox"], [role="option"] {
        color: #f8fafc !important;
        background-color: #142238 !important;
    }

    [data-testid="stChatInput"] {
        background: #101b2d !important;
        border-color: #3b5878 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        color: #f0fbff !important;
        background: linear-gradient(135deg, #164e63, #3730a3);
        border: 1px solid #3b718d;
        border-radius: 11px;
        min-height: 42px;
        font-weight: 700;
        transition: filter .2s ease, border-color .2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #ffffff !important;
        border-color: #67e8f9;
        filter: brightness(1.12);
    }

    /* File uploader and alerts */
    [data-testid="stFileUploader"] section {
        background: #111d30;
        border: 1px dashed #3b526e;
        border-radius: 13px;
    }
    [data-testid="stFileUploader"] *,
    [data-testid="stAlert"] * {
        color: #f1f5f9 !important;
    }

    [data-testid="stExpander"] {
        background: #111d30;
        border: 1px solid #293c56;
        border-radius: 12px;
    }

    hr { border-color: #2b3b52 !important; }

    /* Focus indicators */
    button:focus-visible, input:focus-visible, textarea:focus-visible {
        outline: 2px solid #67e8f9 !important;
        outline-offset: 2px !important;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 22px; border-radius: 17px; }
        .hero-title { font-size: 33px; }
        .hero-subtitle { font-size: 14px; }
        .feature-card { margin-bottom: 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = []

if "study_material" not in st.session_state:
    st.session_state.study_material = ""

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">

            <div class="sidebar-logo-icon">🎓</div>

            <div class="sidebar-logo-title">
                Study Tutor AI
            </div>

            <div class="sidebar-logo-subtitle">
                Your intelligent study companion
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 📚 Study Material")

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        help="Upload lecture notes, textbook chapters, or study material.",
    )

    if uploaded_file:

        if uploaded_file.name != st.session_state.pdf_name:

            try:

                reader = PdfReader(uploaded_file)

                text = ""

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"

                st.session_state.study_material = text
                st.session_state.pdf_name = uploaded_file.name

                st.success(
                    f"✓ {len(reader.pages)} pages loaded"
                )

            except Exception as e:

                st.error(
                    f"Could not read PDF: {e}"
                )

    elif not st.session_state.study_material:

        st.caption(
            "Upload lecture notes or a textbook PDF "
            "to let your tutor study with you."
        )


    st.markdown("---")

    st.markdown("### ⚙️ Tutor Settings")

    difficulty = st.selectbox(
        "Learning Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
        ],
    )

    response_style = st.selectbox(
        "Teaching Style",
        [
            "Simple & Clear",
            "Detailed",
            "Step-by-Step",
            "Exam Focused",
        ],
    )


    st.markdown("---")

    st.markdown("### 🧠 Session Memory")

    message_count = len(
        st.session_state.memory
    )

    if message_count:

        st.markdown(
            f"""
            <div class="status-card">

                <div class="status-row">
                    <span class="status-dot"></span>
                    {message_count} messages remembered
                </div>

                <div class="status-row">
                    📄 Study material:
                    {"Loaded" if st.session_state.study_material else "Not loaded"}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "Your conversation will be remembered "
            "during this study session."
        )


    st.markdown("")

    if st.button("🗑️ Clear Conversation"):

        st.session_state.memory = []

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ AI-POWERED LEARNING
        </div>

        <h1 class="hero-title">
            Study smarter.<br>
            Understand deeper.
        </h1>

        <p class="hero-subtitle">
            Your personal AI tutor for explanations,
            practice, questions, and learning.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">💡</div>

            <div class="feature-title">
                Understand Concepts
            </div>

            <div class="feature-description">
                Get difficult topics explained
                in simple, student-friendly language.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">📚</div>

            <div class="feature-title">
                Learn From Your Notes
            </div>

            <div class="feature-description">
                Upload your study material and
                ask questions about what you're learning.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with col3:

    st.markdown(
        """
        <div class="feature-card">

            <div class="feature-icon">🧠</div>

            <div class="feature-title">
                Personalized Tutoring
            </div>

            <div class="feature-description">
                Your tutor remembers the current
                conversation and adapts to your level.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">💬 Ask Your Tutor</div>',
    unsafe_allow_html=True,
)


# Display previous messages

for message in st.session_state.memory:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your studies..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    add_message(
        st.session_state.memory,
        "user",
        question,
    )


    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    conversation = get_memory_text(
        st.session_state.memory
    )


    # --------------------------------------------------------
    # Tutor context
    # --------------------------------------------------------

    tutor_context = f"""
Student learning level:
{difficulty}

Preferred teaching style:
{response_style}

"""

    enhanced_question = (
        tutor_context
        + "\nStudent question:\n"
        + question
    )


    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🧠 Your tutor is thinking..."
        ):

            try:

                answer = ask_tutor(

                    question=enhanced_question,

                    study_material=(
                        st.session_state.study_material
                    ),

                    conversation_memory=conversation,

                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "I encountered an error while "
                    "processing your question."
                )

                st.error(answer)

                with st.expander(
                    "Technical details"
                ):

                    st.code(str(e))


    # --------------------------------------------------------
    # Save response
    # --------------------------------------------------------

    add_message(
        st.session_state.memory,
        "assistant",
        answer,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:12px;
        padding:15px;
    ">

        Study Tutor AI • Built with Streamlit, CrewAI & Groq

    </div>
    """,
    unsafe_allow_html=True,
)
