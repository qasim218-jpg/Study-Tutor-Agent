```python
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 229, 255, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(76, 0, 255, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #050816 0%,
                #081127 50%,
                #050816 100%
            );

        color: #f5f7ff;
    }

    /* Main content */

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .hero {
        padding: 28px 30px;
        margin-bottom: 25px;

        border: 1px solid rgba(0, 229, 255, 0.20);
        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(10, 25, 60, 0.90),
                rgba(10, 15, 35, 0.75)
            );

        box-shadow:
            0 0 40px rgba(0, 229, 255, 0.08),
            inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .hero-badge {
        display: inline-block;

        padding: 6px 12px;
        margin-bottom: 12px;

        border-radius: 999px;

        background: rgba(0, 229, 255, 0.10);
        border: 1px solid rgba(0, 229, 255, 0.30);

        color: #5ee7ff;

        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .hero-title {
        font-size: clamp(30px, 5vw, 52px);
        font-weight: 800;

        line-height: 1.05;

        margin: 0;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #67e8f9,
                #38bdf8,
                #a78bfa
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #a9b5d0;
        font-size: 16px;
        margin-top: 12px;
        margin-bottom: 0;
    }


    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .feature-card {
        height: 100%;

        padding: 20px;

        border-radius: 18px;

        background:
            rgba(12, 24, 52, 0.65);

        border: 1px solid rgba(120, 160, 255, 0.13);

        box-shadow:
            0 8px 30px rgba(0,0,0,0.20);

        transition: 0.25s ease;
    }

    .feature-card:hover {
        border-color: rgba(0, 229, 255, 0.35);

        box-shadow:
            0 0 25px rgba(0, 229, 255, 0.10);
    }

    .feature-icon {
        font-size: 27px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-weight: 700;
        font-size: 16px;
        color: #ffffff;
    }

    .feature-description {
        color: #8998b7;
        font-size: 13px;
        line-height: 1.5;
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* ========================================================
       CHAT AREA
       ======================================================== */

    .chat-container {
        padding: 10px 0;
    }

    [data-testid="stChatMessage"] {
        background: rgba(12, 25, 52, 0.55);

        border: 1px solid rgba(120, 160, 255, 0.10);

        border-radius: 18px;

        margin-bottom: 12px;

        padding: 8px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #060b1b 0%,
                #081127 100%
            );

        border-right: 1px solid rgba(0, 229, 255, 0.12);
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff;
    }

    .sidebar-logo {
        text-align: center;
        padding: 12px 0 22px 0;
    }

    .sidebar-logo-icon {
        font-size: 42px;
    }

    .sidebar-logo-title {
        font-size: 20px;
        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #67e8f9,
                #818cf8
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-logo-subtitle {
        color: #71809d;
        font-size: 12px;
    }


    /* ========================================================
       STATUS CARD
       ======================================================== */

    .status-card {
        padding: 15px;

        border-radius: 16px;

        background: rgba(8, 20, 43, 0.75);

        border: 1px solid rgba(0, 229, 255, 0.12);

        margin-top: 15px;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 9px;

        color: #a9b5d0;

        font-size: 13px;

        margin: 8px 0;
    }

    .status-dot {
        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 10px rgba(34, 197, 94, 0.8);
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100%;

        border-radius: 12px;

        border: 1px solid rgba(0, 229, 255, 0.25);

        background:
            linear-gradient(
                135deg,
                rgba(0, 170, 255, 0.15),
                rgba(88, 80, 255, 0.15)
            );

        color: #dffaff;

        font-weight: 650;

        min-height: 42px;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #00e5ff;

        box-shadow:
            0 0 18px rgba(0, 229, 255, 0.18);

        color: #ffffff;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(8, 20, 43, 0.55);

        border-radius: 16px;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        border-color: rgba(0, 229, 255, 0.25);
    }


    /* ========================================================
       SELECTBOX / INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {
        background: rgba(8, 20, 43, 0.70);

        border-color: rgba(120, 160, 255, 0.18);

        border-radius: 12px;
    }

    input {
        background: rgba(8, 20, 43, 0.70) !important;
    }


    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: rgba(120, 160, 255, 0.10);
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 34px;
        }

        .hero-subtitle {
            font-size: 14px;
        }

        .feature-card {
            margin-bottom: 10px;
        }
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

        Study Tutor AI • Powered by CrewAI + Groq

    </div>
    """,
    unsafe_allow_html=True,
)
```
