
import streamlit as st
from pypdf import PdfReader

from agent import ask_tutor
from memory import add_message, get_memory_text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Study Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — DARK, PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(56, 189, 248, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 95% 8%,
                rgba(129, 140, 248, 0.10),
                transparent 25%
            ),
            #0b1120;
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p, li, label {
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background: #101b2d;
        border-right: 1px solid #273850;
    }

    [data-testid="stSidebar"] * {
        color: #edf4ff;
    }

    [data-testid="stChatMessage"] {
        background: #111d30;
        border: 1px solid #293c56;
        border-radius: 16px;
        margin-bottom: 12px;
        padding: 12px 14px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #f1f5f9 !important;
    }

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

    .stButton > button,
    .stDownloadButton > button {
        color: #f0fbff !important;
        background: linear-gradient(135deg, #164e63, #3730a3);
        border: 1px solid #3b718d;
        border-radius: 11px;
        min-height: 42px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        color: #ffffff !important;
        border-color: #67e8f9;
        filter: brightness(1.12);
    }

    [data-testid="stFileUploader"] section {
        background: #111d30;
        border: 1px dashed #3b526e;
        border-radius: 13px;
    }

    [data-testid="stExpander"] {
        background: #111d30;
        border: 1px solid #293c56;
        border-radius: 12px;
    }

    [data-testid="stAlert"] p {
        color: inherit;
    }

    hr {
        border-color: #2b3b52 !important;
    }

    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 2px solid #67e8f9 !important;
        outline-offset: 2px !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem;
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

    # Native Streamlit components — no custom HTML
    st.title("🎓 Study Tutor AI")
    st.caption("Your intelligent study companion")

    st.divider()

    st.subheader("📚 Study Material")

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        help=(
            "Upload lecture notes, textbook chapters, "
            "or other study material."
        ),
    )

    if uploaded_file is not None:

        if uploaded_file.name != st.session_state.pdf_name:

            try:
                reader = PdfReader(uploaded_file)

                extracted_pages = []

                for page in reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_pages.append(page_text)

                extracted_text = "\n".join(extracted_pages)

                st.session_state.study_material = extracted_text
                st.session_state.pdf_name = uploaded_file.name

                if extracted_text.strip():
                    st.success(
                        f"PDF loaded: {len(reader.pages)} pages"
                    )
                else:
                    st.warning(
                        "The PDF contains no extractable text. "
                        "It may be scanned or image-based."
                    )

            except Exception as e:
                st.error("Could not read the uploaded PDF.")
                with st.expander("Technical details"):
                    st.code(str(e))

    if st.session_state.study_material:
        st.caption(
            f"Active document: {st.session_state.pdf_name}"
        )
    elif uploaded_file is None:
        st.caption(
            "Upload lecture notes or a textbook PDF "
            "to let your tutor use your study material."
        )

    st.divider()

    st.subheader("⚙️ Tutor Settings")

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

    st.divider()

    st.subheader("🧠 Session Memory")

    message_count = len(st.session_state.memory)

    if message_count:
        st.success(
            f"{message_count} messages in this session"
        )
    else:
        st.caption(
            "Your conversation will be remembered "
            "during this study session."
        )

    if st.session_state.study_material:
        st.caption("📄 Study material: Loaded")
    else:
        st.caption("📄 Study material: Not loaded")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.memory = []
        st.rerun()


# ============================================================
# HERO SECTION — NATIVE STREAMLIT
# ============================================================

st.info("✦ AI-POWERED LEARNING")

st.title("Study smarter. Understand deeper.")

st.write(
    "Your personal AI tutor for explanations, "
    "practice, questions, and learning."
)

st.divider()


# ============================================================
# FEATURE CARDS — NATIVE STREAMLIT
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.subheader("💡 Understand Concepts")
        st.write(
            "Get difficult topics explained in simple, "
            "student-friendly language."
        )

with col2:
    with st.container(border=True):
        st.subheader("📚 Learn From Your Notes")
        st.write(
            "Upload your study material and ask questions "
            "about what you are learning."
        )

with col3:
    with st.container(border=True):
        st.subheader("🧠 Personalized Tutoring")
        st.write(
            "Your tutor uses the current conversation "
            "and adapts responses to your selected level."
        )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()
st.subheader("💬 Ask Your Tutor")


# Display conversation history

for message in st.session_state.memory:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your studies..."
)


if question:

    # --------------------------------------------------------
    # Display and save user question
    # --------------------------------------------------------

    with st.chat_message("user"):
        st.markdown(question)

    add_message(
        st.session_state.memory,
        "user",
        question,
    )

    # --------------------------------------------------------
    # Retrieve conversation memory
    # --------------------------------------------------------

    conversation = get_memory_text(
        st.session_state.memory
    )

    # --------------------------------------------------------
    # Prepare tutor context
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
    # Generate tutor response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🧠 Your tutor is thinking..."):

            try:
                answer = ask_tutor(
                    question=enhanced_question,
                    study_material=(
                        st.session_state.study_material
                    ),
                    conversation_memory=conversation,
                )

                if not isinstance(answer, str) or not answer.strip():
                    answer = (
                        "I couldn't generate a useful response. "
                        "Please try asking your question again."
                    )

                st.markdown(answer)

            except Exception as e:
                answer = (
                    "I encountered an error while processing "
                    "your question. Please try again."
                )

                st.error(answer)

                with st.expander("Technical details"):
                    st.code(str(e))

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    add_message(
        st.session_state.memory,
        "assistant",
        answer,
    )


# ============================================================
# FOOTER — NATIVE STREAMLIT
# ============================================================

st.divider()

st.caption(
    "Study Tutor AI • Built with Streamlit, CrewAI & Groq"
)
