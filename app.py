import hashlib
import io

import streamlit as st
from pypdf import PdfReader

from agent import ask_tutor
from memory import add_message, get_memory_text
from rag import build_index


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

DEFAULT_STATE = {
    "memory": [],
    "study_material": "",
    "pdf_name": "",
    "pdf_hash": "",
    "rag_index": None,
    "pdf_page_count": 0,
    "pdf_error": "",
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# PDF RESET CALLBACK
# ============================================================

def clear_pdf():
    """Remove the active PDF and its semantic index."""

    st.session_state.study_material = ""
    st.session_state.pdf_name = ""
    st.session_state.pdf_hash = ""
    st.session_state.rag_index = None
    st.session_state.pdf_page_count = 0
    st.session_state.pdf_error = ""

    # Clear the uploader before the next script run.
    st.session_state["study_pdf_uploader"] = None

    # Avoid confusing follow-up questions from an old document.
    st.session_state.memory = []


# ============================================================
# PDF PROCESSING
# ============================================================

def process_uploaded_pdf(uploaded_file):
    """
    Extract selectable text, create a semantic index,
    and activate the new document only after indexing succeeds.
    """

    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.sha256(file_bytes).hexdigest()

    # Same document: do not index it again.
    if current_hash == st.session_state.pdf_hash:
        return

    # Prevent accidental use of the previous PDF if the new one fails.
    st.session_state.study_material = ""
    st.session_state.pdf_name = ""
    st.session_state.pdf_hash = ""
    st.session_state.rag_index = None
    st.session_state.pdf_page_count = 0
    st.session_state.pdf_error = ""

    try:
        with st.spinner("Reading your PDF..."):
            reader = PdfReader(io.BytesIO(file_bytes))

            extracted_pages = []

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    extracted_pages.append(page_text)

            extracted_text = "\n".join(extracted_pages).strip()

        if not extracted_text:
            st.session_state.pdf_error = (
                "No selectable text could be extracted. "
                "This document may need OCR."
            )
            return

        with st.spinner(
            "Building the semantic search index. "
            "The first run may take longer while the model loads..."
        ):
            index = build_index(extracted_text)

        # Activate only after indexing succeeds.
        st.session_state.study_material = extracted_text
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_hash = current_hash
        st.session_state.rag_index = index
        st.session_state.pdf_page_count = len(reader.pages)

        # Start a fresh conversation for the new document.
        st.session_state.memory = []

    except Exception as exc:
        st.session_state.study_material = ""
        st.session_state.pdf_name = ""
        st.session_state.pdf_hash = ""
        st.session_state.rag_index = None
        st.session_state.pdf_page_count = 0

        st.session_state.pdf_error = (
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Study Tutor AI")
    st.caption("Your intelligent study companion")

    st.divider()

    st.subheader("📚 Study Material")

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"],
        key="study_pdf_uploader",
        help=(
            "Upload lecture notes, textbook chapters, "
            "or other selectable-text study material."
        ),
    )

    if uploaded_file is not None:
        process_uploaded_pdf(uploaded_file)

    if st.session_state.pdf_error:
        st.error("The PDF could not be prepared.")

        with st.expander("Technical details"):
            st.code(st.session_state.pdf_error)

    if (
        st.session_state.study_material
        and st.session_state.rag_index
    ):
        st.success("PDF indexed and ready")

        st.caption(
            f"Document: {st.session_state.pdf_name}"
        )

        st.caption(
            f"Pages: {st.session_state.pdf_page_count}"
        )

        st.caption(
            "Searchable passages: "
            f"{len(st.session_state.rag_index['chunks']):,}"
        )

        if st.button(
            "🗑️ Remove Study PDF",
            use_container_width=True,
        ):
            clear_pdf()
            st.rerun()

    elif uploaded_file is None:
        st.caption(
            "Upload a study PDF to enable document-based "
            "semantic search."
        )

    st.divider()

    st.subheader("⚙️ Tutor Settings")

    difficulty = st.selectbox(
        "Learning Level",
        ["Beginner", "Intermediate", "Advanced"],
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

    st.caption(
        f"{message_count} messages in this session"
    )

    st.caption(
        "Recent conversation context is limited to help "
        "control token usage."
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.memory = []
        st.rerun()


# ============================================================
# HERO SECTION
# ============================================================

st.info("✦ AI-POWERED LEARNING")

st.title("Study smarter. Understand deeper.")

st.write(
    "Your personal AI tutor for explanations, "
    "practice, questions, and learning."
)

st.divider()


# ============================================================
# FEATURE CARDS
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
            "using semantic search across relevant passages."
        )

with col3:
    with st.container(border=True):
        st.subheader("🧠 Personalized Tutoring")
        st.write(
            "Adapt explanations to your learning level "
            "and preferred teaching style."
        )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()
st.subheader("💬 Ask Your Tutor")

# Display saved conversation.
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

    # Capture previous conversation BEFORE saving the new question.
    conversation = get_memory_text(
        st.session_state.memory
    )

    # Add learning preferences without changing the main UI.
    enhanced_question = f"""
Student learning level: {difficulty}
Preferred teaching style: {response_style}

Student question:
{question}
"""

    # Display and save the user message.
    with st.chat_message("user"):
        st.markdown(question)

    add_message(
        st.session_state.memory,
        "user",
        question,
    )

    # Generate a tutor response using the RAG index.
    with st.chat_message("assistant"):

        with st.spinner("🧠 Your tutor is thinking..."):

            try:
                answer = ask_tutor(
                    question=enhanced_question,
                    study_material=(
                        st.session_state.study_material
                    ),
                    conversation_memory=conversation,
                    rag_index=st.session_state.rag_index,
                )

                if (
                    not isinstance(answer, str)
                    or not answer.strip()
                ):
                    answer = (
                        "I couldn't generate a useful response. "
                        "Please try a more specific question."
                    )

                st.markdown(answer)

            except Exception as exc:
                answer = (
                    "I couldn't process that question. "
                    "Please try again in a moment."
                )

                st.error(answer)

                with st.expander("Technical details"):
                    st.code(
                        f"{type(exc).__name__}: {exc}"
                    )

    # Store the answer for subsequent follow-up questions.
    add_message(
        st.session_state.memory,
        "assistant",
        answer,
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Study Tutor AI • Streamlit • Semantic RAG • Groq"
)
