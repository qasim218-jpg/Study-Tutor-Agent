import hashlib
import io

import streamlit as st
from pypdf import PdfReader

from agent import ask_tutor
from memory import add_message, get_memory_text
from rag import build_index


# ============================================================
# CONFIGURATION
# ============================================================

MAX_PDF_SIZE_MB = 15
MAX_PDF_CHARS = 600_000
MAX_CHAT_MESSAGES = 30

st.set_page_config(
    page_title="Study Tutor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DARK UI
# ============================================================

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }

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

    p, li, label { color: #e2e8f0; }

    [data-testid="stSidebar"] {
        background: #101b2d;
        border-right: 1px solid #273850;
    }

    [data-testid="stSidebar"] * { color: #edf4ff; }

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

    [data-testid="stAlert"] p { color: inherit; }

    hr { border-color: #2b3b52 !important; }

    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible {
        outline: 2px solid #67e8f9 !important;
        outline-offset: 2px !important;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem 2rem; }
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
# MEMORY MANAGEMENT
# ============================================================

def trim_memory():
    """Keep only recent messages to limit session memory."""
    if len(st.session_state.memory) > MAX_CHAT_MESSAGES:
        st.session_state.memory = (
            st.session_state.memory[-MAX_CHAT_MESSAGES:]
        )


def clear_conversation():
    """Clear chat history without removing the PDF."""
    st.session_state.memory = []


# ============================================================
# PDF RESET CALLBACK
# ============================================================

def clear_pdf():
    """
    Clear the document and its index.
    Called by Streamlit before the next script rerun.
    """

    st.session_state.study_material = ""
    st.session_state.pdf_name = ""
    st.session_state.pdf_hash = ""
    st.session_state.rag_index = None
    st.session_state.pdf_page_count = 0
    st.session_state.pdf_error = ""
    st.session_state.memory = []

    # Reset the uploader through the callback.
    st.session_state["study_pdf_uploader"] = None


# ============================================================
# PDF PROCESSING
# ============================================================

def process_uploaded_pdf(uploaded_file):
    """
    Extract selectable PDF text with a size limit, build a
    lightweight BM25 index, and avoid storing a duplicate
    full-text copy in session state.
    """

    if uploaded_file is None:
        return

    # Check file size before reading or parsing the PDF.
    if uploaded_file.size > MAX_PDF_SIZE_MB * 1024 * 1024:
        st.session_state.pdf_error = (
            f"File is too large. Please upload a PDF smaller "
            f"than {MAX_PDF_SIZE_MB} MB."
        )
        return

    try:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(file_bytes).hexdigest()

        # Avoid rebuilding the index for the same file.
        if current_hash == st.session_state.pdf_hash:
            del file_bytes
            return

        # Remove the previous document before processing a new one.
        st.session_state.study_material = ""
        st.session_state.pdf_name = ""
        st.session_state.pdf_hash = ""
        st.session_state.rag_index = None
        st.session_state.pdf_page_count = 0
        st.session_state.pdf_error = ""
        st.session_state.memory = []

        with st.spinner("Reading your PDF..."):
            reader = PdfReader(io.BytesIO(file_bytes))

            extracted_pages = []
            total_chars = 0
            page_count = len(reader.pages)

            for page in reader.pages:
                if total_chars >= MAX_PDF_CHARS:
                    break

                page_text = page.extract_text() or ""

                if not page_text.strip():
                    continue

                remaining = MAX_PDF_CHARS - total_chars
                page_text = page_text[:remaining]

                extracted_pages.append(page_text)
                total_chars += len(page_text)

            extracted_text = "\n".join(extracted_pages).strip()

            # Release intermediate references as soon as possible.
            del extracted_pages

        if not extracted_text:
            st.session_state.pdf_error = (
                "No selectable text could be extracted from "
                "this PDF. It may be scanned and require OCR."
            )
            del file_bytes
            return

        with st.spinner(
            "Preparing your study material. "
            "Large documents may take a little longer..."
        ):
            index = build_index(extracted_text)

        # Store only a short status string and the search index.
        st.session_state.study_material = (
            "PDF indexed and ready for document-based questions."
        )
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_hash = current_hash
        st.session_state.rag_index = index
        st.session_state.pdf_page_count = page_count

        del extracted_text
        del file_bytes

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
            f"Upload a selectable-text PDF up to "
            f"{MAX_PDF_SIZE_MB} MB."
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
            f"Pages in PDF: {st.session_state.pdf_page_count}"
        )
        st.caption(
            "Searchable passages: "
            f"{len(st.session_state.rag_index['chunks']):,}"
        )

        st.caption(
            "Text processing is limited to the first "
            f"{MAX_PDF_CHARS:,} characters."
        )

        st.button(
            "🗑️ Remove Study PDF",
            key="remove_pdf_button",
            use_container_width=True,
            on_click=clear_pdf,
        )

    elif uploaded_file is None:
        st.caption(
            "Upload a PDF to ask questions about your notes."
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

    st.caption(
        f"{len(st.session_state.memory)} recent messages "
        f"(maximum {MAX_CHAT_MESSAGES})"
    )
    st.caption(
        "Only recent conversation messages are retained "
        "in this session."
    )

    st.button(
        "🗑️ Clear Conversation",
        key="clear_conversation_button",
        use_container_width=True,
        on_click=clear_conversation,
    )


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
            "Upload your study material and retrieve "
            "relevant passages using lightweight text search."
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

for message in st.session_state.memory:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT AND RESPONSE
# ============================================================

question = st.chat_input(
    "Ask anything about your studies..."
)

if question:
    question = question.strip()

    if not question:
        st.stop()

    # Capture earlier context before adding this question.
    conversation = get_memory_text(
        st.session_state.memory
    )

    # Preserve compatibility with the updated agent.py.
    enhanced_question = f"""
Student learning level: {difficulty}
Preferred teaching style: {response_style}

Student question:
{question}
"""

    with st.chat_message("user"):
        st.markdown(question)

    add_message(
        st.session_state.memory,
        "user",
        question,
    )
    trim_memory()

    answer = ""

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

    add_message(
        st.session_state.memory,
        "assistant",
        answer,
    )
    trim_memory()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Study Tutor AI • Streamlit • Lightweight RAG • Groq"
)
