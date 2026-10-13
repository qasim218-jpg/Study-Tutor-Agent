import streamlit as st

from pypdf import PdfReader

from agent import ask_tutor
from memory import add_message, get_memory_text


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Study Tutor Agent",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------
# Session Memory
# --------------------------------

if "memory" not in st.session_state:
    st.session_state.memory = []


if "study_material" not in st.session_state:
    st.session_state.study_material = ""


# --------------------------------
# Title
# --------------------------------

st.title("🎓 Study Tutor Agent")

st.write(
    "Your personal AI tutor powered by CrewAI and Groq."
)


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.header("📚 Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        try:

            reader = PdfReader(uploaded_file)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            st.session_state.study_material = text

            st.success(
                f"Loaded {len(reader.pages)} pages."
            )

        except Exception as e:

            st.error(
                f"Could not read PDF: {e}"
            )


    st.divider()

    st.header("🧠 Memory")

    if st.session_state.memory:

        st.write(
            f"{len(st.session_state.memory)} messages "
            "in current session."
        )

    else:

        st.write("No conversation yet.")


    if st.button("🗑️ Clear Memory"):

        st.session_state.memory = []

        st.rerun()


# --------------------------------
# Display Previous Conversation
# --------------------------------

for message in st.session_state.memory:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# --------------------------------
# Chat Input
# --------------------------------

question = st.chat_input(
    "Ask your tutor something..."
)


if question:

    # Show user message

    with st.chat_message("user"):

        st.markdown(question)


    # Save user message

    add_message(
        st.session_state.memory,
        "user",
        question
    )


    # Get previous conversation

    conversation = get_memory_text(
        st.session_state.memory
    )


    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner("Tutor is thinking..."):

            try:

                answer = ask_tutor(
                    question=question,

                    study_material=(
                        st.session_state.study_material
                    ),

                    conversation_memory=conversation
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, something went wrong.\n\n"
                    f"Error: {e}"
                )

                st.error(answer)


    # Save assistant response

    add_message(
        st.session_state.memory,
        "assistant",
        answer
    )
