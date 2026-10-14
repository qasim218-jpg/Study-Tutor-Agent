```python
import os

from crewai import Agent, Task, Crew, LLM

from tools import (
    calculator,
    create_study_material_tool,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------
# INITIALIZE GROQ LLM
# --------------------------------------------------

def create_llm():
    """
    Initialize the Groq-hosted GPT-OSS-20B model
    using CrewAI's OpenAI-compatible configuration.
    """

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to your environment variables "
            "or Render Environment settings."
        )

    return LLM(
        model=MODEL,
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=0.3,
    )


# --------------------------------------------------
# CREATE STUDY TUTOR AGENT
# --------------------------------------------------

def create_study_tutor(study_material=""):

    llm = create_llm()

    study_tool = create_study_material_tool(
        study_material
    )

    tutor = Agent(
        role="Study Tutor",

        goal=(
            "Help students understand academic concepts, "
            "practice what they learn, and improve their knowledge."
        ),

        backstory=(
            "You are a patient and knowledgeable personal tutor. "
            "You explain difficult concepts in simple language. "
            "You adapt explanations to the student's level. "
            "You use examples and analogies when helpful. "
            "You encourage students to think instead of simply "
            "giving answers."
        ),

        llm=llm,

        tools=[
            calculator,
            study_tool,
        ],

        verbose=True,

        allow_delegation=False,
    )

    return tutor


# --------------------------------------------------
# ASK THE STUDY TUTOR
# --------------------------------------------------

def ask_tutor(
    question,
    study_material="",
    conversation_memory=""
):

    tutor = create_study_tutor(
        study_material
    )

    task_description = f"""
You are helping a student.

STUDENT QUESTION:
{question}

PREVIOUS CONVERSATION:
{conversation_memory}

AVAILABLE STUDY MATERIAL:
The uploaded study material may be available through
the Study Material Search tool.

INSTRUCTIONS:

1. Answer the student's question clearly.
2. Use simple, student-friendly language.
3. Adapt the explanation to the student's level.
4. Use examples and analogies when useful.
5. If the question relates to uploaded study material,
   use the Study Material Search tool.
6. If mathematical calculation is required,
   use the Calculator tool.
7. Do not invent information or claim that the
   uploaded material says something you cannot verify.
8. Use previous conversation context when relevant.
9. Encourage learning and understanding.
10. Keep the response reasonably concise.
11. If the available study material does not contain
    the answer, explain this clearly when relevant.

"""

    task = Task(
        description=task_description,

        expected_output=(
            "A clear, accurate and student-friendly "
            "tutoring response."
        ),

        agent=tutor,
    )

    crew = Crew(
        agents=[tutor],
        tasks=[task],
        verbose=True,
    )

    result = crew.kickoff()

    return str(result)
```
