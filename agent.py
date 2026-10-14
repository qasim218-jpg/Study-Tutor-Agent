import os

from crewai import Agent, Task, Crew, LLM

from tools import calculator, create_study_material_tool


MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def create_llm():
    """Initialize the Groq language model."""

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please configure it in Render Environment settings."
        )

    return LLM(
        model=MODEL,
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        temperature=0.3,
    )


def create_study_tutor(study_material=""):

    llm = create_llm()

    study_tool = create_study_material_tool(study_material)

    tutor = Agent(
        role="Study Tutor",
        goal=(
            "Help students understand academic concepts, "
            "practice what they learn, and improve their knowledge."
        ),
        backstory=(
            "You are a patient and knowledgeable personal tutor. "
            "You explain difficult concepts in simple language. "
            "You adapt explanations to the student's level, use "
            "examples when helpful, and encourage independent thinking."
        ),
        llm=llm,
        tools=[calculator, study_tool],
        verbose=True,
        allow_delegation=False,
    )

    return tutor


def ask_tutor(
    question,
    study_material="",
    conversation_memory=""
):

    tutor = create_study_tutor(study_material)

    task_description = f"""
You are helping a student.

STUDENT QUESTION:
{question}

PREVIOUS CONVERSATION:
{conversation_memory}

Instructions:
1. Answer the student's question clearly.
2. Use simple language and adapt to the student's level.
3. Give examples when useful.
4. Use the Study Material Search tool when relevant.
5. Use the Calculator tool when mathematical calculations are needed.
6. Do not invent information from uploaded study material.
7. Use previous conversation context when relevant.
8. Encourage learning and understanding.
9. Keep the response reasonably concise.
10. If the uploaded material does not contain the answer,
    explain this clearly when relevant.
"""

    task = Task(
        description=task_description,
        expected_output=(
            "A clear, accurate and student-friendly tutoring response."
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
