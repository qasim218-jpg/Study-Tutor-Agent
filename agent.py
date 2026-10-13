import os

from crewai import Agent, Task, Crew, LLM

from tools import (
    calculator,
    create_study_material_tool,
)


MODEL = "groq/openai/gpt-oss-120b"


def create_llm():

    return LLM(
        model=MODEL,
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3,
    )


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

Instructions:

1. Answer the student's question clearly.
2. Use simple language.
3. Adapt the explanation to the student's level.
4. Use examples when useful.
5. If the question relates to uploaded study material,
   use the Study Material Search tool.
6. If mathematical calculation is required,
   use the Calculator tool.
7. Do not invent information from the study material.
8. Encourage learning and understanding.
9. Keep the response reasonably concise.

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
