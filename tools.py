from crewai.tools import tool


@tool("Calculator")
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Example:
    25 * 18
    100 / 4
    2 ** 10
    """

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception:
        return "I could not calculate that expression."


@tool("Study Material Search")
def search_study_material(query: str, study_material: str) -> str:
    """
    Search the uploaded study material for information
    relevant to the student's question.
    """

    if not study_material:
        return "No study material has been uploaded."

    query_words = query.lower().split()

    paragraphs = study_material.split("\n")

    relevant = []

    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()

        matches = sum(
            1 for word in query_words
            if word in paragraph_lower
        )

        if matches > 0:
            relevant.append(paragraph)

    if not relevant:
        return "I could not find relevant information in the uploaded study material."

    return "\n".join(relevant[:5])
