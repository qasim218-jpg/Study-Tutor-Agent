def add_message(memory, role, content):
    """
    Add a message to conversation memory.
    """

    memory.append({
        "role": role,
        "content": content
    })


def get_memory_text(memory):
    """
    Convert memory into text that the agent can understand.
    """

    if not memory:
        return "No previous conversation."

    conversation = []

    for message in memory:
        role = message["role"].capitalize()

        conversation.append(
            f"{role}: {message['content']}"
        )

    return "\n".join(conversation)
