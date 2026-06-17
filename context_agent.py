from memory import (
    search_memory
)

def get_context(
    query
):

    memories = (
        search_memory(
            query
        )
    )

    if not memories:

        return ""

    context = ""

    for item in memories:

        context += (
            f"Previous Query: "
            f"{item.get('query','')}\n"
        )

        context += (
            f"Previous Response: "
            f"{item.get('response','')}\n\n"
        )

    return context
