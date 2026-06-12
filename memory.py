import json
import os
from datetime import datetime

MEMORY_FILE = "memory.json"


def load_memory():

    if not os.path.exists(
        MEMORY_FILE
    ):

        return []

    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


def save_memory(
    query,
    tool,
    response
):

    memory = load_memory()

    memory.append(
        {
            "timestamp": str(
                datetime.now()
            ),
            "query": query,
            "tool": tool,
            "response": response
        }
    )

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            memory,
            file,
            indent=4
        )


def get_recent_memory(
    limit=10
):

    memory = load_memory()

    return memory[-limit:]