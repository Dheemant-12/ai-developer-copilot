from memory import (
    load_memory
)

def get_dashboard_stats():

    memory = load_memory()

    stats = {

        "memory_entries": len(
            memory
        ),

        "tools_available": 6,

        "workflows_executed": len(
            memory
        ),

        "system_status": "Online"
    }

    return stats