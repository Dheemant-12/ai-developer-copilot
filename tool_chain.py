def execute_tool_chain(
    task
):

    task = task.lower()

    steps = []

    if (
        "repository" in task
        or "github" in task
    ):

        steps.append(
            "Repository Analysis"
        )

        steps.append(
            "README Generation"
        )

        steps.append(
            "Memory Update"
        )

    elif (
        "code" in task
    ):

        steps.append(
            "Code Explanation"
        )

        steps.append(
            "Bug Fix Analysis"
        )

        steps.append(
            "Memory Update"
        )

    else:

        steps.append(
            "General Chat"
        )

    return steps