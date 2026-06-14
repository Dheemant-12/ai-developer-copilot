def execute_workflow(task):

    task = task.lower()

    workflow_steps = []

    if (
        "repository" in task
        or "github" in task
    ):

        workflow_steps.append(
            "Repository Analysis"
        )

        workflow_steps.append(
            "README Generation"
        )

        workflow_steps.append(
            "Memory Update"
        )

    elif (
        "code" in task
    ):

        workflow_steps.append(
            "Code Explanation"
        )

        workflow_steps.append(
            "Bug Analysis"
        )

        workflow_steps.append(
            "Memory Update"
        )

    else:

        workflow_steps.append(
            "General Chat"
        )

    return workflow_steps