def planner_agent(
    task
):

    return (
        f"Plan:\n"
        f"1. Analyze task\n"
        f"2. Execute solution\n"
        f"3. Review result\n\n"
        f"Task: {task}"
    )


def executor_agent(
    task
):

    return (
        f"Execution Result:\n"
        f"Processed task:\n"
        f"{task}"
    )


def reviewer_agent(
    task,
    result
):

    return (
        f"Review:\n"
        f"Execution completed "
        f"successfully.\n\n"
        f"Task:\n{task}"
    )