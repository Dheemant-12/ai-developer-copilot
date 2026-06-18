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

    review = []

    if len(
        result
    ) < 50:

        review.append(
            "Result may be too short."
        )

    if (
        "error"
        in result.lower()
    ):

        review.append(
            "Potential issue detected."
        )

    if not review:

        review.append(
            "Execution looks good."
        )

    review.append(
        "Consider adding more detail."
    )

    return "\n".join(
        review
    )