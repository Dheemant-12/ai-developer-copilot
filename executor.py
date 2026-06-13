def execute_tool(
    tool_name,
    user_query
):

    if tool_name == "code_explainer":

        return (
            "Code Explainer Selected\n\n"
            f"Task: {user_query}"
        )

    elif tool_name == "bug_fixer":

        return (
            "Bug Fix Assistant Selected\n\n"
            f"Task: {user_query}"
        )

    elif tool_name == "readme_generator":

        return (
            "README Generator Selected\n\n"
            f"Task: {user_query}"
        )

    elif tool_name == "repo_analyzer":

        return (
            "Repository Analyzer Selected\n\n"
            f"Task: {user_query}"
        )

    return (
        "General Chat Selected\n\n"
        f"Task: {user_query}"
    )