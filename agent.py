def route_task(user_input):

    text = user_input.lower()

    if (
        "bug" in text
        or "error" in text
        or "exception" in text
        or "fix" in text
    ):

        return "bug_fixer"

    if (
        "code" in text
        or "function" in text
        or "algorithm" in text
        or "explain" in text
        or "analyze" in text
    ):

        return "code_explainer"

    if (
        "readme" in text
        or "documentation" in text
    ):

        return "readme_generator"

    if (
        "repository" in text
        or "github" in text
        or "repo" in text
    ):

        return "repo_analyzer"

    return "chat"