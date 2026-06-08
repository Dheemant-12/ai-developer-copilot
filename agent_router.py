def build_router_prompt(query):

    return f"""
You are an AI tool router.

Available tools:

code_explainer
bug_fixer
readme_generator
repo_analyzer
chat

Your job is to choose the BEST tool.

User Query:

{query}

Return ONLY the tool name.

Examples:

Explain this code
code_explainer

Fix this error
bug_fixer

Generate README
readme_generator

Analyze this repository
repo_analyzer

Hello
chat
"""