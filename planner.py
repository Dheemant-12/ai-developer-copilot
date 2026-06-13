def build_planner_prompt(query):

    return f"""
You are an AI planning agent.

Create a step-by-step plan.

Rules:

- Break the task into logical steps
- Number each step
- Keep steps concise

User Task:

{query}

Return ONLY the plan.
"""