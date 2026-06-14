def build_reflection_prompt(task):

    return f"""
You are an AI Self Reflection Agent.

Analyze the completed task.

Provide:

Task Summary:
...

What Went Well:
...

Potential Improvements:
...

Future Optimization:
...

Task:

{task}
"""
