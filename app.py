import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from website_scraper import scrape_website
from github_reader import get_repo_contents,get_file_content
from agent_router import (
    build_router_prompt)
from planner import (
    build_planner_prompt
)
from executor import (
    execute_tool
)
from repo_vector_store import (
    create_repo_vector_store
)
from memory import (
    save_memory,
    get_recent_memory
)
from workflow_agent import (
    execute_workflow
)
from executor import (
    execute_tool
)
from tool_chain import (
    execute_tool_chain
)
from repo_vector_store import (
    create_repo_vector_store
)
from context_agent import (
    get_context
)
from repo_vector_store import (
    create_repo_vector_store,
    repo_semantic_search
)
from reflection import (
    build_reflection_prompt
)
from database import (
    init_db,
    save_message,
    load_messages,
    clear_messages
)
from repo_rag import (
    load_repository_code
)
from pdf_handler import (
    extract_pdf_text
)

from chunking import (
    chunk_text
)

from vector_store import (
    create_vector_store,
    semantic_search,
    load_vector_store
)
from memory import (
    save_memory,
    get_recent_memory
)
from memory import (
    save_memory,
    get_recent_memory,
    search_memory
)
from dashboard import (
    get_dashboard_stats
)
from datetime import datetime
# Database
init_db()

# Load Existing Vector Store
vector_db_loaded = load_vector_store()

# Environment
load_dotenv()

api_key = os.getenv(
    "NVIDIA_API_KEY"
)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Page
st.set_page_config(
    page_title="AI Developer Copilot",
    page_icon="🤖"
)

# Sidebar
with st.sidebar:

    st.title("⚙️ Settings")

    if vector_db_loaded:

        st.success(
            "✅ Saved Vector DB Loaded"
        )

    else:

        st.warning(
            "⚠️ No Saved Vector DB Found"
        )

    selected_model = st.selectbox(
        "Choose Model",
        [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "deepseek-ai/deepseek-r1"
        ]
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.7,
        0.1
    )

    max_tokens = st.slider(
        "Max Tokens",
        256,
        2048,
        1024,
        256
    )

    st.divider()

    uploaded_pdfs = st.file_uploader(
        "📄 Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        clear_messages()

        st.session_state.messages = []

        st.rerun()

# Title
st.title(
    "🤖 AI Developer Copilot"
)
stats = get_dashboard_stats()

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Memory Entries",
        stats["memory_entries"]
    )

with col2:

    st.metric(
        "Tools",
        stats["tools_available"]
    )

with col3:

    st.metric(
        "Workflows",
        stats["workflows_executed"]
    )

with col4:

    st.metric(
        "Status",
        stats["system_status"]
    )

st.divider()
st.caption(
    "RAG Question Answering Enabled"
)

# Multi PDF Processing
if uploaded_pdfs:

    combined_text = ""

    for pdf in uploaded_pdfs:

        pdf_text = extract_pdf_text(
            pdf
        )

        combined_text += (
            pdf_text + "\n\n"
        )

    chunks = chunk_text(
        combined_text
    )

    chunk_count = create_vector_store(
        chunks
    )

    st.success(
        "Vector Database Saved Successfully"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "PDFs",
            len(uploaded_pdfs)
        )

    with col2:

        st.metric(
            "Text Length",
            len(combined_text)
        )

    with col3:

        st.metric(
            "Chunks",
            chunk_count
        )
st.divider()

st.subheader(
    "🌐 Website Scraper"
)

website_url = st.text_input(
    "Enter Website URL"
)

if st.button(
    "Scrape Website"
):

    if website_url:

        with st.spinner(
            "Scraping website..."
        ):

            website_text = scrape_website(
                website_url
            )
            st.session_state.website_text = (
            website_text
            )
            st.session_state.website_chunks = (
                chunk_text(
                    website_text
                )
            )

        st.success(
            "Website Scraped Successfully"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Characters",
                len(website_text)
            )

        with col2:

            st.metric(
                "Words",
                len(
                    website_text.split()
                )
            )

        st.text_area(
            "Website Content Preview",
            website_text[:5000],
            height=300
        )   
    else:
        st.warning(
            "Please enter a valid URL."
        )
if st.button(
            "Build Website Knowledge Base"
        ):
            if "website_chunks" in st.session_state:

                create_vector_store(
                    st.session_state.website_chunks
                )

                st.success(
                    "Website Knowledge Base Created"
                )

            else:

                st.warning(
                    "Please scrape a website first."
                )         
st.divider()
st.subheader(
    "📝 Website Summarizer"
)

if st.button(
    "Generate Summary"
):

    if "website_text" in st.session_state:

        summary_prompt = f"""
Summarize the following website content.

Use this format:

Summary:
...

Key Points:
• ...
• ...
• ...

CONTENT:

{st.session_state.website_text[:8000]}
"""

        with st.spinner(
            "Generating summary..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],

                temperature=0.3,

                max_tokens=800
            )

            summary = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Summary Generated"
        )

        st.write(
            summary
        )

    else:

        st.warning(
            "Please scrape a website first."
        )   

st.divider()

st.subheader(
    "🌐 Ask Questions About Website"
)

website_question = st.text_input(
    "Ask a question about the website"
)

if website_question:

    retrieved_chunks = semantic_search(
        website_question,
        top_k=5
    )

    if retrieved_chunks:

        context = "\n\n".join(
            [
                item["chunk"]
                for item in retrieved_chunks
            ]
        )

        website_prompt = f"""
You are a website assistant.

Answer ONLY using the provided website content.

If the answer is not found,
reply:

I could not find that information on the website.

CONTEXT:

{context}

QUESTION:

{website_question}
"""

        with st.spinner(
            "Searching website..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": website_prompt
                    }
                ],

                temperature=0.3,

                max_tokens=800
            )

            website_answer = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Website Answer"
        )

        st.write(
            website_answer
        )

    else:

        st.warning(
            "No relevant website information found."
        )
st.divider()

st.subheader(
    "🐙 GitHub Repository Reader"
)

repo_url = st.text_input(
    "Enter GitHub Repository URL"
)

if st.button(
    "Load Repository"
):

    if repo_url:

        with st.spinner(
            "Loading repository..."
        ):

            files = get_repo_contents(
                repo_url
            )
            st.session_state.repo_files = (
                files
            )
            repo_code = ""

            for file in files:

                if (
                    file["name"].endswith(".py")
                    or file["name"] == "requirements.txt"
                ):

                    content = get_file_content(
                        file["download_url"]
                    )

                    repo_code += (
                        f"\n\nFILE: {file['name']}\n"
                        f"{content[:5000]}"
                    )

            st.session_state.repo_code = (
                repo_code
            )

        st.success(
            "Repository Loaded"
        )

        st.metric(
            "Files Found",
            len(files)
        )

        for file in files:

            st.write(
                f"📄 {file['name']}"
            )

    else:

        st.warning(
            "Please enter a repository URL."
        )
st.divider()
st.subheader(
    "📚 Repository RAG"
)

repo_rag_url = st.text_input(
    "Repository URL"
)

if st.button(
    "Build Repository Knowledge Base"
):

    if repo_rag_url:

        with st.spinner(
            "Loading repository..."
        ):

            repo_text = (
                load_repository_code(
                    repo_rag_url
                )
            )
            repo_chunks = chunk_text(

                repo_text
            )

            repo_chunk_count = (
                create_repo_vector_store(
                    repo_chunks
                )
            )
        st.success(
            "Repository Loaded"
        )
        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Characters",
                len(repo_text)
            )

        with col2:

            st.metric(
                "Chunks",
                repo_chunk_count
            )

        

        st.text_area(
            "Repository Preview",
            repo_text[:5000],
            height=300
        )

        
        st.session_state.repo_chunks = (
            repo_chunks
        )

    else:

        st.warning(
            "Enter repository URL."
        )

st.divider()

st.subheader(
    "💻 Ask Questions About Repository"
)

repo_query = st.text_input(
    "Ask about the repository"
)

if repo_query:

    retrieved_chunks = (
        repo_semantic_search(
            repo_query,
            top_k=5
        )
    )

    if retrieved_chunks:

        context = "\n\n".join(

            [
                item["chunk"]

                for item in
                retrieved_chunks
            ]
        )

        prompt = f"""
You are an expert codebase assistant.

Use ONLY the provided
repository context.

If the answer is not found,
say:

I could not find that
information in the repository.

CONTEXT:

{context}

QUESTION:

{repo_query}
"""

        with st.spinner(
            "Searching repository..."
        ):

            response = (
                client.chat.completions.create(

                    model=selected_model,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.2,

                    max_tokens=1024
                )
            )

        answer = (
            response
            .choices[0]
            .message.content
        )

        st.success(
            "Repository Answer"
        )

        st.write(
            answer
        )

    else:

        st.warning(
            "No repository data found."
        )
st.subheader(
    "🏗️ Repository Architecture Analysis"
)

if st.button(
    "Analyze Repository"
):

    if "repo_files" in st.session_state:

        file_names = "\n".join(
            [
                file["name"]
                for file in st.session_state.repo_files
            ]
        )

        repo_prompt = f"""
Analyze this repository.

Provide:

Project Type:
...

Purpose:
...

Main Components:
...

Architecture Flow:
...

FILES:

{file_names}
"""

        with st.spinner(
            "Analyzing repository..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": repo_prompt
                    }
                ],

                temperature=0.3,

                max_tokens=1000
            )

            analysis = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Repository Analysis Complete"
        )

        st.write(
            analysis
        )

    else:

        st.warning(
            "Please load a repository first."
        )
st.divider()

st.subheader(
    "📄 README Generator"
)

if st.button(
    "Generate README"
):

    if "repo_files" in st.session_state:

        file_names = "\n".join(
            [
                file["name"]
                for file in st.session_state.repo_files
            ]
        )

        readme_prompt = f"""
You are a senior software engineer.

Generate a professional GitHub README.

Analyze the repository code.

Include:

# Project Name

## Overview

## Features

## Technologies Used

## Installation

## Usage

## Architecture

## Future Improvements

REPOSITORY CODE:

{st.session_state.repo_code[:15000]}
"""

        with st.spinner(
            "Generating README..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": readme_prompt
                    }
                ],

                temperature=0.4,

                max_tokens=1500
            )

            generated_readme = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "README Generated"
        )

        st.markdown(
            generated_readme
        )

        st.download_button(
            "Download README.md",
            generated_readme,
            file_name="README.md"
        )

    else:

        st.warning(
            "Please load a repository first."
        )
st.divider()

st.subheader(
    "🧠 Advanced Code Explainer"
)

code_input = st.text_area(
    "Paste Python Code",
    height=300
)

if st.button(
    "Explain Code"
):

    if code_input:

        explain_prompt = f"""
You are a senior software engineer.

Analyze the following code.

Provide:

Purpose:
...

Step-by-Step Explanation:
...

Time Complexity:
...

Space Complexity:
...

Potential Issues:
...

Best Practices:
...

Refactored Version:
...

CODE:

{code_input}
"""

        with st.spinner(
            "Analyzing code..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": explain_prompt
                    }
                ],

                temperature=0.2,

                max_tokens=2000
            )

            explanation = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Code Analysis Complete"
        )

        st.markdown(
            explanation
        )

    else:

        st.warning(
            "Please enter some code."
        )
st.divider()

st.subheader(
    "🐞 Advanced Bug Fix Assistant"
)

bug_code = st.text_area(
    "Paste Buggy Code",
    height=300
)

if st.button(
    "Analyze Bug"
):

    if bug_code:

        bug_prompt = f"""
You are a senior software engineer.

Analyze the following code.

Provide:

Bug Type:
...

Root Cause:
...

Severity:
Low / Medium / High

Fix Recommendation:
...

Corrected Code:
...

Best Practices:
...

CODE:

{bug_code}
"""

        with st.spinner(
            "Analyzing bug..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": bug_prompt
                    }
                ],

                temperature=0.2,

                max_tokens=2000
            )

            bug_report = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Bug Analysis Complete"
        )

        st.markdown(
            bug_report
        )

    else:

        st.warning(
            "Please enter code."
        )        
# RAG Question Section
st.divider()
st.subheader(
    "📚 Ask Questions About PDFs"
)

rag_query = st.text_input(
    "Ask a question about your documents"
)

if rag_query:

    with st.spinner(
        "Optimizing query..."
    ):

        rewrite_prompt = f"""
Rewrite the following user query into a more detailed
and searchable query for document retrieval.

Return ONLY the rewritten query.

User Query:
{rag_query}
"""

        rewrite_response = client.chat.completions.create(

            model=selected_model,

            messages=[
                {
                    "role": "user",
                    "content": rewrite_prompt
                }
            ],

            temperature=0.2,

            max_tokens=100
        )

        optimized_query = (
            rewrite_response
            .choices[0]
            .message.content
            .strip()
        )

    st.info(
        f"Optimized Query: {optimized_query}"
    )

    retrieved_chunks = semantic_search(
        optimized_query,
        top_k=5
    )

    if retrieved_chunks:

        avg_confidence = round(
            sum(
                item["confidence"]
                for item in retrieved_chunks
            ) / len(retrieved_chunks),
            2
        )

        top_chunks = retrieved_chunks[:3]

        context = "\n\n".join(
            [
                f"Chunk {i + 1}:\n{item['chunk']}"
                for i, item in enumerate(
                    top_chunks
                )
            ]
        )

        prompt = f"""
You are an expert document assistant.

Use ONLY the provided context.

Answer using this format:

Summary:
...

Key Points:
• ...
• ...
• ...

Conclusion:
...

If the answer is not found in the context,
reply exactly:

I could not find that information in the documents.

CONTEXT:

{context}

QUESTION:

{rag_query}
"""

        with st.spinner(
            "Generating answer..."
        ):

            response = client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,

                max_tokens=1024
            )

            answer = (
                response
                .choices[0]
                .message.content
            )

        st.success(
            "Answer"
        )

        st.write(
            answer
        )

        retrieved_count = len(
            retrieved_chunks
        )

        used_count = len(
            top_chunks
        )

        coverage = round(
            (used_count / retrieved_count) * 100,
            2
        )

        answer_length = len(
            answer.split()
        )

        rag_score = round(
            (avg_confidence * 0.7) +
            (coverage * 0.3),
            2
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:

            st.metric(
                "Confidence Score",
                f"{avg_confidence}%"
            )

        with metric_col2:

            st.metric(
                "Retrieved Chunks",
                retrieved_count
            )

        with metric_col3:

            st.metric(
                "Used Chunks",
                used_count
            )

        metric_col4, metric_col5, metric_col6 = st.columns(3)

        with metric_col4:

            st.metric(
                "Coverage",
                f"{coverage}%"
            )

        with metric_col5:

            st.metric(
                "Answer Length",
                f"{answer_length} words"
            )

        with metric_col6:

            st.metric(
                "RAG Score",
                f"{rag_score}%"
            )

        with st.expander(
            "📄 Sources Used"
        ):

            for item in retrieved_chunks:

                st.markdown(
                    f"## Source Chunk {item['chunk_id']}"
                )

                source_col1, source_col2 = st.columns(2)

                with source_col1:

                    st.metric(
                        "Confidence",
                        f"{item['confidence']}%"
                    )

                with source_col2:

                    st.metric(
                        "Distance",
                        round(
                            item["distance"],
                            2
                        )
                    )

                st.write(
                    item["chunk"]
                )

                st.divider()

    else:

        st.warning(
            "No relevant chunks found."
        )
st.divider()

st.subheader(
    "🤖 Agent Playground"
)

agent_query = st.text_input(
    "Ask the Agent"
)

if agent_query:

    router_prompt = build_router_prompt(
        agent_query
    )

    with st.spinner(
        "Selecting best tool..."
    ):

        response = client.chat.completions.create(

            model=selected_model,

            messages=[
                {
                    "role": "user",
                    "content": router_prompt
                }
            ],

            temperature=0,

            max_tokens=20
        )

        task = (
            response
            .choices[0]
            .message.content
            .strip()
            .lower()
        )
        agent_response = (
            f"Selected Tool: {task}"
        )

        save_memory(
            agent_query,
            task,
            agent_response
        )
    result = execute_tool(
    task,
    agent_query
)

    st.success(
        f"Selected Tool: {task}"
    )
    execution_result = (
        execute_tool(
            tool,
            agent_query
        )
    )

    st.info(
        execution_result
    )

    st.info(
        result
    )
st.divider()

st.subheader(
    "🧩 Multi-Step Planning Agent"
)

planning_query = st.text_input(
    "Enter Complex Task"
)

if planning_query:

    planner_prompt = (
        build_planner_prompt(
            planning_query
        )
    )

    with st.spinner(
        "Creating plan..."
    ):

        response = client.chat.completions.create(

            model=selected_model,

            messages=[
                {
                    "role": "user",
                    "content": planner_prompt
                }
            ],

            temperature=0.2,

            max_tokens=500
        )

        plan = (
            response
            .choices[0]
            .message.content
        )

    st.success(
        "Plan Generated"
    )

    st.markdown(
        plan
    )
st.divider()

st.subheader(
    "🚀 Autonomous Workflow Agent"
)

workflow_query = st.text_input(
    "Enter Workflow Task"
)

if workflow_query:

    workflow = execute_workflow(
        workflow_query
    )

    st.success(
        "Workflow Started"
    )

    for i, step in enumerate(
        workflow,
        start=1
    ):

        st.write(
            f"Step {i} ✅ {step}"
        )

    st.success(
        "Workflow Complete"
    )
st.divider()
st.divider()

st.subheader(
    "🔗 Tool Chaining Agent"
)

chain_query = st.text_input(
    "Enter Multi-Tool Task"
)

if chain_query:

    chain_steps = (
        execute_tool_chain(
            chain_query
        )
    )

    st.success(
        "Tool Chain Created"
    )

    for i, step in enumerate(
        chain_steps,
        start=1
    ):

        st.write(
            f"Step {i}: {step}"
        )

    st.success(
        "Chain Complete"
    )

st.subheader(
    "🪞 Self Reflection Agent"
)

reflection_task = st.text_input(
    "Enter Completed Task"
)

if reflection_task:

    reflection_prompt = (
        build_reflection_prompt(
            reflection_task
        )
    )

    with st.spinner(
        "Reflecting..."
    ):

        response = client.chat.completions.create(

            model=selected_model,

            messages=[
                {
                    "role": "user",
                    "content": reflection_prompt
                }
            ],

            temperature=0.2,

            max_tokens=500
        )

        reflection = (
            response
            .choices[0]
            .message.content
        )

    st.success(
        "Reflection Complete"
    )

    st.markdown(
        reflection
    )    
st.subheader(
    "🧠 Agent Memory"
)

recent_memory = get_recent_memory()

for item in reversed(
    recent_memory
):

    with st.expander(
        item["query"]
    ):

        st.write(
            f"Time: {item['timestamp']}"
        )

        st.write(
            f"Tool: {item['tool']}"
        )

        st.write(
            f"Response: {item['response']}"
        )

st.divider()
st.subheader(
    "🔍 Memory Retrieval Agent"
)

memory_query = st.text_input(
    "Search Agent Memory"
)

if memory_query:

    results = (
        search_memory(
            memory_query
        )
    )

    if results:

        st.success(
            f"Found {len(results)} memories"
        )

        for item in results:

            with st.expander(
                item.get(
                    "query",
                    "Unknown Query"
                )
            ):

                st.write(
                    f"Tool: "
                    f"{item.get('selected_tool', 'Unknown')}"
                )

                st.write(
                    f"Response: "
                    f"{item.get('response', 'No response stored')}"
                )

    else:

        st.warning(
            "No matching memories found."
        )
st.divider()

st.subheader(
    "🧠 Context-Aware Agent"
)

context_query = st.text_input(
    "Ask Context-Aware Agent"
)

if context_query:

    context = (
        get_context(
            context_query
        )
    )

    prompt = f"""
Use previous memory if relevant.

MEMORY:

{context}

CURRENT QUERY:

{context_query}

Provide a helpful answer.
"""

    with st.spinner(
        "Thinking..."
    ):

        response = (
            client.chat.completions.create(

                model=selected_model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,

                max_tokens=512
            )
        )

    answer = (
        response
        .choices[0]
        .message.content
    )

    st.success(
        "Context-Aware Response"
    )

    st.write(
        answer
    )

    if context:

        with st.expander(
            "Memory Used"
        ):

            st.write(
                context
            )
st.subheader(
    "💬 AI Chat Assistant"
)

#Session state 
if "messages" not in st.session_state:

    stored_messages = load_messages()

    st.session_state.messages = []

    for role, content in stored_messages:

        st.session_state.messages.append({
            "role": role,
            "content": content
        })

# Display Chat
for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

# Normal Chat
prompt = st.chat_input(
    "Ask anything..."
)

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    save_message(
        "user",
        prompt
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )

    with st.chat_message(
        "assistant"
    ):

        placeholder = st.empty()

        full_response = ""

        try:

            completion = client.chat.completions.create(

                model=selected_model,

                messages=st.session_state.messages,

                temperature=temperature,

                max_tokens=max_tokens,

                stream=True
            )

            time.sleep(0.3)

            for chunk in completion:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:

                    full_response += (
                        delta.content
                    )

                    placeholder.markdown(
                        full_response + "▌"
                    )

            placeholder.markdown(
                full_response
            )

        except Exception as e:

            st.error(
                str(e)
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    save_message(
        "assistant",
        full_response
    )