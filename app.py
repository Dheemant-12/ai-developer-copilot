import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

from database import (
    init_db,
    save_message,
    load_messages,
    clear_messages
)

from pdf_handler import (
    extract_pdf_text
)

from chunking import (
    chunk_text
)

from vector_store import (
    create_vector_store,
    semantic_search
)

# Database
init_db()

# Environment
load_dotenv()

api_key = os.getenv(
    "NVIDIA_API_KEY"
)

# NVIDIA Client
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

    st.subheader("📄 PDF Upload")

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
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

st.caption(
    "FAISS Vector Search Enabled"
)

# PDF
if uploaded_pdf:

    pdf_text = extract_pdf_text(
        uploaded_pdf
    )

    chunks = chunk_text(
        pdf_text
    )

    chunk_count = create_vector_store(
        chunks
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Text Length",
            len(pdf_text)
        )

    with col2:

        st.metric(
            "Chunks Created",
            chunk_count
        )

    with st.expander(
        "📖 View Chunks"
    ):

        for i, chunk in enumerate(chunks):

            st.markdown(
                f"### Chunk {i+1}"
            )

            st.text(
                chunk[:1000]
            )

# Search Section
st.divider()

st.subheader(
    "🔍 Semantic Search"
)

search_query = st.text_input(
    "Ask about uploaded PDF"
)

if search_query:

    results = semantic_search(
        search_query
    )

    st.success(
        f"Found {len(results)} relevant chunks"
    )

    for i, result in enumerate(
        results
    ):

        with st.expander(
            f"Result {i+1}"
        ):

            st.write(
                result
            )

# Session
if "messages" not in st.session_state:

    stored_messages = load_messages()

    st.session_state.messages = []

    for role, content in stored_messages:

        st.session_state.messages.append({
            "role": role,
            "content": content
        })

# Display messages
for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )

# Chat
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

        message_placeholder = st.empty()

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

                    message_placeholder.markdown(
                        full_response + "▌"
                    )

            message_placeholder.markdown(
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