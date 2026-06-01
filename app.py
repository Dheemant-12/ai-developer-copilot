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

    uploaded_pdf = st.file_uploader(
        "📄 Upload PDF",
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
    "RAG Question Answering Enabled"
)

# PDF Processing
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
            "Chunks",
            chunk_count
        )

# RAG Question Section
st.divider()

st.subheader(
    "📚 Ask Questions About PDF"
)

rag_query = st.text_input(
    "Ask a question about your document"
)

if rag_query:

    retrieved_chunks = semantic_search(
        rag_query
    )

    if retrieved_chunks:

        answer = ""

        context = "\n\n".join(
        [
            item["chunk"]
            for item in retrieved_chunks
        ]
    )

        prompt = f"""
You are a document assistant.

Answer ONLY using the context below.

If the answer is not found,
say:

'I could not find that information in the document.'

CONTEXT:

{context}

QUESTION:

{rag_query}
"""

        with st.spinner(
            "Searching document..."
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

        avg_confidence = round(
            sum(
                item["confidence"]
                for item in retrieved_chunks
            ) / len(retrieved_chunks),
            2
        )

        st.metric(
            "Confidence Score",
            f"{avg_confidence}%"
        )

        with st.expander(
            "📄 Sources Used"
        ):

            for item in retrieved_chunks:

                st.markdown(
                    f"## Source Chunk {item['chunk_id']}"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Confidence",
                        f"{item['confidence']}%"
                    )

                with col2:

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
# Session State
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