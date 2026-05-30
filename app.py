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

# Initialize database
init_db()

# Load environment variables
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

# NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Page config
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

    if st.button("🗑️ Clear Chat"):

        clear_messages()

        st.session_state.messages = []

        st.rerun()

# Title
st.title("🤖 AI Developer Copilot")

st.caption(
    "Now with persistent chat history"
)

# Initialize session state
if "messages" not in st.session_state:

    stored_messages = load_messages()

    st.session_state.messages = []

    for role, content in stored_messages:

        st.session_state.messages.append({
            "role": role,
            "content": content
        })

# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# Input
prompt = st.chat_input(
    "Ask anything..."
)

if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    save_message("user", prompt)

    with st.chat_message("user"):

        st.markdown(prompt)

    with st.chat_message("assistant"):

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

                    full_response += delta.content

                    message_placeholder.markdown(
                        full_response + "▌"
                    )

            message_placeholder.markdown(
                full_response
            )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    save_message(
        "assistant",
        full_response
    )