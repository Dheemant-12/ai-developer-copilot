import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Get NVIDIA API key
api_key = os.getenv("NVIDIA_API_KEY")

# Initialize NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Page config
st.set_page_config(
    page_title="AI Developer Copilot",
    page_icon="🤖",
    layout="centered"
)

# Sidebar
with st.sidebar:

    st.title("⚙️ Settings")

    # Model selection
    selected_model = st.selectbox(
        "Choose Model",
        [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "deepseek-ai/deepseek-r1"
        ]
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    # Max tokens slider
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256
    )

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# Main title
st.title("🤖 AI Developer Copilot")

# Subtitle
st.caption("Streaming AI Assistant powered by NVIDIA")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask anything...")

# If user enters prompt
if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):

        # Placeholder
        message_placeholder = st.empty()

        # Loading message
        message_placeholder.markdown("⚡ Generating response...")

        # Full response storage
        full_response = ""

        try:

            # Streaming response
            completion = client.chat.completions.create(

                model=selected_model,

                messages=st.session_state.messages,

                temperature=temperature,

                max_tokens=max_tokens,

                stream=True
            )

            # Smooth UX
            time.sleep(0.3)

            # Stream tokens
            for chunk in completion:

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:

                    full_response += delta.content

                    message_placeholder.markdown(
                        full_response + "▌"
                    )

            # Final response
            message_placeholder.markdown(full_response)

        except Exception as e:

            st.error(f"Error: {str(e)}")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })