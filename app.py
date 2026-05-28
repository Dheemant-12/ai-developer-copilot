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

# Page configuration
st.set_page_config(
    page_title="AI Developer Copilot",
    page_icon="🤖",
    layout="centered"
)

# Main title
st.title("🤖 AI Developer Copilot")

# Subtitle
st.caption("Powered by NVIDIA Llama 3.1")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
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

        # Placeholder for streamed text
        message_placeholder = st.empty()

        # Temporary loading text
        message_placeholder.markdown("⚡ Generating response...")

        # Final response storage
        full_response = ""

        try:

            # Create streaming completion
            completion = client.chat.completions.create(

                model="meta/llama-3.1-8b-instruct",

                messages=st.session_state.messages,

                temperature=0.7,

                max_tokens=1024,

                stream=True
            )

            # Small delay for smoother UX
            time.sleep(0.3)

            # Stream tokens live
            for chunk in completion:

                # Skip empty chunks
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Append token if available
                if delta.content:

                    full_response += delta.content

                    # Live typing effect
                    message_placeholder.markdown(
                        full_response + "▌"
                    )

            # Final response without cursor
            message_placeholder.markdown(full_response)

        except Exception as e:

            st.error(f"Error: {str(e)}")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })