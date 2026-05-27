import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("NVIDIA_API_KEY")

# Initialize NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# App title
st.title("NVIDIA AI Chatbot")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
prompt = st.chat_input("Ask anything...")

# When user sends message
if prompt:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate AI response
    completion = client.chat.completions.create(

        model="meta/llama-3.1-8b-instruct",

        messages=st.session_state.messages,

        temperature=0.7,

        max_tokens=1024
    )

    # Extract response
    reply = completion.choices[0].message.content

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(reply)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })