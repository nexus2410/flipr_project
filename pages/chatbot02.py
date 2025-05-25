import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class GroqChatbot:
    def __init__(self, api_key, model="meta-llama/llama-4-scout-17b-16e-instruct"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def chat(self, messages, temperature=0.7, max_tokens=256):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

def main():
    # Check if user is logged in
    if "user" not in st.session_state:
        st.warning("You must be logged in to access the chatbot. Please log in from the main page.")
        st.stop()

    st.set_page_config(page_title="Customer Support Chat", layout="centered")
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 0;'>
            <h1 style='margin-bottom: 0;'>Customer Support Chat</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_key = os.getenv("GROQ_KEY")
    
    if (
        "messages" not in st.session_state
        or st.session_state.get("api_key") != api_key
        or "chatbot" not in st.session_state
    ):
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state.api_key = api_key
        st.session_state.chatbot = GroqChatbot(api_key)

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div style='background: #e3f2fd; color: #0d47a1; padding:10px 16px; border-radius:12px; margin-bottom:8px; text-align:right;'>
                        <b>You:</b> {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div style='background: #f1f8e9; color: #33691e; padding:10px 16px; border-radius:12px; margin-bottom:8px; text-align:left;'>
                        <b>GroqBot:</b> {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # User input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Type your message:",
            key="input",
            height=60,
            placeholder="Ask me anything...",
        )
        col1, col2 = st.columns([5, 1])
        with col2:
            submitted = st.form_submit_button("➤", use_container_width=True)
        with col1:
            st.write("")  # for alignment

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        with st.spinner("GroqBot is typing..."):
            try:
                reply = st.session_state.chatbot.chat(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Sorry, there was an error: {e}",
                    }
                )
        st.rerun()

    col_clear, col_space = st.columns([1, 5])
    with col_clear:
        if st.button("🗑️", use_container_width=True):
            st.session_state.messages = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.rerun()

if __name__ == "__main__":
    main()
