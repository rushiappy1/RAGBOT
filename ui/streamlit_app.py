import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/ask"

def ask_backend(question: str):
    try:
        res = requests.post(BACKEND_URL, json={"question": question})
        res.raise_for_status()
        return res.json().get("answer", "No answer returned")
    except Exception as e:
        st.error(f"Backend error: {e}")
        return "Backend unreachable"

def main():
    st.set_page_config(page_title="Insurance Q&A", layout="centered")
    st.title("💬 Insurance Q&A Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display message history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑‍💻 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

    # Use a form to handle input and auto-clear
    with st.form(key="question_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key="input_box")
        submit_button = st.form_submit_button("Send")

    # Process form submission
    if submit_button and user_input.strip():
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Fetch backend answer
        with st.spinner("Thinking..."):
            answer = ask_backend(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Rerun to show the new messages
        st.rerun()

if __name__ == "__main__":
    main()
