import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(
    page_title="Education Multi-Agent RAG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Multi-Agent Education RAG")
st.caption("LangGraph + FastAPI + FAISS + BM25 + Streamlit")


if "session_id" not in st.session_state:
    st.session_state.session_id = "student_001"

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Settings")

    st.session_state.session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Example Questions")
    st.markdown("""
    - Explain gradient descent
    - Explain it in simple terms
    - Create 5 quiz questions
    - What is the grading policy?
    - What is the assignment deadline?
    """)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if "metadata" in message:
                metadata = message["metadata"]

                with st.expander("View RAG Details"):
                    st.write("**Route:**", metadata.get("route"))
                    st.write("**Standalone Question:**", metadata.get("standalone_question"))
                    st.write("**Verified:**", metadata.get("verified"))
                    st.write("**Confidence:**", metadata.get("confidence"))
                    st.write("**Retrieval Scores:**", metadata.get("retrieval_scores"))

                    st.markdown("### Sources")
                    for i, source in enumerate(metadata.get("sources", []), start=1):
                        st.markdown(f"**Source {i}:**")
                        st.info(source)


user_question = st.chat_input("Ask a question from your course material...")

if user_question:
    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.markdown(user_question)

    payload = {
        "session_id": st.session_state.session_id,
        "question": user_question
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking through LangGraph agents..."):
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()

                answer = result["answer"]

                st.markdown(answer)

                with st.expander("View RAG Details"):
                    st.write("**Route:**", result.get("route"))
                    st.write("**Standalone Question:**", result.get("standalone_question"))
                    st.write("**Verified:**", result.get("verified"))
                    st.write("**Confidence:**", result.get("confidence"))
                    st.write("**Retrieval Scores:**", result.get("retrieval_scores"))

                    st.markdown("### Sources")
                    for i, source in enumerate(result.get("sources", []), start=1):
                        st.markdown(f"**Source {i}:**")
                        st.info(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": result
                })

            except requests.exceptions.ConnectionError:
                st.error("FastAPI backend is not running. Start it using: uvicorn app:app --reload")

            except Exception as e:
                st.error(f"Error: {e}")