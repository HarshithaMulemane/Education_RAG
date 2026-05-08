from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List

from graph import edu_rag_graph


app = FastAPI(
    title="Multi-Agent Education RAG API",
    description="LangGraph-based multi-agent RAG application for education domain",
    version="1.0.0",
)


chat_history: Dict[str, List[dict]] = {}


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get("/")
def health_check():
    return {
        "status": "running",
        "message": "Multi-Agent Education RAG API is live"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    recent_history = chat_history.get(request.session_id, [])[-3:]

    history_text = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer']}"
        for item in recent_history
    ])

    initial_state = {
        "question": request.question,
        "chat_history": history_text,
        "route": "",
        "documents": [],
        "answer": "",
        "verified": False,
        "sources": [],
        "standalone_question": "",
        "retrieval_scores": [],
        "confidence": ""
    }

    result = edu_rag_graph.invoke(initial_state)

    if request.session_id not in chat_history:
        chat_history[request.session_id] = []

    chat_history[request.session_id].append({
        "question": request.question,
        "answer": result["answer"],
        "route": result["route"],
        "verified": result["verified"]
    })

    return {
        "session_id": request.session_id,
        "question": result["question"],
        "route": result["route"],
        "answer": result["answer"],
        "verified": result["verified"],
        "sources": result["sources"],
        "standalone_question": result["standalone_question"],
        "retrieval_scores": result["retrieval_scores"],
        "confidence": result["confidence"]
    }


@app.get("/history/{session_id}")
def get_history(session_id: str):
    return {
        "session_id": session_id,
        "history": chat_history.get(session_id, [])
    }