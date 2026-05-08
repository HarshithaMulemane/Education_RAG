import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from rank_bm25 import BM25Okapi

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"


class EduRAGState(TypedDict):
    question: str
    standalone_question: str
    chat_history: str
    route: str
    documents: List[Document]
    answer: str
    verified: bool
    sources: List[str]
    retrieval_scores: List[float]
    confidence: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()


def router_node(state: EduRAGState):
    question = state["standalone_question"].lower()

    if any(word in question for word in ["quiz", "practice", "test me", "mcq"]):
        route = "quiz"
    elif any(word in question for word in ["deadline", "grade", "grading", "policy", "syllabus"]):
        route = "policy"
    else:
        route = "concept"

    return {"route": route}


def retriever_node(state: EduRAGState):
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(
            "Vector store not found. Please run: python ingest.py"
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(
    state["standalone_question"],
    k=5
    )

    semantic_docs = [doc for doc, score in results]

    keyword_docs = keyword_search(
        state["standalone_question"],
        semantic_docs,
        top_k=2
    )

    combined_docs = semantic_docs[:2] + keyword_docs

    # remove duplicates
    unique_docs = []
    seen = set()

    for doc in combined_docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    sources = [doc.page_content[:300] for doc in unique_docs]
    scores = [float(score) for doc, score in results[:len(unique_docs)]]
    avg_score = sum(scores) / len(scores) if scores else 0
    if avg_score < 0.45:
        confidence = "high"
    elif avg_score < 0.75:
        confidence = "medium"
    else:
        confidence = "low"

    return {
    "documents": unique_docs,
    "sources": sources,
    "retrieval_scores": scores,
    "confidence": confidence
}


def answer_node(state: EduRAGState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
You are an AI teaching assistant.

Answer the student's question using ONLY the course context below.

Rules:
- Do not use outside knowledge.
- Do not add details that are not present in the course context.
- Keep the answer concise and grounded.
- If the answer is not present in the context, say:
"I could not find enough information in the course material to answer this confidently."

Course context:
{context}

Student question:
{state["standalone_question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def quiz_node(state: EduRAGState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
You are an AI teaching assistant.

Create quiz questions using ONLY the exact facts stated in the course context.

Strict rules:
- Generate 4 direct quiz questions.
- Do not create calculation questions.
- Do not create comparison questions.
- Do not infer anything.
- Every correct answer must appear exactly in the course context.
- Every question must include a correct answer line using this format: Correct Answer: ...
- If you are about to create a question using any forbidden word, skip that question.

Forbidden words:
combined, total, least, highest, more, less, sum, difference, compare, calculate

Course context:
{context}

Student request:
{state["standalone_question"]}

Quiz:
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


def verifier_node(state: EduRAGState):
    context = "\n\n".join(doc.page_content for doc in state["documents"])

    prompt = f"""
You are a strict RAG verifier.

Your job is to check whether the answer is fully supported by the course context.

Rules:
- Return YES only if every factual claim in the answer is directly supported by the context.
- Return NO if the answer adds extra details not present in the context.
- Return NO if the answer uses outside knowledge.
- Return only YES or NO.

Course context:
{context}

Answer:
{state["answer"]}

Is the answer fully supported?
"""

    response = llm.invoke(prompt)
    verified = "yes" in response.content.lower()

    return {"verified": verified}

def query_rewriter_node(state: EduRAGState):
    if not state.get("chat_history"):
        return {"standalone_question": state["question"]}

    prompt = f"""
You are a query rewriting assistant.

Given the conversation history and the current question, rewrite the current question into a standalone question.

Important:
- If the current question uses words like it, this, that, they, or explain again, resolve them from the most recent user/assistant exchange.
- Preserve the exact topic from the previous exchange.
- Do not switch topics.
- Return only the standalone question.

Conversation history:
{state["chat_history"]}

Current question:
{state["question"]}

Standalone question:
"""

    response = llm.invoke(prompt)

    return {"standalone_question": response.content.strip()}
    


def route_after_retrieval(state: EduRAGState):
    if state["route"] == "quiz":
        return "quiz_node"
    return "answer_node"

def keyword_search(query, docs, top_k=2):
    tokenized_docs = [doc.page_content.lower().split() for doc in docs]

    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [docs[i] for i in ranked_indices]

def build_graph():
    graph = StateGraph(EduRAGState)

    graph.add_node("router_node", router_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("answer_node", answer_node)
    graph.add_node("quiz_node", quiz_node)
    graph.add_node("verifier_node", verifier_node)

    graph.add_node("query_rewriter_node", query_rewriter_node)

    graph.set_entry_point("query_rewriter_node")
    graph.add_edge("query_rewriter_node", "router_node")
    graph.add_edge("router_node", "retriever_node")

    graph.add_conditional_edges(
        "retriever_node",
        route_after_retrieval,
        {
            "answer_node": "answer_node",
            "quiz_node": "quiz_node",
        }
    )

    graph.add_edge("answer_node", "verifier_node")
    graph.add_edge("quiz_node", "verifier_node")
    graph.add_edge("verifier_node", END)

    return graph.compile()


edu_rag_graph = build_graph()