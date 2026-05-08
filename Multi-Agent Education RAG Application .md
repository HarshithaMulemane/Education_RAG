# Multi-Agent Education RAG Application using LangGraph

## 1. Project Overview

This project is a **Multi-Agent Retrieval-Augmented Generation (RAG) application** built for the education domain using **LangGraph**, **FastAPI**, **FAISS**, **BM25**, and **Streamlit**.

The application acts like an AI Teaching Assistant. It can:

- Answer questions from course material
- Explain educational concepts
- Generate grounded quiz questions
- Answer syllabus, grading, and deadline-related questions
- Handle conversational follow-up questions
- Return retrieved sources, confidence scores, and verification status

Instead of using a single RAG chain, this project uses multiple specialized agents connected through a LangGraph workflow.

---

## 2. Use Case

Students often need quick answers from lecture notes, syllabus documents, assignments, and course policies. Manually searching across multiple documents is slow.

This system allows a student to ask natural language questions and receive grounded responses based only on course material.

Example questions:

```text
Explain gradient descent.
Explain it in simple terms.
What is the grading policy?
What is the assignment deadline?
Create quiz questions from the course notes.
```

---

## 3. Final System Architecture

```text
User
 |
 v
Streamlit Chat UI
 |
 v
FastAPI Backend
 |
 v
LangGraph Multi-Agent Workflow
 |
 +--> Query Rewriter Agent
 |       |
 |       +--> Converts follow-up questions into standalone questions
 |
 +--> Router Agent
 |       |
 |       +--> Classifies query as concept, policy, or quiz
 |
 +--> Hybrid Retriever
 |       |
 |       +--> FAISS Semantic Search
 |       +--> BM25 Keyword Search
 |
 +--> Answer Agent / Quiz Agent
 |       |
 |       +--> Generates grounded response using retrieved context
 |
 +--> Verifier Agent
 |       |
 |       +--> Checks if answer is supported by retrieved documents
 |
 v
Final Response with Answer, Sources, Confidence, and Verification
```

---

## 4. Agents

### 4.1 Query Rewriter Agent

Converts conversational follow-up questions into standalone questions.

Example:

```text
Previous question: Explain gradient descent.
Current question: Explain it in simple terms.
Standalone question: Explain gradient descent in simple terms.
```

This improves retrieval quality for conversational RAG.

---

### 4.2 Router Agent

Classifies the user question into one of three routes:

```text
concept -> Concept explanation
policy  -> Deadline, grading, syllabus, assignment policy
quiz    -> Quiz or practice question generation
```

---

### 4.3 Hybrid Retriever

Uses both semantic and keyword retrieval.

Retrieval methods:

```text
FAISS -> Semantic vector similarity search
BM25  -> Keyword-based search
```

This improves retrieval robustness for both conceptual and exact-term queries.

---

### 4.4 Answer Agent

Generates grounded answers using only retrieved context.

Rules:

```text
- Use only course context
- Do not use outside knowledge
- Do not hallucinate
- Return fallback if context is insufficient
```

Fallback:

```text
I could not find enough information in the course material to answer this confidently.
```

---

### 4.5 Quiz Agent

Generates quiz questions directly from retrieved context.

Rules:

```text
- Generate direct quiz questions only
- Do not infer facts
- Do not create calculation or comparison questions
- Every answer must appear in the source context
```

---

### 4.6 Verifier Agent

Checks whether the generated answer is fully supported by retrieved course context.

Returns:

```text
verified: true
verified: false
```

This acts as a hallucination-control layer.

---

## 5. Tech Stack

```text
Python
LangGraph
LangChain
OpenAI API
FAISS
BM25 / rank-bm25
FastAPI
Uvicorn
Streamlit
PyPDF / TextLoader
python-dotenv
```

---

## 6. Project Structure

```text
Education_RAG/
│
├── app.py
├── graph.py
├── ingest.py
├── evaluate.py
├── streamlit_app.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   └── course_notes.txt
│
└── vectorstore/
    └── faiss_index/
```

---

## 7. Setup Instructions

### Step 1: Create virtual environment

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 3: Create `.env`

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

### Step 4: Add course material

Create:

```text
data/course_notes.txt
```

Example content:

```text
Machine learning is a field of artificial intelligence where computers learn patterns from data.

Supervised learning uses labeled data to train models. Examples include classification and regression.

Gradient descent is an optimization algorithm used to minimize the loss function by updating model parameters step by step.

Neural networks are machine learning models inspired by the human brain. They contain layers of connected neurons.

The final assignment deadline is May 20. Late submissions will have a 10 percent penalty.

The grading policy is 40 percent assignments, 30 percent midterm, 20 percent final exam, and 10 percent participation.
```

---

## 8. Build Vector Store

Run ingestion:

```bash
python ingest.py
```

Expected output:

```text
Loading documents...
Splitting documents into chunks...
Created X chunks
Creating embeddings and FAISS vector store...
Vector store saved successfully
```

If you change the source document or chunking strategy, delete the old vector store and rerun ingestion:

```bash
rm -rf vectorstore/faiss_index
python ingest.py
```

---

## 9. Run FastAPI Backend

```bash
uvicorn app:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 10. API Usage

### POST `/ask`

Request:

```json
{
  "session_id": "student_001",
  "question": "Explain gradient descent"
}
```

Response:

```json
{
  "session_id": "student_001",
  "question": "Explain gradient descent",
  "standalone_question": "Explain gradient descent",
  "route": "concept",
  "answer": "Gradient descent is an optimization algorithm used to minimize the loss function by updating model parameters step by step.",
  "verified": true,
  "confidence": "high",
  "retrieval_scores": [0.3605, 0.4270],
  "sources": [
    "Gradient descent is an optimization algorithm used to minimize the loss function..."
  ]
}
```

---

## 11. Chat History Endpoint

### GET `/history/{session_id}`

Example:

```text
http://127.0.0.1:8000/history/student_001
```

Returns session-based chat history.

---

## 12. Run Streamlit UI

Make sure FastAPI is already running.

Then run:

```bash
streamlit run streamlit_app.py
```

Streamlit UI runs at:

```text
http://localhost:8501
```

Demo questions:

```text
Explain gradient descent.
Explain it in simple terms.
What is the grading policy?
What is the assignment deadline?
Create quiz questions.
```

---

## 13. Evaluation

Run:

```bash
python evaluate.py
```

The evaluation script tests:

```text
1. Concept explanation
2. Policy retrieval
3. Deadline retrieval
4. Quiz generation
```

Example output:

```text
Question: Explain gradient descent
Route: concept
Verified: True
Confidence: high

Question: What is the grading policy?
Route: policy
Verified: True
Confidence: high

Question: What is the assignment deadline?
Route: policy
Verified: True
Confidence: high

Question: Create quiz questions
Route: quiz
Verified: True
Confidence: medium
```

---

## 14. Evaluation Strategy

This project uses a lightweight evaluation approach inspired by:

```text
LLM-as-a-judge
RAGAS
DeepEval
TruLens
```

Current implemented evaluation:

```text
- Verifier agent for grounding
- Retrieval confidence score
- Retrieval source inspection
- Route validation
- Golden test questions
```

Metrics to discuss:

```text
Faithfulness        -> Is the answer grounded in retrieved context?
Answer relevance    -> Does the answer address the question?
Context precision   -> Are retrieved chunks useful?
Context recall      -> Did retrieval fetch needed evidence?
Hallucination rate  -> Did the answer add unsupported claims?
Route accuracy      -> Did router classify the query correctly?
```

---

## 15. Important RAG Improvements Implemented

### Chunk Size Optimization

Large chunks initially mixed unrelated concepts like gradient descent and grading policy. Reducing chunk size improved retrieval precision.

### Query Rewriting

Follow-up questions like:

```text
Explain it in simple terms.
```

are rewritten into:

```text
Explain gradient descent in simple terms.
```

### Hybrid Retrieval

Combines:

```text
FAISS semantic retrieval
BM25 keyword retrieval
```

### Confidence Scoring

Uses FAISS retrieval distance scores.

```text
Lower score = stronger semantic match
Higher score = weaker semantic match
```

Example confidence logic:

```text
best_score < 0.45 -> high
best_score < 0.75 -> medium
else              -> low
```

### Verifier Agent

Checks if the generated answer is fully supported by retrieved context.

---

## 16. About

```text
I built a multi-agent conversational RAG application for the education domain using LangGraph and FastAPI. Instead of a single linear RAG chain, I designed specialized agents for query rewriting, routing, hybrid retrieval, answer generation, quiz generation, and verification.

The system supports session-aware memory, so students can ask follow-up questions like “Explain it in simple terms.” A query rewriter agent converts those follow-ups into standalone retrieval-ready questions before retrieval.

For retrieval, I used a hybrid approach combining FAISS semantic search and BM25 keyword search. This improves both conceptual understanding and exact-term matching for syllabus or policy questions.

To reduce hallucinations, the answer and quiz agents are constrained to use only retrieved course context. A verifier agent checks whether the final response is grounded in the retrieved documents. I also expose retrieval scores and confidence labels for observability and debugging.
```

---

## 17. Demo Flow

Recommended demo sequence:

```text
1. Explain gradient descent.
2. Explain it in simple terms.
3. What is the grading policy?
4. What is the assignment deadline?
5. Create quiz questions.
6. Open RAG details in Streamlit.
7. Run python evaluate.py.
```

This demonstrates:

```text
- Memory
- Query rewriting
- Routing
- Hybrid retrieval
- Grounded generation
- Verification
- Confidence scoring
- Evaluation
- UI integration
```

---

## 18. Future Improvements

```text
- Add PDF upload from UI
- Add page-level citations
- Add persistent chat history using PostgreSQL or MongoDB
- Add Redis caching
- Add reranking model
- Add RAGAS / DeepEval automated evaluation
- Add TruLens dashboard
- Add Docker deployment
- Add authentication
- Add teacher/admin dashboard
- Add personalized student learning profiles
```

---

## 19. Final Output

By the end of this project, you have:

```text
- Working FastAPI backend
- Streamlit chat UI
- LangGraph multi-agent workflow
- Query rewriting
- Session memory
- FAISS vector retrieval
- BM25 keyword retrieval
- Hybrid RAG
- Source-grounded answers
- Quiz generation
- Verifier agent
- Retrieval confidence scoring
- Evaluation script
```
