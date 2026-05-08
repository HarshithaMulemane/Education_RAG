from graph import edu_rag_graph


test_questions = [
    {
        "question": "Explain gradient descent",
        "expected_topic": "Gradient descent minimizes loss function"
    },
    {
        "question": "What is the grading policy?",
        "expected_topic": "40 percent assignments, 30 percent midterm, 20 percent final exam, 10 percent participation"
    },
    {
        "question": "What is the assignment deadline?",
        "expected_topic": "May 20"
    },
    {
        "question": "Create 5 quiz questions",
        "expected_topic": "quiz questions from course content"
    }
]


def run_eval():
    results = []

    for item in test_questions:
        initial_state = {
            "question": item["question"],
            "standalone_question": "",
            "chat_history": "",
            "route": "",
            "documents": [],
            "answer": "",
            "verified": False,
            "sources": [],
            "retrieval_scores": [],
            "confidence": ""
        }

        result = edu_rag_graph.invoke(initial_state)

        results.append({
            "question": item["question"],
            "expected_topic": item["expected_topic"],
            "answer": result["answer"],
            "route": result["route"],
            "verified": result["verified"],
            "confidence": result["confidence"],
            "retrieval_scores": result["retrieval_scores"],
            "sources": result["sources"]
        })

    return results


if __name__ == "__main__":
    eval_results = run_eval()

    for i, result in enumerate(eval_results, start=1):
        print("=" * 80)
        print(f"Test Case {i}")
        print("Question:", result["question"])
        print("Expected:", result["expected_topic"])
        print("Route:", result["route"])
        print("Verified:", result["verified"])
        print("Confidence:", result["confidence"])
        print("Scores:", result["retrieval_scores"])
        print("Answer:", result["answer"])