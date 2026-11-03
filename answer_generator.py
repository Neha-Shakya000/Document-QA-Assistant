from transformers import pipeline


QA_MODEL = "deepset/tinyroberta-squad2"

NOT_FOUND = "Information not found in provided document"


def load_qa_model():

    qa_model = pipeline(
        "question-answering",
        model=QA_MODEL,
        tokenizer=QA_MODEL,
        device=-1
    )

    return qa_model


def generate_answer(question, results, qa_model, threshold=0.10):

    if not results or not question.strip():

        return {
            "answer": NOT_FOUND,
            "source": None,
            "page": None,
            "found": False
        }

    best_answer = None

    for result in results:

        context = result["chunk"]

        try:

            prediction = qa_model(
                question=question,
                context=context,
                handle_impossible_answer=True
            )

            answer = prediction.get("answer", "").strip()
            score = float(prediction.get("score", 0))

            if not answer:
                continue

            if score < threshold:
                continue

            if best_answer is None or score > best_answer["score"]:

                best_answer = {
                    "answer": answer,
                    "source": context,
                    "page": result["page"],
                    "score": score
                }

        except Exception:
            continue

    if best_answer is None:

        return {
            "answer": NOT_FOUND,
            "source": None,
            "page": None,
            "found": False
        }

    return {
        "answer": best_answer["answer"],
        "source": best_answer["source"],
        "page": best_answer["page"],
        "score": best_answer["score"],
        "found": True
    }
