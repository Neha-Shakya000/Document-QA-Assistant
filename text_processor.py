import nltk
import re

from langchain_core.documents import Document


def get_sentences(text):

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        return nltk.sent_tokenize(text)

    except Exception:
        return re.split(r'(?<=[.!?])\s+', text)


def create_chunks(pages, max_words=180, overlap_sentences=2):

    documents = []
    chunk_id = 1

    for page_data in pages:

        page_number = page_data["page"]
        text = page_data["text"]

        # Clean unnecessary spaces and line breaks
        text = re.sub(r'\s+', ' ', text).strip()

        sentences = get_sentences(text)

        current_sentences = []
        current_words = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            words = sentence.split()

            # Split unusually long sentences
            if len(words) > max_words:

                if current_sentences:

                    documents.append(
                        Document(
                            page_content=" ".join(current_sentences),
                            metadata={
                                "page": page_number,
                                "chunk_id": chunk_id
                            }
                        )
                    )

                    chunk_id += 1
                    current_sentences = []
                    current_words = 0

                for start in range(0, len(words), max_words):

                    part = " ".join(
                        words[start:start + max_words]
                    )

                    documents.append(
                        Document(
                            page_content=part,
                            metadata={
                                "page": page_number,
                                "chunk_id": chunk_id
                            }
                        )
                    )

                    chunk_id += 1

                continue

            if current_words + len(words) > max_words:

                documents.append(
                    Document(
                        page_content=" ".join(current_sentences),
                        metadata={
                            "page": page_number,
                            "chunk_id": chunk_id
                        }
                    )
                )

                chunk_id += 1

                current_sentences = current_sentences[
                    -overlap_sentences:
                ]

                current_words = sum(
                    len(s.split()) for s in current_sentences
                )

            current_sentences.append(sentence)
            current_words += len(words)

        if current_sentences:

            documents.append(
                Document(
                    page_content=" ".join(current_sentences),
                    metadata={
                        "page": page_number,
                        "chunk_id": chunk_id
                    }
                )
            )

            chunk_id += 1

    return documents

