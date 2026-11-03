import re


def clean_text(text):
    # Join words broken across PDF lines
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    # Replace line breaks with spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(pages, max_words=150, overlap_words=30):
    """
    Create sentence-aware chunks.
    Page number is preserved with every chunk.
    """

    chunks = []
    chunk_id = 1

    for page_data in pages:

        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        if not text:
            continue

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        current_sentences = []
        current_word_count = 0

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            words = sentence.split()

            # Split unusually long sentences
            if len(words) > max_words:

                if current_sentences:
                    chunk_text = " ".join(current_sentences)

                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": chunk_text
                    })

                    chunk_id += 1
                    current_sentences = []
                    current_word_count = 0

                for start in range(0, len(words), max_words):
                    part = " ".join(words[start:start + max_words])

                    chunks.append({
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": part
                    })

                    chunk_id += 1

                continue

            if current_word_count + len(words) > max_words:

                chunk_text = " ".join(current_sentences)

                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

                # Keep a small overlap from the previous chunk
                overlap = []
                overlap_count = 0

                for old_sentence in reversed(current_sentences):

                    old_count = len(old_sentence.split())

                    if overlap_count + old_count > overlap_words:
                        break

                    overlap.insert(0, old_sentence)
                    overlap_count += old_count

                current_sentences = overlap
                current_word_count = overlap_count

            current_sentences.append(sentence)
            current_word_count += len(words)

        if current_sentences:

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": " ".join(current_sentences)
            })

            chunk_id += 1

    return chunks
    