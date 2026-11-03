from document_reader import extract_pages
from text_processor import create_chunks
from retriever import retrieve_relevant_chunks
from answer_generator import generate_answer


pdf_path = "sample.pdf"

# 1. Read document
pages = extract_pages(pdf_path)

# 2. Create chunks
chunks = create_chunks(pages)

print("Total Pages:", len(pages))
print("Total Chunks:", len(chunks))

# 3. Ask question
question = input("\nEnter your question: ").strip()

# 4. Retrieve relevant chunks
results = retrieve_relevant_chunks(
    question,
    chunks,
    top_k=5
)

print("\nRETRIEVED CHUNKS:")

for result in results:

    print("\nChunk:", result["chunk_id"])
    print("Page:", result["page"])
    print("Similarity Score:", round(result["score"], 4))
    print("Content:", result["chunk"])

# 5. Generate answer
response = generate_answer(question, results)

print("\nAI ANSWER:")
print(response["answer"])

print("\nSUPPORTING SOURCE:")
print(response["source"])

print("\nSOURCE PAGE:")
print(response["page"])

print("\nANSWER FOUND:")
print(response["found"])

