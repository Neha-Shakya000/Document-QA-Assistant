from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_relevant_chunks(query, chunks, top_n=3):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Combine the query and chunks into a single list for vectorization
    all_texts = [query] + chunks

    # Fit and transform the texts to get TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compute cosine similarity between the query (first vector) and all chunks
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Get the indices of the top_n most similar chunks
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]

    # Retrieve the top_n most relevant chunks
    relevant_chunks = [chunks[i] for i in top_indices]

    return relevant_chunks

