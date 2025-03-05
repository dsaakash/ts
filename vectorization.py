from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()

def get_tfidf_vectors(text1_list, text2_list):
    """
    Convert text pairs into TF-IDF vectors.
    Returns cosine similarity scores.
    """
    text_pairs = text1_list + text2_list
    tfidf_matrix = vectorizer.fit_transform(text_pairs)

    # Split back into two sets
    text1_vectors = tfidf_matrix[:len(text1_list)]
    text2_vectors = tfidf_matrix[len(text1_list):]

    # Compute cosine similarity
    cosine_similarities = (text1_vectors * text2_vectors.T).toarray().diagonal()
    return cosine_similarities
