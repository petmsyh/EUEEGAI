import numpy as np

def local_similarity_search(query, model, points, limit):
    query_vector = model.encode(query)
    vectors = np.array([p["vector"] for p in points])
    similarities = np.dot(vectors, query_vector) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    )
    top_indices = np.argsort(similarities)[::-1][:limit]
    results = [points[i] for i in top_indices]
    return results