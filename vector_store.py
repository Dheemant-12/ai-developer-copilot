import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

vector_db = None
stored_chunks = []


def create_vector_store(chunks):

    global vector_db
    global stored_chunks

    stored_chunks = chunks

    embeddings = embedding_model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    vector_db = faiss.IndexFlatL2(
        dimension
    )

    vector_db.add(
        embeddings
    )

    return len(chunks)


def semantic_search(
    query,
    top_k=3
):

    global vector_db
    global stored_chunks

    if vector_db is None:
        return []

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = vector_db.search(
        query_embedding,
        top_k
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < len(stored_chunks):

            confidence = max(
                0,
                round(
                    100 - float(distance),
                    2
                )
            )

            results.append(
                {
                    "chunk": stored_chunks[idx],
                    "chunk_id": idx + 1,
                    "distance": float(distance),
                    "confidence": confidence
                }
            )

    return results