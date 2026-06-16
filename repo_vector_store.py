import faiss
import pickle
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

REPO_INDEX_FILE = (
    "repo_faiss_index.bin"
)

REPO_CHUNKS_FILE = (
    "repo_chunks.pkl"
)


def create_repo_vector_store(
    chunks
):

    embeddings = model.encode(
        chunks
    )

    embeddings = np.array(
        embeddings
    ).astype(
        "float32"
    )

    dimension = (
        embeddings.shape[1]
    )

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    faiss.write_index(
        index,
        REPO_INDEX_FILE
    )

    with open(
        REPO_CHUNKS_FILE,
        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    return len(
        chunks
    )
def load_repo_vector_store():

    try:

        index = faiss.read_index(
            REPO_INDEX_FILE
        )

        with open(
            REPO_CHUNKS_FILE,
            "rb"
        ) as file:

            chunks = pickle.load(
                file
            )

        return (
            index,
            chunks
        )

    except Exception:

        return (
            None,
            None
        )
def repo_semantic_search(
    query,
    top_k=5
):

    index, chunks = (
        load_repo_vector_store()
    )

    if index is None:

        return []

    query_embedding = (
        model.encode(
            [query]
        )
    )

    query_embedding = (
        np.array(
            query_embedding
        ).astype(
            "float32"
        )
    )

    distances, indices = (
        index.search(
            query_embedding,
            top_k
        )
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < len(
            chunks
        ):

            results.append(
                {
                    "chunk": chunks[idx],
                    "distance": float(
                        distance
                    )
                }
            )

    return results    