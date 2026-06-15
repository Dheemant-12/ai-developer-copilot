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