import faiss
import numpy as np
from app.utils.embedding_model import get_embedding


import os

documents = []
if os.path.exists("data/documents.txt"):
    with open("data/documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines()]

if not documents:
    documents = ["No documents found. Please add content to data/documents.txt."]

embeddings = np.array([get_embedding(doc) for doc in documents])


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


def semantic_search(query: str):

    query_vector = np.array([get_embedding(query)])

    distances, indices = index.search(query_vector, 3)

    results = [documents[i] for i in indices[0]]

    return results