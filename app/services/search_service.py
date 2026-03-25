import faiss
import numpy as np
from app.utils.embedding_model import get_embedding


import os

_index = None
_documents = None

def load_data():
    global _documents
    if _documents is None:
        if os.path.exists("data/documents.txt"):
            with open("data/documents.txt", "r", encoding="utf-8") as f:
                _documents = [line.strip() for line in f.readlines()]
        if not _documents:
            _documents = ["No documents found. Please add content to data/documents.txt."]
    return _documents

def get_index():
    global _index
    if _index is None:
        docs = load_data()
        embeddings = np.array([get_embedding(doc) for doc in docs])
        dimension = embeddings.shape[1]
        _index = faiss.IndexFlatL2(dimension)
        _index.add(embeddings)
    return _index


def semantic_search(query: str):
    index = get_index()
    docs = load_data()
    
    query_vector = np.array([get_embedding(query)])
    distances, indices = index.search(query_vector, 3)
    results = [docs[i] for i in indices[0]]

    return results