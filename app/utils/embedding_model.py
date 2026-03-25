from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")


def get_embedding(text: str):
    """Generate embedding using fastembed (lightweight, no HuggingFace needed)."""
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()