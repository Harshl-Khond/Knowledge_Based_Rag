from fastembed import TextEmbedding

_model = None

def get_model():
    global _model
    if _model is None:
        # Explicitly setting cache_dir to /tmp which is writable on Vercel
        _model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/tmp/fastembed_cache")
    return _model


def get_embedding(text: str):
    """Generate embedding using fastembed (lightweight, no HuggingFace needed)."""
    model = get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()