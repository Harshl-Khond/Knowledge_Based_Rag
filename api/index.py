import os
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"

from app.main import app
