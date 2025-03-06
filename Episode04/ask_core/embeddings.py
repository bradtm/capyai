"""Embedding providers for the ask_core system."""

import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings


class OllamaEmbeddings(Embeddings):
    """Ollama embeddings using local models."""
    
    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip('/')
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama server not responding")
        except requests.exceptions.RequestException:
            raise RuntimeError("Ollama server not running. Start with: ollama serve")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents using Ollama."""
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Ollama."""
        response = requests.post(
            f"{self.base_url}/api/embeddings", 
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]


def create_embeddings(model_name: str) -> Embeddings:
    """Create embeddings instance based on model name."""
    if model_name in ["bge-m3", "nomic-embed-text"]:
        return OllamaEmbeddings(model=model_name)
    else:
        return OpenAIEmbeddings(model=model_name)


def get_embedding_model_from_metadata(store_type: str, store_path: str, collection_name: str = None) -> str:
    """Try to detect embedding model from vector store metadata."""
    if store_type == "faiss":
        metadata_file = os.path.join(store_path, "index_metadata.json")
        if os.path.exists(metadata_file):
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('embedding_model')
            except Exception:
                pass
    elif store_type == "chroma":
        try:
            import chromadb
            client = chromadb.PersistentClient(path=store_path)
            if collection_name:
                collection = client.get_collection(collection_name)
                metadata = collection.metadata or {}
                embedding_model = metadata.get('embedding_model')
                if embedding_model:
                    return embedding_model
            else:
                # If no collection name specified, try to find collections with metadata
                collections = client.list_collections()
                for collection_info in collections:
                    try:
                        collection = client.get_collection(collection_info.name)
                        metadata = collection.metadata or {}
                        embedding_model = metadata.get('embedding_model')
                        if embedding_model:
                            return embedding_model
                    except Exception:
                        continue
        except Exception:
            pass
    return None