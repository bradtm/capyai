# ask_core - Modular RAG system supporting FAISS, Pinecone, and Chroma vector stores
"""
A modular retrieval-augmented generation system that supports multiple vector stores
and LLM providers with optional reranking capabilities.
"""

import os

# Fix HuggingFace tokenizers warning when using reranking
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "1.0.0"