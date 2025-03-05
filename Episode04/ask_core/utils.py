"""Utility functions and helpers for the ask_core system."""

import os
from typing import List, Tuple, Optional
from langchain_core.documents import Document


def format_documents(docs: List[Document]) -> str:
    """Format documents into a context string."""
    return "\n\n".join([doc.page_content for doc in docs])


def display_results(query: str, docs_with_scores: List[Tuple[Document, float]], 
                   answer: str, store_info: dict, llm_info: dict, 
                   preview_bytes: int = 0, verbose: bool = False, 
                   reranking_enabled: bool = False):
    """Display search results in the original ask.py format."""
    # Main answer output (like original ask.py)
    print(f"\nAnswer: {answer}")
    
    # Show source documents with similarity scores (like original ask.py)
    if docs_with_scores:
        print(f"\nReferences: {len(docs_with_scores)} documents")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            doc_id = doc.metadata.get('doc_id', 'unknown')
            source = doc.metadata.get('source', 'unknown')
            
            # Build source info with chunk information
            chunk_index = doc.metadata.get('chunk_index')
            total_chunks = doc.metadata.get('total_chunks')
            if chunk_index is not None and total_chunks is not None:
                source_info = f"from {os.path.basename(source)}, chunk {chunk_index + 1} of {total_chunks}"
            else:
                source_info = f"from {os.path.basename(source)}"
            
            # Handle content preview based on --preview-bytes
            if preview_bytes > 0:
                content = doc.page_content[:preview_bytes].replace('\n', ' ')
                if len(doc.page_content) > preview_bytes:
                    content += "..."
                preview_text = f": {content}"
            else:
                preview_text = ""
            
            # Format score based on store type and reranking
            store_type = store_info.get('store_type', 'faiss')
            if reranking_enabled:
                # Rerank scores are typically between -10 to 10 (higher is better)
                print(f"  {i}. {doc_id} ({source_info}) [rerank: {score:.4f}]{preview_text}")
            elif store_type == "pinecone":
                # Pinecone scores are typically between 0-1 (higher is better)
                print(f"  {i}. {doc_id} ({source_info}) [similarity: {score:.4f}]{preview_text}")
            elif store_type in ["faiss", "chroma"]:
                # FAISS/Chroma scores are distances (lower is better), convert to similarity
                similarity = 1 / (1 + score)  # Convert distance to similarity-like score
                print(f"  {i}. {doc_id} ({source_info}) [similarity: {similarity:.4f}]{preview_text}")
            
            # Add newline between references for better readability (only when showing content)
            if preview_bytes > 0 and i < len(docs_with_scores):
                print()


def print_system_info(store_info: dict, llm_info: dict, embedding_model: str, 
                     reranker_info: Optional[dict] = None, verbose: bool = False):
    """Print system information if verbose mode is enabled."""
    if not verbose:
        return
    
    print(f"*** SYSTEM INFO ***")
    print(f"Store Type: {store_info.get('store_type')}")
    print(f"Store Path: {store_info.get('store_path')}")
    if store_info.get('document_count', -1) >= 0:
        print(f"Documents: {store_info.get('document_count')}")
    
    print(f"Embedding Model: {embedding_model}")
    print(f"LLM Type: {llm_info.get('llm_type')}")
    print(f"LLM Model: {llm_info.get('model_name')}")
    
    if reranker_info and reranker_info.get('available'):
        print(f"Reranker: {reranker_info.get('reranker_type')} ({reranker_info.get('model_name')})")


def validate_environment_variables() -> dict:
    """Validate required environment variables."""
    errors = []
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        errors.append("OPENAI_API_KEY must be set in the environment")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "openai_key_set": bool(openai_key)
    }


def get_store_config(args) -> dict:
    """Get vector store configuration from arguments."""
    if args.store == "faiss":
        return {
            "faiss_path": args.faiss_path
        }
    elif args.store == "pinecone":
        return {
            "pinecone_key": args.pinecone_key,
            "pinecone_index": args.pinecone_index
        }
    elif args.store == "chroma":
        return {
            "chroma_path": args.chroma_path,
            "chroma_index": args.chroma_index
        }
    else:
        raise ValueError(f"Unsupported store type: {args.store}")


def print_usage_examples():
    """Print usage examples for the ask system."""
    examples = """
Examples:
  # Query FAISS index (default)
  ask.py "What is artificial intelligence?"
  
  # Query Pinecone index
  ask.py --store pinecone "What is machine learning?"
  
  # Query Chroma index
  ask.py --store chroma --chroma-path ./my_chroma "Explain neural networks"
  
  # Query with reranking (any store)
  ask.py --rerank "What is deep learning?"
  ask.py --store pinecone --rerank --rerank-type huggingface --rerank-model quality "Explain transformers"
  
  # Show detailed results
  ask.py --rerank --show-rerank-results -v "What is AI?"
  
  # Use different LLM models  
  ask.py --llm-type openai --llm-model gpt-4 "What is AI?"
  ask.py --llm-type huggingface --llm-model gemma-3-1b "What is AI?"
    """
    print(examples)