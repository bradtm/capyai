"""Main QA system orchestrator for the ask_core system."""

import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from .embeddings import create_embeddings, get_embedding_model_from_metadata
from .vector_stores import VectorStoreManager
from .llm_models import LLMManager
from .rerankers import RerankerManager, check_reranking_dependencies
from .utils import format_documents, print_system_info


class QASystem:
    """Main QA system that orchestrates all components."""
    
    def __init__(self, store_type: str = "faiss", embedding_model: str = "text-embedding-3-small",
                 llm_type: str = "openai", llm_model: str = "gpt-3.5-turbo",
                 enable_reranking: bool = False, reranker_type: str = "huggingface",
                 reranker_model: str = "quality", **store_kwargs):
        """
        Initialize the QA system with all components.
        
        Args:
            store_type: Type of vector store ("faiss", "pinecone", "chroma")
            embedding_model: Embedding model to use
            llm_type: Type of LLM ("openai", "huggingface", "ollama")  
            llm_model: LLM model name
            enable_reranking: Whether to enable reranking
            reranker_type: Type of reranker
            reranker_model: Reranker model name
            **store_kwargs: Additional arguments for vector store setup
        """
        self.store_type = store_type
        self.embedding_model = embedding_model
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.enable_reranking = enable_reranking
        
        # Auto-detect embedding model from metadata if using default
        if embedding_model == "text-embedding-3-small" and store_type == "faiss":
            detected_model = get_embedding_model_from_metadata(
                store_type, store_kwargs.get("faiss_path", "faiss_index")
            )
            if detected_model:
                self.embedding_model = detected_model
        
        # Initialize components
        self.embeddings = create_embeddings(self.embedding_model)
        self.vector_store = VectorStoreManager(store_type, self.embeddings, **store_kwargs)
        self.llm_manager = LLMManager(llm_type, llm_model)
        
        # Initialize reranker if enabled
        self.reranker = None
        if enable_reranking:
            deps = check_reranking_dependencies()
            if deps["reranking_core_available"]:
                self.reranker = RerankerManager(reranker_type, reranker_model)
            else:
                raise RuntimeError(deps["error_message"])
    
    def query(self, question: str, top_k: int = 4, rerank_top_k: Optional[int] = None,
              verbose: bool = False, show_rerank_results: bool = False) -> Dict[str, Any]:
        """
        Query the QA system and return results.
        
        Args:
            question: Question to ask
            top_k: Number of documents to retrieve initially
            rerank_top_k: Number of documents after reranking (defaults to top_k)
            verbose: Whether to print verbose output
            show_rerank_results: Whether to show detailed reranking results
            
        Returns:
            Dictionary containing answer, documents, scores, and metadata
        """
        if rerank_top_k is None:
            rerank_top_k = top_k
        
        # Print system info if verbose
        if verbose:
            store_info = self.vector_store.get_store_info()
            llm_info = self.llm_manager.get_model_info()
            reranker_info = self.reranker.get_reranker_info() if self.reranker else None
            print_system_info(store_info, llm_info, self.embedding_model, reranker_info, verbose)
            
            print(f"*** Searching for: {question} ***")
        
        # Retrieve documents
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=top_k)
        
        # Apply reranking if enabled
        if self.reranker and self.reranker.is_available() and docs_with_scores:
            if show_rerank_results and verbose:
                docs_with_scores = self.reranker.show_reranking_results(
                    question, docs_with_scores, rerank_top_k
                )
            else:
                docs_with_scores = self.reranker.rerank_documents(
                    question, docs_with_scores, rerank_top_k, verbose
                )
        
        # Generate answer
        docs = [doc for doc, score in docs_with_scores]
        context = format_documents(docs)
        answer = self.llm_manager.generate_response(context, question)
        
        return {
            "question": question,
            "answer": answer,
            "documents": docs,
            "scores": [score for doc, score in docs_with_scores],
            "docs_with_scores": docs_with_scores,
            "context": context,
            "metadata": {
                "store_type": self.store_type,
                "embedding_model": self.embedding_model,
                "llm_type": self.llm_type,
                "llm_model": self.llm_model,
                "reranking_enabled": self.enable_reranking and self.reranker is not None,
                "num_documents": len(docs)
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        store_info = self.vector_store.get_store_info()
        llm_info = self.llm_manager.get_model_info()
        
        info = {
            "store_info": store_info,
            "llm_info": llm_info,
            "embedding_model": self.embedding_model,
            "reranking_enabled": self.enable_reranking
        }
        
        if self.reranker:
            info["reranker_info"] = self.reranker.get_reranker_info()
        
        return info
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that all components are properly set up."""
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check vector store
        try:
            store_info = self.vector_store.get_store_info()
            if store_info["document_count"] == 0:
                validation["warnings"].append("Vector store appears to be empty")
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"Vector store error: {str(e)}")
        
        # Check LLM
        try:
            llm_info = self.llm_manager.get_model_info()
            if not llm_info["has_modular_llm"] and not llm_info["has_legacy_model"]:
                validation["valid"] = False
                validation["errors"].append("No LLM model configured")
        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(f"LLM error: {str(e)}")
        
        # Check reranker if enabled
        if self.enable_reranking:
            if not self.reranker or not self.reranker.is_available():
                validation["warnings"].append("Reranking enabled but reranker not available")
        
        return validation