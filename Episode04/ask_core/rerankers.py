"""Reranking functionality for the ask_core system."""

from typing import List, Tuple, Optional
from langchain_core.documents import Document

# Optional reranking imports
try:
    from rerank_core import create_reranker, get_available_models
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False


class RerankerManager:
    """Manager for document reranking functionality."""
    
    def __init__(self, reranker_type: str = "huggingface", model_name: str = "quality", device: Optional[str] = None):
        self.reranker_type = reranker_type
        self.model_name = model_name
        self.device = self._auto_detect_device() if device is None else device
        self.reranker = None
        
        if RERANKING_AVAILABLE:
            self._setup_reranker()
    
    def _auto_detect_device(self) -> str:
        """Auto-detect the best available device for HuggingFace models."""
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon (M1/M2/M3) GPU
            else:
                device = "cpu"
        except ImportError:
            # If torch is not available, fall back to CPU
            device = "cpu"
        
        return device
    
    def _setup_reranker(self):
        """Setup the reranker based on type and model."""
        try:
            # Pass device parameter if using HuggingFace models
            if self.reranker_type in ["huggingface", "qwen3", "mlx-qwen3"]:
                self.reranker = create_reranker(
                    reranker_type=self.reranker_type,
                    model_name=self.model_name,
                    device=self.device
                )
            else:
                # For other reranker types that don't need device specification
                self.reranker = create_reranker(
                    reranker_type=self.reranker_type,
                    model_name=self.model_name
                )
        except Exception as e:
            raise RuntimeError(f"Error initializing reranker: {e}")
    
    def is_available(self) -> bool:
        """Check if reranking is available."""
        return RERANKING_AVAILABLE and self.reranker is not None
    
    def rerank_documents(self, query: str, docs_with_scores: List[Tuple[Document, float]], 
                        top_k: int = 4, verbose: bool = False) -> List[Tuple[Document, float]]:
        """Rerank documents and return the top-k results."""
        if not self.is_available():
            raise RuntimeError("Reranker not available or not initialized")
        
        if not docs_with_scores:
            return []
        
        if verbose:
            print(f"*** Reranking {len(docs_with_scores)} documents ***")
        
        # Rerank documents
        reranked_results = self.reranker.rerank(
            query, docs_with_scores, top_k=top_k, verbose=verbose
        )
        
        # Convert back to docs_with_scores format with rerank scores
        reranked_docs = [(result.document, result.rerank_score) for result in reranked_results]
        
        if verbose:
            print(f"*** Reranking complete, using top {len(reranked_docs)} documents ***")
        
        return reranked_docs
    
    def show_reranking_results(self, query: str, docs_with_scores: List[Tuple[Document, float]], 
                              top_k: int = 4) -> List[Tuple[Document, float]]:
        """Rerank documents and show detailed results."""
        if not self.is_available():
            return docs_with_scores
        
        reranked_results = self.reranker.rerank(
            query, docs_with_scores, top_k=top_k, verbose=True
        )
        
        print("\n*** RERANKING RESULTS ***")
        print(f"Original → New | Original Score → Rerank Score | Doc ID")
        print("-" * 120)
        
        for result in reranked_results:
            doc_id = result.document.metadata.get('doc_id', 'unknown')
            rank_change = f"{result.original_rank + 1:2d} → {result.new_rank + 1:2d}"
            score_change = f"{result.original_score:6.3f} → {result.rerank_score:6.3f}"
            
            # Show rank change direction
            if result.new_rank < result.original_rank:
                direction = "↑"
            elif result.new_rank > result.original_rank:
                direction = "↓"
            else:
                direction = "="
            
            print(f"{rank_change} {direction:1s} | {score_change:20s} | {doc_id}")
        
        return [(result.document, result.rerank_score) for result in reranked_results]
    
    def get_reranker_info(self) -> dict:
        """Get information about the reranker."""
        info = {
            "reranker_type": self.reranker_type,
            "model_name": self.model_name,
            "device": self.device,
            "available": self.is_available(),
            "reranking_core_available": RERANKING_AVAILABLE
        }
        
        if RERANKING_AVAILABLE:
            try:
                info["available_models"] = get_available_models()
            except:
                info["available_models"] = "unknown"
        
        return info


def check_reranking_dependencies() -> dict:
    """Check which reranking dependencies are available."""
    return {
        "reranking_core_available": RERANKING_AVAILABLE,
        "error_message": None if RERANKING_AVAILABLE else (
            "Reranking dependencies not installed. "
            "HuggingFace: pip install sentence-transformers, "
            "Qwen3: pip install transformers torch, "
            "MLX-Qwen3 (Apple Silicon): pip install mlx mlx-lm"
        )
    }
