"""
Modular reranking implementation for RAG systems.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)


@dataclass
class RerankResult:
    """Result from reranking operation."""
    document: Any  # LangChain document
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int


class BaseReranker(ABC):
    """Abstract base class for all rerankers."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        docs_with_scores: List[Tuple[Any, float]], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query."""
        pass
    
    @abstractmethod
    def rerank_simple(
        self, 
        query: str, 
        docs: List[str], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """Simple reranking interface for text documents."""
        pass


class HuggingFaceReranker(BaseReranker):
    """
    HuggingFace-based document reranker using cross-encoder models.
    
    Popular models:
    - ms-marco-MiniLM-L-6-v2: Fast, good general performance
    - ms-marco-MiniLM-L-12-v2: Better quality, slightly slower
    - bge-reranker-base: High quality Chinese/English support
    - bge-reranker-large: Best quality but slower
    """
    
    DEFAULT_MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", 
        "quality": "BAAI/bge-reranker-base",
        "best": "BAAI/bge-reranker-large"
    }
    
    def __init__(self, model_name: str = "balanced", device: Optional[str] = None):
        """
        Initialize the reranker.
        
        Args:
            model_name: Model name or preset (fast/balanced/quality/best)
            device: Device to run on (cuda/cpu/auto). Auto-detects if None.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        # Handle preset model names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        self.device = device
        self._model = None
        self._is_loaded = False
    
    def _load_model(self, verbose=False):
        """Lazy load the model on first use."""
        if not self._is_loaded:
            try:
                if verbose:
                    print(f"*** Loading reranker model: {self.model_name} ***")
                self._model = CrossEncoder(self.model_name, device=self.device, trust_remote_code=True)
                self._is_loaded = True
                if verbose:
                    print(f"*** Reranker model loaded successfully ***")
            except Exception as e:
                raise RuntimeError(f"Failed to load reranker model {self.model_name}: {e}")
    
    def rerank(
        self, 
        query: str, 
        docs_with_scores: List[Tuple[Any, float]], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            docs_with_scores: List of (document, original_score) tuples
            top_k: Number of top documents to return. If None, returns all.
            
        Returns:
            List of RerankResult objects sorted by rerank score (descending)
        """
        if not docs_with_scores:
            return []
        
        self._load_model(verbose=verbose)
        
        # Prepare query-document pairs for cross-encoder
        pairs = []
        documents = []
        original_scores = []
        
        for doc, score in docs_with_scores:
            # Use page_content for reranking
            doc_text = getattr(doc, 'page_content', str(doc))
            pairs.append([query, doc_text])
            documents.append(doc)
            original_scores.append(score)
        
        # Get reranking scores
        try:
            rerank_scores = self._model.predict(pairs)
        except Exception as e:
            print(f"Warning: Reranking failed ({e}), returning original order")
            # Fallback to original order if reranking fails
            return [
                RerankResult(
                    document=doc,
                    original_score=score,
                    rerank_score=0.0,
                    original_rank=i,
                    new_rank=i
                )
                for i, (doc, score) in enumerate(docs_with_scores)
            ]
        
        # Create results with both scores
        results = []
        for i, (doc, orig_score, rerank_score) in enumerate(zip(documents, original_scores, rerank_scores)):
            results.append(RerankResult(
                document=doc,
                original_score=orig_score,
                rerank_score=float(rerank_score),
                original_rank=i,
                new_rank=-1  # Will be set after sorting
            ))
        
        # Sort by rerank score (descending)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Update new ranks
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def rerank_simple(
        self, 
        query: str, 
        docs_with_scores: List[Tuple[Any, float]], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[Tuple[Any, float]]:
        """
        Simple reranking that returns docs with rerank scores.
        
        Args:
            query: Search query
            docs_with_scores: List of (document, original_score) tuples
            top_k: Number of top documents to return
            
        Returns:
            List of (document, rerank_score) tuples sorted by relevance
        """
        results = self.rerank(query, docs_with_scores, top_k, verbose=verbose)
        return [(result.document, result.rerank_score) for result in results]


class Qwen3Reranker(BaseReranker):
    """
    Qwen3-based document reranker using causal language models.
    
    Uses Qwen3-Reranker models that generate yes/no tokens for relevance scoring.
    """
    
    DEFAULT_MODELS = {
        "qwen3-0.6b": "Qwen/Qwen3-Reranker-0.6B",
        "qwen3-4b": "Qwen/Qwen3-Reranker-4B", 
        "qwen3-8b": "Qwen/Qwen3-Reranker-8B"
    }
    
    def __init__(self, model_name: str = "qwen3-8b", device: Optional[str] = None):
        """
        Initialize the Qwen3 reranker.
        
        Args:
            model_name: Model name or preset (qwen3-0.6b/qwen3-4b/qwen3-8b)
            device: Device to run on (cuda/cpu/auto). Auto-detects if None.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Qwen3 reranking. "
                "Install with: pip install transformers torch"
            )
        
        # Handle preset model names
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]
        
        self.model_name = model_name
        # Apple Silicon M2 optimization: prefer MPS (Metal) over CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda" 
            else:
                self.device = "cpu"
        else:
            self.device = device
        self._tokenizer = None
        self._model = None
        self._is_loaded = False
    
    def _load_model(self, verbose=False):
        """Lazy load the model on first use."""
        if not self._is_loaded:
            try:
                if verbose:
                    print(f"*** Loading Qwen3 reranker model: {self.model_name} ***")
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Load model with appropriate settings for different devices
                if self.device == "mps":
                    # MPS (Apple Silicon Metal) settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    ).to(self.device)
                elif self.device == "cuda":
                    # CUDA GPU settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        trust_remote_code=True
                    )
                else:
                    # CPU settings
                    self._model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    ).to(self.device)
                self._model.eval()
                self._is_loaded = True
                
                if verbose:
                    print(f"*** Qwen3 reranker model loaded successfully ***")
            except Exception as e:
                raise RuntimeError(f"Failed to load Qwen3 reranker model {self.model_name}: {e}")
    
    def _score_pair(self, query: str, passage: str) -> float:
        """Score a single query-passage pair."""
        # Format the input as per Qwen3 reranker requirements
        prompt = f"Query: {query}\nPassage: {passage}\nInstructional: Does the passage answer the query? Answer yes or no.\n"
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0, -1]
            
            # Get logits for yes/no tokens
            yes_token_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
            
            yes_logit = logits[yes_token_id].item()
            no_logit = logits[no_token_id].item()
            
            # Convert to probability-like score (0-1 range)
            score = torch.softmax(torch.tensor([no_logit, yes_logit]), dim=0)[1].item()
            return score
    
    def rerank(
        self, 
        query: str, 
        docs_with_scores: List[Tuple[Any, float]], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[RerankResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            docs_with_scores: List of (document, original_score) tuples
            top_k: Number of top documents to return. If None, returns all.
            verbose: Whether to show verbose output
            
        Returns:
            List of RerankResult objects sorted by relevance
        """
        self._load_model(verbose)
        
        if verbose:
            print(f"*** Reranking {len(docs_with_scores)} documents with Qwen3 ***")
        
        # Create pairs for scoring
        results = []
        for i, (doc, original_score) in enumerate(docs_with_scores):
            try:
                rerank_score = self._score_pair(query, doc.page_content)
                results.append(RerankResult(
                    document=doc,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    original_rank=i,
                    new_rank=0  # Will be updated after sorting
                ))
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to score document {i}: {e}")
                # Use original score as fallback
                results.append(RerankResult(
                    document=doc,
                    original_score=original_score,
                    rerank_score=original_score,
                    original_rank=i,
                    new_rank=0
                ))
        
        # Sort by rerank score (descending)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Update new ranks
        for new_rank, result in enumerate(results):
            result.new_rank = new_rank
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def rerank_simple(
        self, 
        query: str, 
        docs: List[str], 
        top_k: Optional[int] = None,
        verbose: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Simple reranking that returns docs with rerank scores.
        
        Args:
            query: Search query
            docs: List of document strings
            top_k: Number of top documents to return
            verbose: Whether to show verbose output
            
        Returns:
            List of (document, rerank_score) tuples sorted by relevance
        """
        # Convert strings to dummy document objects
        class DummyDoc:
            def __init__(self, content):
                self.page_content = content
        
        docs_with_scores = [(DummyDoc(doc), 1.0) for doc in docs]
        results = self.rerank(query, docs_with_scores, top_k, verbose=verbose)
        return [(result.document.page_content, result.rerank_score) for result in results]


def create_reranker(
    reranker_type: str = "huggingface", 
    model_name: str = "balanced",
    device: Optional[str] = None
) -> BaseReranker:
    """
    Factory function to create appropriate reranker based on type.
    
    Args:
        reranker_type: Type of reranker ("huggingface" or "qwen3")
        model_name: Model name or preset
        device: Device to run on
        
    Returns:
        BaseReranker instance
        
    Raises:
        ValueError: If reranker_type is not supported
    """
    if reranker_type.lower() == "huggingface":
        return HuggingFaceReranker(model_name, device)
    elif reranker_type.lower() == "qwen3":
        return Qwen3Reranker(model_name, device)
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}. Use 'huggingface' or 'qwen3'")


def get_available_models() -> dict:
    """Get available preset models for all reranker types."""
    models = {
        "huggingface": HuggingFaceReranker.DEFAULT_MODELS.copy(),
        "qwen3": Qwen3Reranker.DEFAULT_MODELS.copy()
    }
    return models


def test_reranker(model_name: str = "fast") -> bool:
    """Test if reranker can be loaded and used."""
    try:
        reranker = HuggingFaceReranker(model_name)
        
        # Simple test with dummy data
        class DummyDoc:
            def __init__(self, content):
                self.page_content = content
        
        docs = [
            (DummyDoc("This is about machine learning algorithms"), 0.8),
            (DummyDoc("This discusses cooking recipes"), 0.7),
            (DummyDoc("This explains neural networks"), 0.6)
        ]
        
        results = reranker.rerank_simple("machine learning", docs, top_k=2)
        return len(results) == 2
        
    except Exception as e:
        print(f"Reranker test failed: {e}")
        return False
