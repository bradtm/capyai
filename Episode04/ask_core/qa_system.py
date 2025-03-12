"""Main QA system orchestrator for the ask_core system."""

import os
from typing import List, Tuple, Optional, Dict, Any
from langchain_core.documents import Document

from .embeddings import create_embeddings, get_embedding_model_from_metadata
from .vector_stores import VectorStoreManager
from .llm_models import LLMManager
from .rerankers import RerankerManager, check_reranking_dependencies
from .utils import format_documents, print_system_info, expand_context_chunks
from .answer_validator import AnswerValidator


class QASystem:
    """Main QA system that orchestrates all components."""
    
    def __init__(self, store_type: str = "faiss", embedding_model: str = "text-embedding-3-small",
                 llm_type: str = "openai", llm_model: str = "gpt-3.5-turbo",
                 enable_reranking: bool = False, reranker_type: str = "huggingface",
                 reranker_model: str = "quality", expand_context: int = 2, 
                 use_chunks_only: bool = False, enable_answer_validation: bool = True,
                 **store_kwargs):
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
            expand_context: Number of chunks to expand before/after each match (0 to disable)
            use_chunks_only: Whether to disable context expansion
            enable_answer_validation: Whether to enable answer validation against context
            **store_kwargs: Additional arguments for vector store setup
        """
        self.store_type = store_type
        self.embedding_model = embedding_model
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.enable_reranking = enable_reranking
        self.expand_context = expand_context
        self.use_chunks_only = use_chunks_only
        self.enable_answer_validation = enable_answer_validation
        
        # Auto-detect embedding model from metadata if using default
        if embedding_model == "text-embedding-3-small":
            detected_model = None
            if store_type == "faiss":
                detected_model = get_embedding_model_from_metadata(
                    store_type, store_kwargs.get("faiss_path", "faiss_index")
                )
            elif store_type == "chroma":
                detected_model = get_embedding_model_from_metadata(
                    store_type, 
                    store_kwargs.get("chroma_path", "./chroma_db"),
                    store_kwargs.get("collection_name")
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
        
        # Initialize answer validator if enabled
        self.answer_validator = None
        if enable_answer_validation:
            self.answer_validator = AnswerValidator()
    
    def query(self, question: str, top_k: int = 4, rerank_top_k: Optional[int] = None,
              verbose: bool = False, extra_verbose: bool = False, show_rerank_results: bool = False) -> Dict[str, Any]:
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
        
        # Apply context expansion if enabled
        if not self.use_chunks_only and self.expand_context > 0 and docs_with_scores:
            if verbose:
                print(f"*** Expanding context with window size ±{self.expand_context} ***")
            
            if extra_verbose:
                docs_with_scores = self._expand_context_with_detailed_logging(
                    docs_with_scores, question
                )
            else:
                docs_with_scores = expand_context_chunks(
                    docs_with_scores, self.vector_store, self.expand_context
                )
        
        # Generate answer
        docs = [doc for doc, score in docs_with_scores]
        
        if extra_verbose:
            # Generate individual answers per document
            individual_answers = []
            for doc, score in docs_with_scores:
                source = doc.metadata.get('source', 'unknown')
                context_size = len(doc.page_content)
                print(f"Sending {context_size} chars to LLM for {source}")
                
                doc_answer = self.llm_manager.generate_response(doc.page_content, question)
                individual_answers.append(doc_answer)
                
                # Truncate answer for display
                truncated_answer = doc_answer[:100] + "..." if len(doc_answer) > 100 else doc_answer
                print(f"Answer from {source}: {truncated_answer}")
            
            # Generate final merged answer
            context = format_documents(docs)
            context_size = len(context)
            print(f"\nGenerating merged answer from {len(docs)} documents ({context_size} chars total)")
            answer = self.llm_manager.generate_response(context, question)
        else:
            context = format_documents(docs)
            answer = self.llm_manager.generate_response(context, question)
        
        # Validate answer against context if enabled
        validation_details = {}
        if self.answer_validator and docs:
            # Use verbose for validation if either verbose or extra_verbose is enabled
            validation_verbose = verbose or extra_verbose
            is_valid, validated_answer, validation_details = self.answer_validator.validate_answer(
                answer, context, question, verbose=validation_verbose
            )
            
            if not is_valid:
                if verbose:
                    print(f"*** Answer validation failed: {validation_details['final_decision']['reason']} ***")
                answer = validated_answer  # This will be "I don't know"
            elif verbose:
                print("*** Answer validation passed ***")
        
        return {
            "question": question,
            "answer": answer,
            "documents": docs,
            "scores": [score for doc, score in docs_with_scores],
            "docs_with_scores": docs_with_scores,
            "context": context,
            "validation_details": validation_details,
            "metadata": {
                "store_type": self.store_type,
                "embedding_model": self.embedding_model,
                "llm_type": self.llm_type,
                "llm_model": self.llm_model,
                "reranking_enabled": self.enable_reranking and self.reranker is not None,
                "answer_validation_enabled": self.enable_answer_validation and self.answer_validator is not None,
                "num_documents": len(docs)
            }
        }
    
    def _expand_context_with_detailed_logging(self, docs_with_scores: List[Tuple[Document, float]], 
                                            question: str) -> List[Tuple[Document, float]]:
        """
        Expand context with detailed per-document logging similar to cask.py.
        
        Args:
            docs_with_scores: Original documents with scores
            question: The query question
            
        Returns:
            Expanded documents with scores
        """
        from collections import defaultdict
        from .utils import merge_overlapping_ranges
        
        if self.expand_context == 0:
            return docs_with_scores
        
        # Group documents by source file
        docs_by_source = defaultdict(list)
        
        for doc, score in docs_with_scores:
            source = doc.metadata.get('source', 'unknown')
            chunk_index = doc.metadata.get('chunk_index')
            total_chunks = doc.metadata.get('total_chunks')
            
            if chunk_index is not None and total_chunks is not None:
                docs_by_source[source].append({
                    'doc': doc,
                    'score': score,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                })
            else:
                # If no chunk info, keep as-is
                docs_by_source[source].append({
                    'doc': doc,
                    'score': score,
                    'chunk_index': None,
                    'total_chunks': None
                })
        
        expanded_docs = []
        
        # Process each source file with detailed logging
        for source, source_docs in docs_by_source.items():
            print(f"  - Processing {source}")
            
            # Separate docs with and without chunk info
            chunked_docs = [d for d in source_docs if d['chunk_index'] is not None]
            non_chunked_docs = [d for d in source_docs if d['chunk_index'] is None]
            
            # Add non-chunked docs as-is
            for doc_info in non_chunked_docs:
                expanded_docs.append((doc_info['doc'], doc_info['score']))
            
            if not chunked_docs:
                continue
                
            # Sort chunked docs by chunk index
            chunked_docs.sort(key=lambda x: x['chunk_index'])
            
            # Show found chunks (convert to 1-based for display consistency)
            chunk_positions = [d['chunk_index'] + 1 for d in chunked_docs]
            total_chunks = chunked_docs[0]['total_chunks']
            print(f"    - Found {len(chunked_docs)} relevant chunks at positions: {chunk_positions}")
            
            # Create expansion ranges and merge overlapping ones
            ranges = []
            for doc_info in chunked_docs:
                chunk_idx = doc_info['chunk_index']
                total_chunks = doc_info['total_chunks']
                
                # Calculate expansion range
                start_idx = max(0, chunk_idx - self.expand_context)
                end_idx = min(total_chunks - 1, chunk_idx + self.expand_context)
                
                ranges.append({
                    'start': start_idx,
                    'end': end_idx,
                    'original_doc': doc_info['doc'],
                    'score': doc_info['score'],
                    'chunk_index': chunk_idx
                })
            
            # Merge overlapping ranges
            merged_ranges = merge_overlapping_ranges(ranges)
            
            # Show expansion details
            for range_info in merged_ranges:
                start_idx = range_info['start']
                end_idx = range_info['end']
                expanded_chunks = list(range(start_idx, end_idx + 1))
                
                # Convert to 1-based indexing for display consistency
                expanded_chunks_display = [idx + 1 for idx in expanded_chunks]
                
                print(f"    - Expanding to include chunks: {expanded_chunks_display}")
                print(f"    - Retrieved {len(expanded_chunks)} contextual chunks (was getting 1 chunk without expansion)")
                
                # Retrieve chunks in this range
                range_docs = []
                for chunk_idx in expanded_chunks:
                    # Search for chunk by source and chunk_index
                    chunk_results = self.vector_store.vectorstore.get(
                        where={
                            "$and": [
                                {"source": {"$eq": source}},
                                {"chunk_index": {"$eq": chunk_idx}}
                            ]
                        },
                        limit=1
                    )
                    
                    if chunk_results and chunk_results['documents']:
                        doc_content = chunk_results['documents'][0]
                        doc_metadata = chunk_results['metadatas'][0] if chunk_results['metadatas'] else {}
                        
                        # Create Document object
                        chunk_doc = Document(
                            page_content=doc_content,
                            metadata=doc_metadata
                        )
                        range_docs.append(chunk_doc)
                
                # Calculate context size
                context_size = sum(len(doc.page_content) for doc in range_docs)
                # Estimate single chunk size for comparison
                avg_chunk_size = context_size // len(range_docs) if range_docs else 1000
                print(f"    - Context size: {context_size} chars (vs ~{avg_chunk_size} chars for single chunk)")
                
                # Use the score from the best matching chunk in this range
                best_score = max([r['score'] for r in range_info.get('merged_docs', [range_info])])
                
                if range_docs:
                    # Merge all docs in range into one
                    merged_content = "\n\n".join([doc.page_content for doc in range_docs])
                    
                    # Use metadata from the original document that was found, not the first in range
                    original_doc = range_info['original_doc']
                    merged_metadata = original_doc.metadata.copy()
                    merged_metadata['expanded_chunks'] = len(range_docs)
                    merged_metadata['context_size'] = context_size
                    merged_metadata['expanded_range'] = f"{start_idx}-{end_idx}"
                    
                    merged_doc = Document(
                        page_content=merged_content,
                        metadata=merged_metadata
                    )
                    
                    expanded_docs.append((merged_doc, best_score))
        
        return expanded_docs
    
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